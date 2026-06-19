import argparse
import random
import datetime
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import BertConfig
import numpy as np
from data_loader import *
from model import Model
from evaluation import *
import os
import time
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters for model
    parser.add_argument('-dn', '--dataset_name', type=str, default='chart_check')
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('-bert', '--bert_version', type=str, default='base', choices=['base', 'sci', 'pubmed', 'multilingual', 'large'])
    parser.add_argument('-vit', '--vit_version', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('-t5', '--t5_version', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('-ep', '--evidence_provided', type=str, default='gold', choices=['gold', 'retrieved'])
    parser.add_argument('-ne', '--num_epochs', type=int, default=100)  # for small dataset (ai_chart_claim), set 100; for large dataset, set 30
    parser.add_argument('-ls', '--log_steps', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=4)
    parser.add_argument('-ml', '--max_text_length', type=int, default=128)
    parser.add_argument('-nt', '--num_sampled_evid_texts', type=int, default=5)
    parser.add_argument('-ni', '--num_sampled_images', type=int, default=1)
    parser.add_argument('-mt', '--max_new_tokens', type=int, default=128)
    parser.add_argument('-l', '--lambda_exp_reg', type=float, default=0.5)

    # hyperparameters for training
    parser.add_argument('-ddp', '--distributed_training', type=bool, default=True)
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='used only when ddp is False')
    parser.add_argument('-rs', '--random_seed', type=int, default=519)

    return parser.parse_args()


def set_random_seed(random_seed):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


def cleanup():

    dist.destroy_process_group()


def load_data(args):

    if args.bert_version == 'sci':
        args.bert_pretrained_model_name_or_path = 'allenai/scibert_scivocab_uncased'
    elif args.bert_version == 'pubmed':
        args.bert_pretrained_model_name_or_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif args.bert_version == 'base':
        args.bert_pretrained_model_name_or_path = 'bert-base-uncased'
    elif args.bert_version == 'large':
        args.bert_pretrained_model_name_or_path = 'bert-large-cased'
    elif args.bert_version == 'multilingual':
        args.bert_pretrained_model_name_or_path = 'bert-base-multilingual-cased'

    if args.vit_version == 'base':
        args.vit_pretrained_model_name_or_path = 'google/vit-base-patch16-224-in21k'
    elif args.vit_version == 'large':
        args.vit_pretrained_model_name_or_path = 'google/vit-large-patch16-224-in21k'

    if args.t5_version == 'base':
        args.t5_pretrained_model_name_or_path = 't5-base'
    elif args.t5_version == 'large':
        args.t5_pretrained_model_name_or_path = 't5-large'

    data_center = DataCenter(args)
    training_data, test_data = Data(data_center, 'train'), Data(data_center, 'test')
    training_sampler = DistributedSampler(training_data, shuffle=True) if args.distributed_training else RandomSampler(training_data)
    test_sampler = SequentialSampler(test_data)
    training_loader = DataLoader(training_data, batch_size=args.minibatch_size, sampler=training_sampler)
    test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

    return data_center, training_loader, test_loader


def train(args):

    # Setup CUDA, GPU & distributed training
    if not args.distributed_training:
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.global_rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device = torch.device('cuda:' + str(args.local_rank))

    set_random_seed(args.random_seed + args.local_rank)

    if args.local_rank in [-1, 0]:
        print('******************************************************')
        print('********************** training **********************')
        print('******************************************************')

    if args.local_rank in [-1, 0]:
        print('Loading data...')
    data_center, training_loader, test_loader = load_data(args)

    if args.local_rank in [-1, 0]:
        print('Loading model...')
    model = Model(args, data_center).to(args.device)

    if args.local_rank in [-1, 0]:
        print(model)

    # define DDP here
    if args.distributed_training:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    if args.local_rank in [-1, 0]:
        print('Start training...')

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate)

    t = time.time()
    for epoch_id in range(1, args.num_epochs + 1):
        # training
        one_epoch_loss = 0.0
        ddp_model.train()
        if args.distributed_training:
            training_loader.sampler.set_epoch(epoch_id)
        data_center.sample_evid_texts()
        data_center.sample_images()
        for batch_id, batch in tqdm(enumerate(training_loader), total=len(training_loader)):
            claim_ids, labels = batch
            optimizer.zero_grad()
            res = ddp_model(claim_ids, labels, data_center, mode='train')
            loss = res[0]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                one_epoch_loss += loss.item()
            if args.distributed_training:
                torch.distributed.barrier()
        one_epoch_loss /= len(training_loader)
        # validation
        if epoch_id % args.log_steps == 0 and args.local_rank in [-1, 0]:
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_id, args.num_epochs), '\tLoss: %f' % one_epoch_loss)
            ckpt_folder_exists = os.path.exists('./ckpt')
            if not ckpt_folder_exists:
                os.makedirs('./ckpt')
            torch.save(model.state_dict(), './ckpt/' + args.dataset_name + '_ver_and_exp_' + str(epoch_id) + '.pt')
            test(model, data_center, test_loader)
        if args.distributed_training:
            torch.distributed.barrier()

    if args.distributed_training:
        cleanup()


def test(model, data_center, test_loader):

    model.eval()
    y_true, y_pred, exp_pred, exp_true, total_claim_ids = [], [], [], [], []
    for batch_id, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        claim_ids, labels = batch
        res = model(claim_ids, labels, data_center, mode='test')
        total_claim_ids.extend(claim_ids.detach().cpu().numpy().tolist())
        y_pred.extend(res[1].detach().cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        if data_center.has_explanations:
            exp_pred.extend(data_center.decoder_tokenizer.batch_decode(res[2], skip_special_tokens=True))
            decoder_labels = res[3]
            decoder_labels[decoder_labels == -100] = data_center.decoder_tokenizer.pad_token_id
            exp_true.extend(data_center.decoder_tokenizer.batch_decode(decoder_labels, skip_special_tokens=True))

    y_pred = np.array(y_pred)[:data_center.num_test_claims]
    y_true = np.array(y_true)[:data_center.num_test_claims]
    classification(y_pred, y_true)
    if data_center.has_explanations:
        exp_pred = np.array(exp_pred)[:data_center.num_test_claims].tolist()
        exp_true = np.array(exp_true)[:data_center.num_test_claims].tolist()
        explanation_generation(exp_pred, exp_true)

    results = []
    for i in range(len(y_true)):
        result = {}
        result['claim_id'] = int(total_claim_ids[i])
        result['y_true'], result['y_pred'], result['exp_true'], result['exp_pred'] = int(y_true[i]), int(y_pred[i]), \
        exp_true[i], exp_pred[i]
        results.append(result)
    with open('../data/' + data_center.dataset_name + '/results.json', 'w') as f:
        json.dump(results, f, indent=4)


def main(args):

    if args.mode == 'train':
        train(args)
    else:
        ################## You should use single GPU for testing. ####################
        print('******************************************************')
        print('********************** testing ***********************')
        print('******************************************************')
        args.distributed_training = False
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
        set_random_seed(args.random_seed)
        data_center, training_loader, test_loader = load_data(args)
        model = Model(args, data_center).to(args.device)
        ckpt = torch.load('./ckpt/' + args.dataset_name + '_ver_and_exp_100.pt', map_location='cpu')
        model.load_state_dict(ckpt)
        test(model, data_center, test_loader)


if __name__ == '__main__':
    main(parse_args())
import torch
import torch.nn as nn
import numpy as np
import sklearn
import transformers
from transformers import BertConfig, ViTConfig, BertModel, ViTModel
from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):

    def __init__(self, args, data):

        super(Model, self).__init__()
        self.data = data
        self.parse_args(args)
        if args.local_rank in [-1, 0]:
            self.show_config()
        self.generate_modules(args)

    def parse_args(self, args):

        self.current_device = args.device
        self.mode = args.mode
        self.ddp = args.distributed_training
        if self.ddp:
            self.world_size = args.world_size
        self.bert_version = args.bert_version
        self.vit_version = args.vit_version
        self.dataset_name = args.dataset_name
        self.num_claims = self.data.num_claims
        self.num_training_claims = self.data.num_training_claims
        self.num_evidences = self.data.num_evidences
        self.num_images = self.data.num_images
        self.num_labels = self.data.num_labels
        args.num_labels = self.data.num_labels
        self.has_claim_images = self.data.has_claim_images
        self.max_text_length = args.max_text_length
        self.num_sampled_images = args.num_sampled_images
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.save_retrieval_result = args.save_retrieval_result

    def show_config(self):

        print('******************************************************')
        print('torch version:', torch.__version__)
        print('np version:', np.__version__)
        print('sklearn version:', sklearn.__version__)
        print('transformers version:', transformers.__version__)
        print('device:', self.current_device)
        print('distributed training:', self.ddp)
        if self.ddp:
            print('world size:', self.world_size)
        print('mode:', self.mode)
        print('bert version:', self.bert_version)
        print('vit version:', self.vit_version)
        print('dataset name:', self.dataset_name)
        print('max text length:', self.max_text_length)
        print('#claims:', self.num_claims)
        print('#training claims:', self.num_training_claims)
        print('#evidence:', self.num_evidences)
        print('#images:', self.num_images)
        print('#labels:', self.num_labels)
        print('has claim images:', self.has_claim_images)
        print('#sampled images:', self.num_sampled_images)
        print('#epochs:', self.num_epochs)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('save retrieval result:', self.save_retrieval_result)
        print('******************************************************')

    def generate_modules(self, args):

        bert_config = BertConfig.from_pretrained(args.bert_pretrained_model_name_or_path)
        self.bert = BertModel.from_pretrained(args.bert_pretrained_model_name_or_path, config=bert_config)

        vit_config = ViTConfig.from_pretrained(args.vit_pretrained_model_name_or_path)
        self.vit = ViTModel.from_pretrained(args.vit_pretrained_model_name_or_path, config=vit_config)

        self.encoder = Encoder(bert_config, vit_config, args)
        self.decoder = Decoder()

    def preprocess_claims_and_evidence(self, claim_evid_pairs, data):

        claim_ids, evid_ids = claim_evid_pairs[:, 0].detach().cpu().numpy(), claim_evid_pairs[:, 1].detach().cpu().numpy()

        # claim_texts = [data.claims[claim_id]['claim_text'] for claim_id in claim_ids]
        # claim_input_ids, claim_attention_mask = data.generate_input_ids_and_attention_mask(claim_texts)
        claim_input_ids, claim_attention_mask = data.claim_input_ids[claim_ids], data.claim_attention_mask[claim_ids]
        claim_input_ids = np.reshape(claim_input_ids, [-1, self.max_text_length])
        claim_input_ids = torch.LongTensor(claim_input_ids).to(self.current_device)
        claim_attention_mask = np.reshape(claim_attention_mask, [-1, self.max_text_length])
        claim_attention_mask = torch.LongTensor(claim_attention_mask).to(self.current_device)

        # evid_texts = [data.evidences[evid_id]['evid_text'] for evid_id in evid_ids]
        # evid_input_ids, evid_attention_mask = data.generate_input_ids_and_attention_mask(evid_texts)
        evid_input_ids, evid_attention_mask = data.evid_input_ids[evid_ids], data.evid_attention_mask[evid_ids]
        evid_input_ids = np.reshape(evid_input_ids, [-1, self.max_text_length])
        evid_input_ids = torch.LongTensor(evid_input_ids).to(self.current_device)
        evid_attention_mask = np.reshape(evid_attention_mask, [-1, self.max_text_length])
        evid_attention_mask = torch.LongTensor(evid_attention_mask).to(self.current_device)

        claim_pixel_values = None
        if data.has_claim_images:
            claim_image_names = np.reshape(data.sampled_claim_images[claim_ids], [-1])
            # claim_pixel_values = data.generate_pixel_values(claim_image_names)
            # claim_pixel_values = np.concatenate([claim_pixel_values[claim_image_name] for claim_image_name in claim_image_names], axis=0)
            claim_pixel_values = np.concatenate([data.pixel_values[claim_image_name] for claim_image_name in claim_image_names], axis=0)
            if len(claim_pixel_values.shape) == 3:  # in this case there is only one data point
                claim_pixel_values = np.expand_dims(claim_pixel_values, axis=0)
            claim_pixel_values = torch.FloatTensor(claim_pixel_values).to(self.current_device)

        evid_image_names = np.reshape(data.sampled_evid_images[evid_ids], [-1])
        # evid_pixel_values = data.generate_pixel_values(evid_image_names)
        # evid_pixel_values = np.concatenate([evid_pixel_values[evid_image_name] for evid_image_name in evid_image_names], axis=0)
        evid_pixel_values = np.concatenate([data.pixel_values[evid_image_name] for evid_image_name in evid_image_names], axis=0)
        if len(evid_pixel_values.shape) == 3:  # in this case there is only one data point
            evid_pixel_values = np.expand_dims(evid_pixel_values, axis=0)
        evid_pixel_values = torch.FloatTensor(evid_pixel_values).to(self.current_device)

        return (claim_input_ids,
                claim_attention_mask,
                evid_input_ids,
                evid_attention_mask,
                claim_pixel_values,
                evid_pixel_values)

    def forward(self, claim_evid_pairs, data, mode):

        # preprocess claims and evidence
        (claim_input_ids,
         claim_attention_mask,
         evid_input_ids,
         evid_attention_mask,
         claim_pixel_values,
         evid_pixel_values) = self.preprocess_claims_and_evidence(claim_evid_pairs, data)

        # encoder
        claim_emb = self.encoder(self.bert,
                                 claim_input_ids,
                                 claim_attention_mask,
                                 self.vit,
                                 claim_pixel_values,
                                 mode)
        evid_emb = self.encoder(self.bert,
                                evid_input_ids,
                                evid_attention_mask,
                                self.vit,
                                evid_pixel_values,
                                mode)

        # decoder
        loss = self.decoder(claim_emb, evid_emb)

        return [loss, claim_emb, evid_emb]
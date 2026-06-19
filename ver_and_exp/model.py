import torch
import torch.nn as nn
import numpy as np
import sklearn
import transformers
from transformers import BertConfig, ViTConfig, T5Config, BertModel, ViTModel, T5ForConditionalGeneration
from encoder import Encoder
from aggregator import TwoLevelAggregator
from decoder import Classifier, ExplanationGenerator, ExplanationRegularizer


class Model(nn.Module):

    def __init__(self, args, data):

        super(Model, self).__init__()
        self.data = data
        self.parse_args(args)
        if args.local_rank in [-1, 0]:
            self.show_config()
        self.generate_modules(args)

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.current_device = args.device
        self.mode = args.mode
        self.ddp = args.distributed_training
        if self.ddp:
            self.world_size = args.world_size
        self.bert_version = args.bert_version
        self.vit_version = args.vit_version
        self.t5_version = args.t5_version
        self.evidence_provided = args.evidence_provided
        self.num_claims = self.data.num_claims
        self.num_training_claims = self.data.num_training_claims
        self.num_evidences = self.data.num_evidences
        self.num_images = self.data.num_images
        self.num_labels = self.data.num_labels
        args.num_labels = self.data.num_labels
        self.has_claim_images = self.data.has_claim_images
        self.has_explanations = self.data.has_explanations
        self.max_text_length = args.max_text_length
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.lambda_exp_reg = args.lambda_exp_reg
        self.max_new_tokens = args.max_new_tokens

    def show_config(self):

        print('******************************************************')
        print('dataset name:', self.dataset_name)
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
        print('t5 version:', self.t5_version)
        print('evidence provided:', self.evidence_provided)
        print('max text length:', self.max_text_length)
        print('#claims:', self.num_claims)
        print('#training claims:', self.num_training_claims)
        print('#evidence:', self.num_evidences)
        print('#images:', self.num_images)
        print('#labels:', self.num_labels)
        print('has claim images:', self.has_claim_images)
        print('has explanations:', self.has_explanations)
        print('#sampled evidence texts:', self.num_sampled_evid_texts)
        print('#sampled images:', self.num_sampled_images)
        print('#epochs:', self.num_epochs)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('hyperparameter for explanation regularization:', self.lambda_exp_reg)
        print('max num of tokens for explanation:', self.max_new_tokens)
        print('******************************************************')

    def generate_modules(self, args):

        bert_config = BertConfig.from_pretrained(args.bert_pretrained_model_name_or_path)
        self.bert = BertModel.from_pretrained(args.bert_pretrained_model_name_or_path, config=bert_config)

        vit_config = ViTConfig.from_pretrained(args.vit_pretrained_model_name_or_path)
        self.vit = ViTModel.from_pretrained(args.vit_pretrained_model_name_or_path, config=vit_config)

        self.encoder = Encoder(bert_config, vit_config, args)
        self.two_level_aggregator = TwoLevelAggregator(bert_config, vit_config, args)
        self.classifier = Classifier(bert_config, vit_config, args)

        if self.has_explanations:
            t5_config = T5Config.from_pretrained(args.t5_pretrained_model_name_or_path)
            self.t5 = T5ForConditionalGeneration.from_pretrained(args.t5_pretrained_model_name_or_path, config=t5_config)

            self.exp_generator = ExplanationGenerator(bert_config, vit_config, t5_config, args)
            self.exp_regularizer = ExplanationRegularizer(t5_config, args)

    def preprocess_claims_and_evidence(self, claim_ids, data):

        claim_ids = claim_ids.detach().cpu().numpy()
        evid_ids = np.reshape(data.sampled_evid_ids[claim_ids], [-1])

        # claim_texts = [data.claims[claim_id]['claim_text'] for claim_id in claim_ids]
        # claim_input_ids, claim_attention_mask, _ = data.generate_input_ids_and_attention_mask(claim_texts)
        claim_input_ids, claim_attention_mask = data.claim_input_ids[claim_ids], data.claim_attention_mask[claim_ids]
        claim_input_ids = np.reshape(claim_input_ids, [-1, self.max_text_length])
        claim_input_ids = torch.LongTensor(claim_input_ids).to(self.current_device)
        claim_attention_mask = np.reshape(claim_attention_mask, [-1, self.max_text_length])
        claim_attention_mask = torch.LongTensor(claim_attention_mask).to(self.current_device)

        # evid_texts = [data.evidences[evid_id]['evid_text'] for evid_id in evid_ids]
        # evid_input_ids, evid_attention_mask, _ = data.generate_input_ids_and_attention_mask(evid_texts)
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

    def preprocess_explanations(self, claim_ids, labels, data):

        claim_ids = claim_ids.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        exp_input_texts = []
        for idx, claim_id in enumerate(claim_ids):
            evid_ids = np.reshape(data.sampled_evid_ids[claim_id], [-1])
            for evid_id in evid_ids:
                label = data.label_id2label_name[labels[idx]]
                label = 'does not have enough information to verify' if label == 'nei' else label + 's'  # may need to replace nei with something else
                exp_input_text = 'The evidence ' + label + ' the claim.'
                exp_input_text += ' </s> Claim: '
                exp_input_text += data.claims[claim_id]['claim_text']
                exp_input_text += ' </s> Evidence: '
                exp_input_text += data.evidences[evid_id]['evid_text']
                exp_input_texts.append(exp_input_text)

        exp_input_ids, exp_attention_mask, _ = data.generate_input_ids_and_attention_mask(exp_input_texts, is_decoder=True, output_decoder_labels=False)
        exp_input_ids = np.reshape(exp_input_ids, [-1, self.max_text_length])
        exp_input_ids = torch.LongTensor(exp_input_ids).to(self.current_device)
        exp_attention_mask = np.reshape(exp_attention_mask, [-1, self.max_text_length])
        exp_attention_mask = torch.LongTensor(exp_attention_mask).to(self.current_device)

        exp_output_texts = [data.claims[claim_id]['explanation'] for claim_id in claim_ids]
        _, decoder_attention_mask, decoder_labels = data.generate_input_ids_and_attention_mask(exp_output_texts, is_decoder=True, output_decoder_labels=True)
        decoder_labels = np.reshape(decoder_labels, [-1, self.max_new_tokens])
        decoder_labels = torch.LongTensor(decoder_labels).to(self.current_device)
        decoder_attention_mask = np.reshape(decoder_attention_mask, [-1, self.max_new_tokens])
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask).to(self.current_device)

        return (exp_input_ids,
                exp_attention_mask,
                decoder_labels,
                decoder_attention_mask)

    def forward(self, claim_ids, labels, data, mode):

        # preprocess claims and evidence
        (claim_input_ids,
         claim_attention_mask,
         evid_input_ids,
         evid_attention_mask,
         claim_pixel_values,
         evid_pixel_values) = self.preprocess_claims_and_evidence(claim_ids, data)

        # claim encoder
        claim_text_hidden_states, claim_image_hidden_states = self.encoder(self.bert,
                                                                           claim_input_ids,
                                                                           claim_attention_mask,
                                                                           self.vit,
                                                                           claim_pixel_values,
                                                                           'claim',
                                                                           mode)

        # evidence encoder
        evid_text_hidden_states, evid_image_hidden_states = self.encoder(self.bert,
                                                                         evid_input_ids,
                                                                         evid_attention_mask,
                                                                         self.vit,
                                                                         evid_pixel_values,
                                                                         'evid',
                                                                         mode)

        # two-level aggregator
        claim_emb, evid_emb = self.two_level_aggregator(claim_text_hidden_states,
                                                        claim_image_hidden_states,
                                                        evid_text_hidden_states,
                                                        evid_attention_mask,
                                                        evid_image_hidden_states,
                                                        claim_attention_mask,
                                                        mode)

        loss = 0
        # classifier
        clf_loss, y_pred, y_pred_prob = self.classifier(claim_emb, evid_emb, labels, mode)
        loss += clf_loss

        # explanation generator
        exp_loss, exp_clf_loss, kl_loss, generated_ids, decoder_labels, exp_y_pred = 0, 0, 0, None, None, None
        if self.has_explanations:
            # preprocess explanations
            labels = labels if mode == 'train' else y_pred
            (exp_input_ids,
             exp_attention_mask,
             decoder_labels,
             decoder_attention_mask) = self.preprocess_explanations(claim_ids, labels, data)

            # explanation generation
            exp_loss, lm_logits, generated_ids = self.exp_generator(self.t5,
                                                                    exp_input_ids,
                                                                    exp_attention_mask,
                                                                    decoder_labels,
                                                                    claim_image_hidden_states,
                                                                    evid_image_hidden_states,
                                                                    mode)

            # explanation regularizer
            exp_clf_loss, kl_loss, exp_y_pred = self.exp_regularizer(lm_logits,
                                                                     decoder_attention_mask,
                                                                     labels,
                                                                     y_pred_prob)

            loss += exp_loss + self.lambda_exp_reg * (exp_clf_loss + kl_loss)

        return [loss, y_pred, generated_ids, decoder_labels]
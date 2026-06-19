from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor, T5Tokenizer
from PIL import Image
import numpy as np
import collections
import os
import json
from tqdm import tqdm
import shutil


class DataCenter():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.preprocess_data()
        self.sample_evid_texts()
        self.sample_images()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.evidence_provided = args.evidence_provided
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images
        self.max_text_length = args.max_text_length
        self.max_new_tokens = args.max_new_tokens
        self.bert_pretrained_model_name_or_path = args.bert_pretrained_model_name_or_path
        self.vit_pretrained_model_name_or_path = args.vit_pretrained_model_name_or_path
        self.t5_pretrained_model_name_or_path = args.t5_pretrained_model_name_or_path

    def load_data(self):

        with open('../data/' + self.dataset_name + '/claims.json', 'r') as file:
            claims_list = json.load(file)
            self.claims = {}
            self.claims = {claim['claim_id']: claim for claim in claims_list}
        with open('../data/' + self.dataset_name + '/evidence.json', 'r') as file:
            evidences_list = json.load(file)
            self.evidences = {}
            self.evidences = {evidence['evid_id']: evidence for evidence in evidences_list}

        self.image_dir = '../data/' + self.dataset_name + '/images'
        self.image_names = [image_name for image_name in os.listdir(self.image_dir) if image_name.endswith(('jpg', 'jpeg', 'png'))]
        self.num_claims, self.num_evidences, self.num_images = len(self.claims), len(self.evidences), len(self.image_names)

        self.text_tokenizer = BertTokenizer.from_pretrained(self.bert_pretrained_model_name_or_path)
        self.image_processor = ViTImageProcessor.from_pretrained(self.vit_pretrained_model_name_or_path)

        self.has_explanations = False
        if 'explanation' in self.claims[0]:
            self.has_explanations = True
            self.decoder_tokenizer = T5Tokenizer.from_pretrained(self.t5_pretrained_model_name_or_path)

    def split_data(self):

        self.training_claim_ids, self.dev_claim_ids, self.test_claim_ids = [], [], []
        if 'train_dev_test' in self.claims[0]:
            for claim_id in self.claims.keys():
                if self.claims[claim_id]['train_dev_test'] == 'train':
                    self.training_claim_ids.append(claim_id)
                elif self.claims[claim_id]['train_dev_test'] == 'dev':
                    self.dev_claim_ids.append(claim_id)
                else:
                    self.test_claim_ids.append(claim_id)
        else:
            self.training_claim_ids = np.arange(int(self.num_claims * 0.72))
            self.dev_claim_ids = np.arange(len(self.training_claim_ids), int(self.num_claims * 0.8))
            self.test_claim_ids = np.arange(len(self.training_claim_ids) + len(self.dev_claim_ids), self.num_claims)
        self.training_claim_ids, self.dev_claim_ids, self.test_claim_ids = np.array(self.training_claim_ids), np.array(self.dev_claim_ids), np.array(self.test_claim_ids)

        if len(self.test_claim_ids) == 0:  # this means the dataset doesn't have test set but only dev set
            self.test_claim_ids = self.dev_claim_ids
        self.num_training_claims, self.num_test_claims = len(self.training_claim_ids), len(self.test_claim_ids)

        self.label_names = np.unique([self.claims[claim_id]['label'] for claim_id in self.claims.keys()])
        self.label_name2label_id = {label_name: label_id for label_id, label_name in enumerate(self.label_names)}
        self.label_id2label_name = {label_id: label_name for label_name, label_id in self.label_name2label_id.items()}
        self.training_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.training_claim_ids])
        self.dev_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.dev_claim_ids])
        self.test_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.test_claim_ids])
        self.num_labels = len(self.label_names)

        self.training_claim_evid_pairs = []
        for claim_id in self.training_claim_ids:
            for training_evid_id in self.claims[claim_id]['gold_evid_ids']:
                self.training_claim_evid_pairs.append([claim_id, training_evid_id])
        self.training_claim_evid_pairs = np.unique(self.training_claim_evid_pairs, axis=0)

        self.test_claim_evid_pairs = []
        for claim_id in self.test_claim_ids:
            for test_evid_id in self.claims[claim_id]['gold_evid_ids']:
                self.test_claim_evid_pairs.append([claim_id, test_evid_id])
        self.test_claim_evid_pairs = np.unique(self.test_claim_evid_pairs, axis=0)

    def preprocess_data(self):

        claim_texts = []
        for claim_id in range(self.num_claims):
            claim_texts.append(self.claims[claim_id]['claim_text'])

        evid_texts = []
        for evid_id in range(self.num_evidences):
            evid_texts.append(self.evidences[evid_id]['evid_text'])

        if self.has_explanations:
            exp_texts = []
            for claim_id in range(self.num_claims):
                exp_texts.append(self.claims[claim_id]['explanation'])

        self.claim_input_ids, self.claim_attention_mask, _ = self.generate_input_ids_and_attention_mask(claim_texts)
        self.evid_input_ids, self.evid_attention_mask, _ = self.generate_input_ids_and_attention_mask(evid_texts)
        self.pixel_values = self.generate_pixel_values(self.image_names)

    def generate_input_ids_and_attention_mask(self, texts, is_decoder=False, output_decoder_labels=False):

        tokenizer = self.decoder_tokenizer if is_decoder else self.text_tokenizer
        max_length = self.max_new_tokens if is_decoder and output_decoder_labels else self.max_text_length

        input_ids, attention_mask = [], []
        for text in texts:
            text = text.strip()
            tokenized_text = tokenizer.batch_encode_plus([text], max_length=max_length, padding='max_length', truncation=True)
            input_ids.extend(tokenized_text['input_ids'])
            attention_mask.extend(tokenized_text['attention_mask'])
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        decoder_labels = None
        if is_decoder:
            decoder_input_ids = np.array(input_ids)
            decoder_attention_mask = np.array(attention_mask)
            decoder_labels = np.copy(decoder_input_ids).tolist()
            decoder_labels = np.array([
                [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
                [zip(masks, labels) for masks, labels in zip(decoder_attention_mask, decoder_labels)]
            ])

        return input_ids, attention_mask, decoder_labels

    def generate_pixel_values(self, image_names):

        # pixel_values = {}
        # for image_name in image_names:
        #     image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        #     try:
        #         processed_image = self.image_processor(images=image)
        #     except:
        #         image = Image.open(os.path.join(self.image_dir, '0.jpg')).convert('RGB')
        #         processed_image = self.image_processor(images=image)
        #     pixel_values[image_name] = processed_image['pixel_values']
        #     image.close()

        pixel_values = {}
        for image_name in image_names:
            image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
            processed_image = self.image_processor(images=image)
            pixel_values[image_name] = processed_image['pixel_values']
            image.close()

        return pixel_values

    def sample_evid_texts(self):

        self.sampled_evid_ids = []
        for claim_id in range(self.num_claims):
            claim = self.claims[claim_id]
            if self.evidence_provided == 'gold':
                evid_ids = claim['gold_evid_ids']
                replace = len(evid_ids) < self.num_sampled_evid_texts
                sampled_evid_indices = np.random.choice(len(evid_ids), size=self.num_sampled_evid_texts, replace=replace)
                self.sampled_evid_ids.append([evid_ids[sampled_evid_idx] for sampled_evid_idx in sampled_evid_indices])
            else:
                evid_ids = claim['retrieved_evid_ids']
                if len(evid_ids) >= self.num_sampled_evid_texts:
                    self.sampled_evid_ids.append([evid_ids[sampled_evid_idx] for sampled_evid_idx in range(self.num_sampled_evid_texts)])
                else:
                    sampled_evid_indices = np.random.choice(len(evid_ids), size=self.num_sampled_evid_texts - len(evid_ids), replace=True)
                    sampled_evid_ids_1 = [evid_ids[sampled_evid_idx] for sampled_evid_idx in range(len(evid_ids))]
                    sampled_evid_ids_2 = [evid_ids[sampled_evid_idx] for sampled_evid_idx in sampled_evid_indices]
                    self.sampled_evid_ids.append(sampled_evid_ids_1 + sampled_evid_ids_2)
        self.sampled_evid_ids = np.array(self.sampled_evid_ids)

    def sample_images(self):

        self.sampled_evid_images = []
        for evid_id in range(self.num_evidences):
            image_names = self.evidences[evid_id]['evid_images']
            replace = len(image_names) < self.num_sampled_images
            sampled_image_indices = np.random.choice(len(image_names), size=self.num_sampled_images, replace=replace)
            self.sampled_evid_images.append([image_names[sampled_image_idx] for sampled_image_idx in sampled_image_indices])
        self.sampled_evid_images = np.array(self.sampled_evid_images)

        self.has_claim_images = False
        if 'claim_images' in self.claims[0]:
            self.has_claim_images = True
            self.sampled_claim_images = []
            for claim_id in range(self.num_claims):
                image_names = self.claims[claim_id]['claim_images']
                replace = len(image_names) < self.num_sampled_images
                sampled_image_indices = np.random.choice(len(image_names), size=self.num_sampled_images, replace=replace)
                self.sampled_claim_images.append([image_names[sampled_image_idx] for sampled_image_idx in sampled_image_indices])
            self.sampled_claim_images = np.array(self.sampled_claim_images)


class Data(Dataset):

    def __init__(self, data, mode):

        super(Data, self).__init__()
        self.data = data
        self.mode = mode

        if self.mode == 'train':
            self.claim_ids = self.data.training_claim_ids
            self.labels = self.data.training_labels
        elif self.mode == 'test':
            self.claim_ids = self.data.test_claim_ids
            self.labels = self.data.test_labels

    def __len__(self):

        return len(self.claim_ids)

    def __getitem__(self, idx):

        claim_id = self.claim_ids[idx]
        label = self.labels[idx]

        return claim_id, label
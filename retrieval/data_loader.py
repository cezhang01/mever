from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from PIL import Image
import numpy as np
import collections
import os
import json
from tqdm import tqdm


class DataCenter():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.preprocess_data()
        self.sample_images()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.num_sampled_images = args.num_sampled_images
        self.max_text_length = args.max_text_length
        self.bert_pretrained_model_name_or_path = args.bert_pretrained_model_name_or_path
        self.vit_pretrained_model_name_or_path = args.vit_pretrained_model_name_or_path

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
        self.training_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.training_claim_ids])
        self.dev_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.dev_claim_ids])
        self.test_labels = np.array([self.label_name2label_id[self.claims[claim_id]['label']] for claim_id in self.test_claim_ids])
        self.num_labels = len(self.label_names)

        self.training_claim_evid_pairs = []
        for claim_id in self.training_claim_ids:
            for training_evid_id in self.claims[claim_id]['gold_evid_ids']:
                self.training_claim_evid_pairs.append([claim_id, training_evid_id])
        self.training_claim_evid_pairs = np.unique(self.training_claim_evid_pairs, axis=0)

        self.test_evid_ids, self.test_claim_evid_pairs = [], []
        for claim_id in self.test_claim_ids:
            self.test_evid_ids.extend(self.claims[claim_id]['gold_evid_ids'])
            for test_evid_id in self.claims[claim_id]['gold_evid_ids']:
                self.test_claim_evid_pairs.append([claim_id, test_evid_id])
        self.test_evid_ids = np.unique(self.test_evid_ids)
        self.num_test_evid = len(self.test_evid_ids)
        self.test_claim_evid_pairs = np.unique(self.test_claim_evid_pairs, axis=0)

    def preprocess_data(self):

        claim_texts = []
        for claim_id in range(self.num_claims):
            claim_texts.append(self.claims[claim_id]['claim_text'])

        evid_texts = []
        for evid_id in range(self.num_evidences):
            evid_texts.append(self.evidences[evid_id]['evid_text'])

        self.claim_input_ids, self.claim_attention_mask = self.generate_input_ids_and_attention_mask(claim_texts)
        self.evid_input_ids, self.evid_attention_mask = self.generate_input_ids_and_attention_mask(evid_texts)
        self.pixel_values = self.generate_pixel_values(self.image_names)

    def generate_input_ids_and_attention_mask(self, texts):

        input_ids, attention_mask = [], []
        for text in texts:
            text = text.strip().lower()
            tokenized_text = self.text_tokenizer.batch_encode_plus([text], max_length=self.max_text_length, padding='max_length', truncation=True)
            input_ids.extend(tokenized_text['input_ids'])
            attention_mask.extend(tokenized_text['attention_mask'])
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        return input_ids, attention_mask

    def generate_pixel_values(self, image_names):

        pixel_values = {}
        for image_name in image_names:
            image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
            processed_image = self.image_processor(images=image)
            pixel_values[image_name] = processed_image['pixel_values']
            image.close()

        return pixel_values

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
            self.claim_evid_pairs = self.data.training_claim_evid_pairs
        elif self.mode == 'test_claim':
            self.claim_evid_pairs = np.array([[claim_id, 0] for claim_id in range(self.data.num_claims)])
        elif self.mode == 'test_evid':
            self.claim_evid_pairs = np.array([[0, evid_id] for evid_id in range(self.data.num_evidences)])

    def __len__(self):

        return len(self.claim_evid_pairs)

    def __getitem__(self, idx):

        claim_evid_pair = self.claim_evid_pairs[idx]

        return claim_evid_pair
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.modeling_outputs import BaseModelOutput
from copy import deepcopy


class Classifier(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(Classifier, self).__init__()
        self.bert_hidden_size = bert_config.hidden_size
        self.num_labels = args.num_labels
        self.current_device = args.device

        self.linear_layers = nn.ModuleList([nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size),
                                            nn.Linear(self.bert_hidden_size, self.num_labels)])
        self.ce_loss = nn.functional.cross_entropy
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, claim_emb, evid_emb, labels, mode):

        emb_concat = torch.concat([claim_emb, evid_emb], dim=1)
        for layer_id in range(len(self.linear_layers)):
            emb_concat = self.linear_layers[layer_id](emb_concat)
        logits = emb_concat
        y_pred = torch.argmax(logits, dim=-1)
        y_pred_prob = self.softmax(logits)
        one_hot = nn.functional.one_hot(labels.to(self.current_device), num_classes=self.num_labels)
        one_hot = one_hot.float()
        loss = self.ce_loss(logits, one_hot, reduction='none')
        loss = torch.mean(loss)

        return loss, y_pred, y_pred_prob


class ExplanationGenerator(nn.Module):

    def __init__(self, bert_config, vit_config, t5_config, args):

        super(ExplanationGenerator, self).__init__()
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images
        self.max_text_length = args.max_text_length
        self.max_new_tokens = args.max_new_tokens
        self.current_device = args.device
        self.model_dim = t5_config.d_model
        self.vit_hidden_size = vit_config.hidden_size
        self.num_labels = args.num_labels
        self.t5_pad_token_id = t5_config.pad_token_id

        self.generate_modules(bert_config, vit_config, t5_config, args)

    def generate_modules(self, bert_config, vit_config, t5_config, args):

        self.emb_proj_layers = nn.ModuleList([nn.Linear(self.vit_hidden_size, self.model_dim)])
        bert_config_exp = deepcopy(bert_config)
        bert_config_exp.hidden_size = self.model_dim
        bert_config_exp.num_attention_heads = 1
        self.bert_att_layers = nn.ModuleList([BertSelfAttention(bert_config_exp)])
        self.linear_layers = nn.ModuleList([nn.Linear(self.model_dim, self.model_dim),
                                            nn.Linear(self.model_dim, self.num_labels)])
        self.ce_loss = nn.functional.cross_entropy
        self.softmax = nn.Softmax(dim=-1)

    def multimodal_t5_encoder(self, t5, input_ids, attention_mask, claim_image_hidden_states, evid_image_hidden_states):

        inputs_embeds = t5.encoder.embed_tokens(input_ids)

        # embedding projection
        if claim_image_hidden_states is not None:
            claim_image_hidden_states = self.emb_proj_layers[0](claim_image_hidden_states)
            claim_image_cls_emb = claim_image_hidden_states[:, 0, :]
            claim_image_cls_emb = torch.reshape(torch.tile(torch.unsqueeze(claim_image_cls_emb, dim=1), [1, self.num_sampled_evid_texts, 1]), [-1, self.model_dim])
        evid_image_hidden_states = self.emb_proj_layers[0](evid_image_hidden_states)
        evid_image_cls_emb = evid_image_hidden_states[:, 0, :]

        # concatenating input embeddings
        if claim_image_hidden_states is not None:
            inputs_embeds = torch.concat([torch.unsqueeze(claim_image_cls_emb, dim=1), inputs_embeds], dim=1)
            station_mask = torch.ones([attention_mask.size(0), 1], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.concat([station_mask, attention_mask], dim=1)
        evid_image_cls_emb = torch.reshape(evid_image_cls_emb, [-1, self.num_sampled_images, self.model_dim])
        inputs_embeds = torch.concat([evid_image_cls_emb, inputs_embeds], dim=1)
        station_mask = torch.ones([attention_mask.size(0), self.num_sampled_images], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.concat([station_mask, attention_mask], dim=1)

        # multimodal t5 encoder
        encoder_outputs = t5.encoder(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        # fusion-in-decoder
        encoder_hidden_states = torch.reshape(encoder_hidden_states, [-1, self.num_sampled_evid_texts, attention_mask.size(1), self.model_dim])
        attention_mask = torch.reshape(attention_mask, [-1, self.num_sampled_evid_texts, attention_mask.size(1)])
        attention_mask_repeat = torch.tile(torch.unsqueeze(attention_mask, dim=-1), [1, 1, 1, self.model_dim])
        encoder_hidden_states = torch.multiply(encoder_hidden_states, attention_mask_repeat)

        encoder_hidden_states_sum = torch.sum(encoder_hidden_states, dim=1)
        attention_mask_sum = torch.tile(torch.unsqueeze(torch.sum(attention_mask, dim=1), dim=-1), [1, 1, self.model_dim])
        encoder_hidden_states = encoder_hidden_states_sum / (attention_mask_sum + 1e-6)

        # attention_mask = torch.max(attention_mask, dim=1)
        attention_mask = torch.any(attention_mask, dim=1).to(attention_mask.dtype)

        return encoder_hidden_states, attention_mask

    def forward(self, t5, input_ids, attention_mask, decoder_labels, claim_image_hidden_states, evid_image_hidden_states, mode, num_beams=4):

        # multimodal t5 encoder
        encoder_hidden_states, attention_mask = self.multimodal_t5_encoder(t5,
                                                                           input_ids,
                                                                           attention_mask,
                                                                           claim_image_hidden_states,
                                                                           evid_image_hidden_states)

        # t5 decoder
        decoder_input_ids = t5._shift_right(decoder_labels)
        decoder_outputs = t5.decoder(input_ids=decoder_input_ids,
                                     encoder_hidden_states=encoder_hidden_states,
                                     encoder_attention_mask=attention_mask)
        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = t5.lm_head(sequence_output)

        # language modeling loss
        loss = self.ce_loss(lm_logits.view(-1, lm_logits.size(-1)), decoder_labels.view(-1), reduction='none')
        loss = torch.mean(loss)

        generated_ids = None
        if mode == 'test':
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states,
                                              hidden_states=None,
                                              attentions=None)
            generated_ids = t5.generate(encoder_outputs=encoder_outputs,
                                        max_length=self.max_new_tokens,
                                        num_beams=num_beams,)
                                        # no_repeat_ngram_size=2,
                                        # early_stopping=True)

        return loss, lm_logits, generated_ids


class ExplanationRegularizer(nn.Module):

    def __init__(self, t5_config, args):

        super(ExplanationRegularizer, self).__init__()
        self.current_device = args.device
        self.model_dim = t5_config.d_model
        self.vocab_size = t5_config.vocab_size
        self.num_labels = args.num_labels

        self.linear_layers = nn.ModuleList([nn.Linear(self.vocab_size, 128),
                                            nn.Linear(128, self.num_labels)])
        self.ce_loss = nn.functional.cross_entropy
        self.softmax = nn.Softmax(dim=-1)

    def mean_pooling(self, embeddings, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            avg_emb = torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        else:
            avg_emb = torch.mean(embeddings, dim=1)

        return avg_emb

    def max_pooling(self, embeddings, mask=None):

        if mask is not None:
            mask = (1.0 - mask) * -10000.0
            mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            max_emb = torch.max(embeddings + mask, dim=1)[0]
        else:
            max_emb = torch.max(embeddings, dim=1)[0]

        return max_emb

    def forward(self, lm_logits, decoder_attention_mask, labels, y_pred_prob):

        exp_emb = self.mean_pooling(lm_logits, decoder_attention_mask)

        for layer_id in range(len(self.linear_layers)):
            exp_emb = self.linear_layers[layer_id](exp_emb)
        logits = exp_emb
        exp_y_pred = torch.argmax(logits, dim=-1)
        exp_y_pred_prob = self.softmax(logits)
        one_hot = nn.functional.one_hot(labels.to(self.current_device), num_classes=self.num_labels)
        one_hot = one_hot.float()
        exp_clf_loss = self.ce_loss(logits, one_hot, reduction='none')
        exp_clf_loss = torch.mean(exp_clf_loss)

        kl_loss1 = nn.functional.kl_div(y_pred_prob.log(), exp_y_pred_prob, reduction='none')
        kl_loss2 = nn.functional.kl_div(exp_y_pred_prob.log(), y_pred_prob, reduction='none')
        kl_loss1, kl_loss2 = torch.mean(kl_loss1), torch.mean(kl_loss2)
        kl_loss = kl_loss1 + kl_loss2

        return exp_clf_loss, kl_loss, exp_y_pred
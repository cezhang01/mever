import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):

    def __init__(self, self_dim, neigh_dim):

        super(GraphConvLayer, self).__init__()
        self.self_dim = self_dim
        self.neigh_dim = neigh_dim
        self.dropout_prob = 0

        self.generate_modules()

    def generate_modules(self):

        self.linear_layers = nn.ModuleList([nn.Linear(self.self_dim, self.self_dim),
                                            nn.Linear(self.neigh_dim, self.self_dim)])
        self.att_layer = nn.ModuleList([nn.Linear(2 * self.self_dim, 1)])
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def message_passing(self, self_emb, neigh_emb, act, mode):

        if mode == 'train':
            self_emb = self.dropout_layer(self_emb)
            neigh_emb = self.dropout_layer(neigh_emb)
        self_emb = self.linear_layers[0](self_emb)
        neigh_emb = self.linear_layers[1](neigh_emb)

        att = self.att(self_emb, neigh_emb)
        neigh_emb_agg = self.agg(self_emb, neigh_emb, att, act)

        return neigh_emb_agg

    def att(self, self_emb, neigh_emb):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        emb_concat = torch.cat([self_emb, neigh_emb], dim=1)
        att = torch.reshape(self.att_layer[0](emb_concat), [-1, n])
        att = self.softmax(self.sigmoid(att))  # may need to change sigmoid here

        return att

    def agg(self, self_emb, neigh_emb, att, act):

        att = torch.unsqueeze(att, dim=1)
        neigh_emb = torch.reshape(neigh_emb, [att.size(0), att.size(-1), -1])
        neigh_emb_agg = torch.reshape(torch.matmul(att, neigh_emb), self_emb.size())
        # self_emb_agg = act(0.5 * self_emb + 0.5 * neigh_emb_agg)
        neigh_emb_agg = act(neigh_emb_agg)

        return neigh_emb_agg

    def forward(self, self_emb, neigh_emb, act, mode):

        neigh_emb_agg = self.message_passing(self_emb, neigh_emb, act, mode)

        return neigh_emb_agg


class EncoderLayer(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(EncoderLayer, self).__init__()
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images
        self.num_hidden_layers = bert_config.num_hidden_layers
        self.bert_hidden_size = bert_config.hidden_size
        self.vit_hidden_size = vit_config.hidden_size

        self.graph_conv_layers = nn.ModuleList([GraphConvLayer(self.bert_hidden_size, self.bert_hidden_size),
                                                GraphConvLayer(self.bert_hidden_size, self.vit_hidden_size),
                                                GraphConvLayer(self.vit_hidden_size, self.vit_hidden_size)])
        self.linear_layer = nn.ModuleList([nn.Linear(self.bert_hidden_size, self.vit_hidden_size)])
        self.identity = nn.Identity()

    def forward(self, bert, bert_hidden_states, attention_mask, vit, vit_hidden_states, claim_or_evid, mode):

        all_bert_hidden_states, all_vit_hidden_states = (), ()
        [num_texts, num_tokens, bert_emb_dim] = bert_hidden_states.size()
        if vit_hidden_states is not None:
            [num_images, num_patches, vit_emb_dim] = vit_hidden_states.size()

        for layer_id in range(self.num_hidden_layers):
            all_bert_hidden_states = all_bert_hidden_states + (bert_hidden_states,)
            if vit_hidden_states is not None:
                all_vit_hidden_states = all_vit_hidden_states + (vit_hidden_states,)
            if layer_id > 0:
                # multi-text reasoning
                bert_cls_emb = bert_hidden_states[:, 2, :].clone()  # [num_texts, bert_emb_dim]
                if claim_or_evid == 'evid' or claim_or_evid == 'evidence':
                    bert_cls_emb_reshape = torch.reshape(bert_cls_emb, [-1, self.num_sampled_evid_texts, self.bert_hidden_size])  # evidence fully connected graph
                elif claim_or_evid == 'claim':
                    bert_cls_emb_reshape = torch.reshape(bert_cls_emb, [-1, 1, self.bert_hidden_size])  # claim self-loop
                multi_text_reasoning = []
                for i in range(bert_cls_emb_reshape.size(1)):
                    multi_text_reasoning_tmp = self.graph_conv_layers[0](bert_cls_emb_reshape[:, i, :], bert_cls_emb, self.identity, mode)
                    multi_text_reasoning.append(torch.unsqueeze(multi_text_reasoning_tmp, dim=1))
                multi_text_reasoning = torch.concat(multi_text_reasoning, dim=1)
                multi_text_reasoning = torch.reshape(multi_text_reasoning, [-1, self.bert_hidden_size])
                bert_hidden_states[:, 0, :] = multi_text_reasoning

                if vit_hidden_states is not None:
                    # cross-modal reasoning, passing vit embeddings to bert
                    vit_cls_emb = vit_hidden_states[:, 2, :].clone()  # [num_texts * num_sampled_images, vit_emb_dim]
                    vit2bert_reasoning = self.graph_conv_layers[1](bert_cls_emb, vit_cls_emb, self.identity, mode)
                    bert_hidden_states[:, 1, :] = vit2bert_reasoning

                    # multi-image reasoning
                    vit_cls_emb_reshape = torch.reshape(vit_cls_emb, [-1, self.num_sampled_images, self.vit_hidden_size])
                    multi_image_reasoning = []
                    for i in range(self.num_sampled_images):
                        multi_image_reasoning_tmp = self.graph_conv_layers[2](vit_cls_emb_reshape[:, i, :], vit_cls_emb, self.identity, mode)
                        multi_image_reasoning.append(torch.unsqueeze(multi_image_reasoning_tmp, dim=1))
                    multi_image_reasoning = torch.concat(multi_image_reasoning, dim=1)
                    multi_image_reasoning = torch.reshape(multi_image_reasoning, [-1, self.vit_hidden_size])
                    vit_hidden_states[:, 0, :] = multi_image_reasoning

                    # cross-modal reasoning, passing bert embeddings to vit
                    bert2vit_reasoning = self.linear_layer[0](bert_cls_emb)
                    bert2vit_reasoning = torch.tile(torch.unsqueeze(bert2vit_reasoning, dim=1), [1, self.num_sampled_images, 1])
                    bert2vit_reasoning = torch.reshape(bert2vit_reasoning, [-1, self.vit_hidden_size])
                    vit_hidden_states[:, 1, :] = bert2vit_reasoning
                    vit_layer_outputs = vit.encoder.layer[layer_id](vit_hidden_states)

                bert_layer_outputs = bert.encoder.layer[layer_id](bert_hidden_states, attention_mask=attention_mask)
            else:
                attention_mask_tmp = attention_mask.clone()
                attention_mask_tmp[:, :, :, :2] = -10000.0
                bert_layer_outputs = bert.encoder.layer[0](bert_hidden_states, attention_mask=attention_mask_tmp)
                if vit_hidden_states is not None:
                    vit_layer_outputs = vit.encoder.layer[0](vit_hidden_states)

            bert_hidden_states = bert_layer_outputs[0]
            if vit_hidden_states is not None:
                vit_hidden_states = vit_layer_outputs[0]

        all_bert_hidden_states = all_bert_hidden_states + (bert_hidden_states,)
        if vit_hidden_states is not None:
            all_vit_hidden_states = all_vit_hidden_states + (vit_hidden_states,)

        return [all_bert_hidden_states, all_vit_hidden_states]


class Encoder(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(Encoder, self).__init__()
        self.bert_hidden_size = bert_config.hidden_size
        self.vit_hidden_size = vit_config.hidden_size

        self.encoder_layer = EncoderLayer(bert_config, vit_config, args)
        self.emb_proj_layers = nn.ModuleList([nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
                                              nn.Linear(self.vit_hidden_size, self.bert_hidden_size)])

    def forward(self, bert, input_ids, attention_mask, vit, pixel_values, claim_or_evid, mode):

        num_texts = input_ids.size(0)
        if pixel_values is not None:
            num_images = pixel_values.size(0)

        # initialize bert hidden states
        bert_hidden_states = bert.embeddings(input_ids=input_ids)

        # add station attention mask
        station_mask = torch.zeros([num_texts, 2], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L
        attention_mask[:, 0] = 1.0  # multi-text reasoning
        if pixel_values is not None:
            attention_mask[:, 1] = 1.0  # cross-modal image2text reasoning
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # add station placeholder
        station_placeholder = torch.zeros([num_texts, 2, bert_hidden_states.size(-1)]).type(bert_hidden_states.dtype).to(bert_hidden_states.device)
        bert_hidden_states = torch.cat([station_placeholder, bert_hidden_states], dim=1)  # N 1+L D

        # initialize vit hidden states
        vit_hidden_states = None
        if pixel_values is not None:
            vit_hidden_states = vit.embeddings(pixel_values)
            # add station placeholder
            station_placeholder = torch.zeros([num_images, 2, vit_hidden_states.size(-1)]).type(vit_hidden_states.dtype).to(vit_hidden_states.device)
            vit_hidden_states = torch.cat([station_placeholder, vit_hidden_states], dim=1)  # N 1+L D

        # encoder
        encoder_outputs = self.encoder_layer(bert=bert,
                                             bert_hidden_states=bert_hidden_states,
                                             attention_mask=extended_attention_mask,
                                             vit=vit,
                                             vit_hidden_states=vit_hidden_states,
                                             claim_or_evid=claim_or_evid,
                                             mode=mode)

        # output
        all_bert_hidden_states, all_vit_hidden_states = encoder_outputs[0], encoder_outputs[1]

        bert_hidden_states = all_bert_hidden_states[-1]
        bert_hidden_states = bert_hidden_states[:, 2:, :]

        vit_hidden_states, vit_cls_emb = None, None
        if len(all_vit_hidden_states) > 0:
            vit_hidden_states = all_vit_hidden_states[-1]
            vit_hidden_states = vit_hidden_states[:, 2:, :]

        return bert_hidden_states, vit_hidden_states
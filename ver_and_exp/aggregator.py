import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class TokenLevelAggregator(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(TokenLevelAggregator, self).__init__()
        self.bert_hidden_size = bert_config.hidden_size
        self.vit_hidden_size = vit_config.hidden_size
        self.num_sampled_images = args.num_sampled_images
        self.chunk_size_feed_forward = bert_config.chunk_size_feed_forward

        self.bert_att_layers = nn.ModuleList([BertSelfAttention(bert_config),
                                              BertSelfAttention(bert_config)])
        self.linear_layers = nn.ModuleList([nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size),
                                            nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size)])

    def forward(self, claim_hidden_states, claim_attention_mask, evid_text_hidden_states, evid_image_hidden_states):

        num_patches = evid_image_hidden_states.size(1)
        extended_claim_attention_mask = (1.0 - claim_attention_mask[:, None, None, :]) * -10000.0

        # evidence text-image aggregation
        evid_image_hidden_states = torch.reshape(evid_image_hidden_states, [-1, self.num_sampled_images, num_patches, self.bert_hidden_size])
        evid_image_hidden_states_agg_list = []
        for i in range(self.num_sampled_images):
            evid_image_hidden_states_agg = self.bert_att_layers[0](evid_text_hidden_states, encoder_hidden_states=evid_image_hidden_states[:, i, :, :])[0]
            evid_image_hidden_states_agg = torch.unsqueeze(evid_image_hidden_states_agg, dim=0)
            evid_image_hidden_states_agg_list.append(evid_image_hidden_states_agg)
        evid_image_hidden_states_agg_mean = torch.mean(torch.concat(evid_image_hidden_states_agg_list, dim=0), dim=0)
        evid_hidden_states = self.linear_layers[0](torch.concat([evid_text_hidden_states, evid_image_hidden_states_agg_mean], dim=-1))

        # claim-evidence aggregation
        claim_hidden_states = self.bert_att_layers[1](evid_hidden_states,
                                                      encoder_hidden_states=claim_hidden_states,
                                                      encoder_attention_mask=extended_claim_attention_mask)[0]

        claim_emb = claim_hidden_states[:, 0, :]

        return claim_hidden_states, claim_emb


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
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def message_passing(self, self_emb, neigh_emb, act, mode):

        if mode == 'train':
            self_emb = self.dropout_layer(self_emb)
            neigh_emb = self.dropout_layer(neigh_emb)
        # self_emb = self.linear_layers[0](self_emb)
        # neigh_emb = self.linear_layers[1](neigh_emb)

        att = self.att(self_emb, neigh_emb)
        neigh_emb_agg = self.agg(self_emb, neigh_emb, att, act)

        return neigh_emb_agg

    def att(self, self_emb, neigh_emb):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        emb_concat = torch.cat([self_emb, neigh_emb], dim=1)
        att = torch.reshape(self.att_layer[0](emb_concat), [-1, n])
        att = self.softmax(self.leaky_relu(att))  # may need to change sigmoid here

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


class EvidLevelAggregator(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(EvidLevelAggregator, self).__init__()
        self.bert_hidden_size = bert_config.hidden_size
        self.vit_hidden_size = vit_config.hidden_size
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images

        self.generate_modules(bert_config, vit_config, args)

    def generate_modules(self, bert_config, vit_config, args):

        self.linear_layers = nn.ModuleList([nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size),
                                            nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size),
                                            nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size)])
        self.graph_conv_layers = nn.ModuleList([GraphConvLayer(self.bert_hidden_size, self.bert_hidden_size) for _ in range(3)])
        self.weight_layer = nn.ModuleList([nn.Linear(2 * self.bert_hidden_size, 1)])
        self.identity = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, claim_emb, evid_text_emb, evid_image_emb, mode):

        # text aggregation
        evid_text_emb_agg = self.graph_conv_layers[0](claim_emb, evid_text_emb, self.identity, mode)

        # hierarchical image aggregation
        claim_emb_repeat = torch.reshape(torch.tile(torch.unsqueeze(claim_emb, dim=1), [1, self.num_sampled_evid_texts, 1]), evid_text_emb.size())
        query_emb = self.linear_layers[0](torch.concat([claim_emb_repeat, evid_text_emb], dim=-1))
        evid_image_emb_agg = self.graph_conv_layers[1](query_emb, evid_image_emb, self.identity, mode)

        query_emb = self.linear_layers[1](torch.concat([claim_emb, evid_text_emb_agg], dim=-1))
        evid_image_emb_agg = self.graph_conv_layers[2](query_emb, evid_image_emb_agg, self.identity, mode)

        # text and image aggregation
        evid_emb = self.linear_layers[2](torch.concat([evid_text_emb_agg, evid_image_emb_agg], dim=-1))

        return evid_emb


class TwoLevelAggregator(nn.Module):

    def __init__(self, bert_config, vit_config, args):

        super(TwoLevelAggregator, self).__init__()
        self.bert_hidden_size = bert_config.hidden_size
        self.vit_hidden_size = vit_config.hidden_size
        self.num_sampled_evid_texts = args.num_sampled_evid_texts
        self.num_sampled_images = args.num_sampled_images
        self.evidence_provided = args.evidence_provided

        self.generate_modules(bert_config, vit_config, args)

    def generate_modules(self, bert_config, vit_config, args):

        self.emb_proj_layers = nn.ModuleList([nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
                                              nn.Linear(self.vit_hidden_size, self.bert_hidden_size)])
        self.token_level_aggregator = nn.ModuleList([TokenLevelAggregator(bert_config, vit_config, args)])
        self.evid_level_aggregator = nn.ModuleList([EvidLevelAggregator(bert_config, vit_config, args)])
        self.bert_att_layers = nn.ModuleList([BertSelfAttention(bert_config)])
        self.linear_layers = nn.ModuleList([nn.Linear(2 * self.bert_hidden_size, self.bert_hidden_size)])
        self.att_layer = nn.ModuleList([nn.Linear(self.bert_hidden_size, 1)])
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def claim_text_and_image_aggregation(self, claim_text_hidden_states, claim_image_hidden_states):

        [num_images, num_patches, bert_emb_dim] = claim_image_hidden_states.size()

        claim_image_hidden_states = torch.reshape(claim_image_hidden_states, [-1, self.num_sampled_images, num_patches, self.bert_hidden_size])
        claim_image_hidden_states_agg_list = []
        for i in range(self.num_sampled_images):
            claim_image_hidden_states_agg = self.bert_att_layers[0](claim_text_hidden_states, encoder_hidden_states=claim_image_hidden_states[:, i, :, :])[0]
            claim_image_hidden_states_agg = torch.unsqueeze(claim_image_hidden_states_agg, dim=0)
            claim_image_hidden_states_agg_list.append(claim_image_hidden_states_agg)
        claim_image_hidden_states_agg_mean = torch.mean(torch.concat(claim_image_hidden_states_agg_list, dim=0), dim=0)
        claim_hidden_states = self.linear_layers[0](torch.concat([claim_text_hidden_states, claim_image_hidden_states_agg_mean], dim=-1))
        claim_hidden_states = claim_hidden_states + claim_text_hidden_states

        return claim_hidden_states

    def ranking_aware_attention(self, claim_emb, base=0.5):  # base may need to change

        att = torch.reshape(self.att_layer[0](claim_emb), [-1, self.num_sampled_evid_texts])
        att = self.sigmoid(att)  # may need to change sigmoid
        scalar = torch.pow(base, torch.arange(start=0, end=self.num_sampled_evid_texts, dtype=att.dtype, device=att.device))
        scalar = torch.tile(torch.unsqueeze(scalar, dim=0), [att.size(0), 1])
        att = self.softmax(torch.multiply(scalar, att))
        att = torch.unsqueeze(att, dim=1)

        claim_emb = torch.reshape(claim_emb, [att.size(0), self.num_sampled_evid_texts, -1])
        claim_emb_agg = torch.squeeze(torch.matmul(att, claim_emb))

        return claim_emb_agg

    def forward(self, claim_text_hidden_states, claim_image_hidden_states, evid_text_hidden_states, evid_attention_mask, evid_image_hidden_states, claim_attention_mask, mode):

        claim_text_hidden_states = self.emb_proj_layers[0](claim_text_hidden_states)
        if claim_image_hidden_states is not None:
            claim_image_hidden_states = self.emb_proj_layers[1](claim_image_hidden_states)
        evid_text_hidden_states = self.emb_proj_layers[0](evid_text_hidden_states)
        evid_image_hidden_states = self.emb_proj_layers[1](evid_image_hidden_states)

        evid_text_emb = evid_text_hidden_states[:, 0, :]
        evid_image_emb = evid_image_hidden_states[:, 0, :]

        [num_evid_texts, num_tokens, bert_emb_dim] = evid_text_hidden_states.size()
        [num_evid_images, num_patches, bert_emb_dim] = evid_image_hidden_states.size()

        # token-level aggregation
        # if claim_image_hidden_states is not None:
        #     claim_hidden_states = self.claim_text_and_image_aggregation(claim_text_hidden_states, claim_image_hidden_states)
        # else:
        #     claim_hidden_states = claim_text_hidden_states

        claim_hidden_states = claim_text_hidden_states
        claim_hidden_states = torch.reshape(torch.tile(torch.unsqueeze(claim_hidden_states, dim=1), [1, self.num_sampled_evid_texts, 1, 1]), evid_text_hidden_states.size())
        claim_attention_mask = torch.reshape(torch.tile(torch.unsqueeze(claim_attention_mask, dim=1), [1, self.num_sampled_evid_texts, 1]), [-1, claim_attention_mask.size(-1)])
        claim_hidden_states, claim_emb = self.token_level_aggregator[0](claim_hidden_states, claim_attention_mask, evid_text_hidden_states, evid_image_hidden_states)

        if self.evidence_provided == 'gold':
            claim_emb = torch.reshape(claim_emb, [-1, self.num_sampled_evid_texts, bert_emb_dim])
            claim_emb = torch.mean(claim_emb, dim=1)
            claim_emb = torch.reshape(claim_emb, [-1, bert_emb_dim])
        else:
            claim_emb = self.ranking_aware_attention(claim_emb)
            claim_emb = torch.reshape(claim_emb, [-1, bert_emb_dim])

        # evidence-level aggregation
        evid_emb = self.evid_level_aggregator[0](claim_emb, evid_text_emb, evid_image_emb, mode)

        return claim_emb, evid_emb
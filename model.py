import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CompGCNLayer
from decoder import *
from utils import build_candidate_subgraph
from gcn_layer import CandRGCNLayer
import dgl.function as fn

class RGCNCell(nn.Module):
    def __init__(self, h_dim, out_dim, num_rels,
                 num_hidden_layers=1, dropout=0,  rel_emb=None):
        super(RGCNCell, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rel_emb = rel_emb.cuda()

        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers):
            h2h = UnionRGCNLayer(self.h_dim, self.out_dim, self.num_rels, activation=F.rrelu, dropout=self.dropout, rel_emb=self.rel_emb)
            self.layers.append(h2h)

    def forward(self, g, init_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        for i, layer in enumerate(self.layers):
            layer(g, [])
        return g.ndata.pop('h')

class CompGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation, dropout):
        super(CompGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CompGCNLayer(in_feats, n_hidden, activation, 0.))
        for i in range(n_layers - 1):
            self.layers.append(CompGCNLayer(n_hidden, n_hidden, activation, dropout))
        self.layers.append(CompGCNLayer(n_hidden, out_feats, None, dropout))

    def forward(self, features, relations, g):
        h = features
        for layer in self.layers:
            h = layer(h, relations, g)
        return h

class GatingMechanism(nn.Module):
    def __init__(self, num_e, num_rel, h_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.Tensor(num_e, h_dim), requires_grad=True).float()
        self.num_rels = num_rel
        nn.init.xavier_uniform_(self.gate_theta)
        self.linear = nn.Linear(h_dim, 1)

    def forward(self, X: torch.LongTensor, Y: torch.LongTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.linear(self.gate_theta[X]))
        return gate


class NET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(NET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.eps = 1e-8
        self.h_dim = args.embedding_dim
        self.gpu='cuda:0'
        self.num_ents = num_e

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel + 1, args.embedding_dim))  # rel_embeds[0] for self-loop
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.comp_gcn = CompGCN(args.embedding_dim, args.embedding_dim, args.embedding_dim, args.graph_layer, F.relu,
                                0.2)
        self.gru_cell = nn.GRUCell(args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()

        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)
        self.tanh = nn.Tanh()
        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.decoder_ob1 = TimeConvTransE(num_e, args.embedding_dim, 0.3, 0.3, 0.3)
        self.decoder_ob2 = TimeConvTransE(num_e, args.embedding_dim, 0.3, 0.3, 0.3)
        self.rdecoder_re1 = TimeConvTransR(num_rel, args.embedding_dim, 0.3, 0.3, 0.3)
        self.rdecoder_re2 = TimeConvTransR(num_rel, args.embedding_dim, 0.3, 0.3, 0.3)

        self.cand_layer_s = CandRGCNLayer(args.embedding_dim, args.embedding_dim, self.num_rel, 100,
                                          activation=F.rrelu, dropout=0.2, self_loop=True, skip_connect=False)
        self.cand_layer_o = CandRGCNLayer(args.embedding_dim, args.embedding_dim, self.num_rel, 100,
                                          activation=F.rrelu, dropout=0.2, self_loop=True, skip_connect=False)

        self.gru_s = nn.GRUCell(2 * args.embedding_dim, args.embedding_dim)
        self.gru_o = nn.GRUCell(2 * args.embedding_dim, args.embedding_dim)

        self.time_linear1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.time_linear2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.decoder1 = TimeConvTransE(self.num_e, args.embedding_dim, 0.2, 0.2, 0.2)
        self.decoder2 = TimeConvTransE(self.num_e, args.embedding_dim, 0.2, 0.2, 0.2)
        self.k = 45
        self.time_interval = args.timestamps
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.sin = torch.sin
        self.gamma = 0.2
        self.gate = GatingMechanism(self.num_e, self.num_rel, args.embedding_dim)
        self.time_gate = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.rgcn = RGCNCell(args.embedding_dim,
                             args.embedding_dim,
                             num_rel * 2,
                             2,
                             0.2,
                             self.rel_embeds[1:].cuda())

        self.alpha_t = torch.nn.Parameter(torch.Tensor(num_e, args.embedding_dim), requires_grad=True).float()
        self.beta_t = torch.nn.Parameter(torch.Tensor(num_e, args.embedding_dim), requires_grad=True).float()
        self.temporal_w = torch.nn.Parameter(torch.Tensor(args.embedding_dim*2, args.embedding_dim), requires_grad=True).float()
        self.time_gate = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.alpha = 0.5
        self.pi = 3.14159265358979323846
        self.static_emb = torch.nn.Parameter(torch.Tensor(num_e, args.embedding_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.alpha_t)
        torch.nn.init.normal_(self.beta_t)
        torch.nn.init.normal_(self.temporal_w)
        print('RSPED Initiated')

    def forward_my(self, quadruples, his_g, err_mat):
        last_h, current_h = None, None
        self.inputs = [F.normalize(self.get_dynamic_emb(quadruples[0, 3]))]
        # self.inputs = [F.normalize(self.entity_embeds)]

        for g in his_g:
            envolve_embs = self.comp_gcn(self.entity_embeds, self.rel_embeds, g)
            # envolve_embs = self.comp_gcn(self.inputs[0], self.rel_embeds, g)
            if last_h is None:
                current_h = self.gru_cell(envolve_embs)
                last_h = current_h
                self.inputs.append(current_h)
            else:
                current_h = self.gru_cell(envolve_embs, last_h)
                self.inputs.append(current_h)

        related_emb = torch.spmm(err_mat, self.rel_embeds[1:])
        if current_h is None:
            current_h = self.entity_embeds + related_emb
            # current_h = self.inputs[0] + related_emb
        else:
            current_h += related_emb
        current_h = self.get_composed(current_h, related_emb)
        return current_h

    def forward(self, batch_block, his_g, mode_lk, total_data=None):
        quadruples, s_frequency, o_frequency, s_history_label_true, o_history_label_true, err_mat, err_mat_inv = batch_block
        err_mat = err_mat.cuda()

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax

        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        current_h = self.forward_my(quadruples, his_g, err_mat)
        current_h_2 = self.forward_my(quadruples, his_g, err_mat_inv)

        self.fact_time = self.get_init_time(quadruples)

        if mode_lk in ['Training', 'Valid']:
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, current_h, self.rel_embeds[1:self.num_rel + 1],
                                                          self.linear_pred_layer_s, self.linear_pred_layer_s2,
                                                          s_history_tag, s_frequency, s_non_history_tag, self.fact_time)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, current_h_2, self.rel_embeds[self.num_rel + 1:],
                                                          self.linear_pred_layer_o, self.linear_pred_layer_o2,
                                                          o_history_tag, o_frequency, o_non_history_tag, self.fact_time)

            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds[1:self.num_rel + 1], s_history_label_true,
                                                 s_frequency_hidden, current_h)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds[self.num_rel + 1:], o_history_label_true,
                                                 o_frequency_hidden, current_h_2)

            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            total_loss = nce_loss + spc_loss
            return total_loss

        elif mode_lk in ['Test']:
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, current_h, self.rel_embeds[1:self.num_rel + 1],
                                                           self.linear_pred_layer_s, self.linear_pred_layer_s2,
                                                           s_history_tag, s_frequency, s_non_history_tag,
                                                           self.fact_time)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, current_h_2, self.rel_embeds[self.num_rel + 1:],
                                                           self.linear_pred_layer_o, self.linear_pred_layer_o2,
                                                           o_history_tag, o_frequency, o_non_history_tag,
                                                           self.fact_time)


            sub_rank = self.link_predict(s_preds, s, o, r, total_data, 's')
            obj_rank = self.link_predict(o_preds, o, s, r, total_data, 'o')

            return sub_rank, obj_rank,

        else:
            print("Invalid mode!")
            exit()

    def get_dynamic_emb(self,t):
        # return self.static_emb
        timevec = self.alpha * self.alpha_t*t + (1-self.alpha) * torch.sin(2 * self.pi * self.beta_t*t)
        attn = torch.cat([self.entity_embeds, timevec],1)
        return torch.mm(attn, self.temporal_w)

    def get_composed(self, cur_output, related_emb):
        self.time_weights = []
        for i in range(len(self.inputs)):
            self.time_weights.append(self.time_gate(self.inputs[i]+related_emb).cuda())
        self.time_weights.append(torch.zeros(self.num_ents,self.h_dim).cuda())
        self.time_weights = torch.stack(self.time_weights,0)
        self.time_weights = torch.softmax(self.time_weights,0)
        output = cur_output*self.time_weights[-1]
        for i in range(len(self.inputs)):
            output += self.time_weights[i]*self.inputs[i]
        return F.normalize(output)

    def get_composed2(self, cur_output, related_emb):
        self.time_weights = []
        for i in range(len(self.inputs)):
            self.time_weights.append(self.time_gate(self.inputs[i]+related_emb).cuda())
        self.time_weights.append(torch.zeros(self.num_ents,self.h_dim).cuda())
        self.time_weights = torch.stack(self.time_weights,0)
        self.time_weights = torch.softmax(self.time_weights,0)
        output = cur_output*self.time_weights[-1]
        for i in range(len(self.inputs)):
            output += self.time_weights[i]*self.inputs[i]
        return F.normalize(output)

    def calculate_nce_loss(self, actor1, actor2, r, current_h, rel_embeds, pred_layer, pred_layer2, history_tag,
                           frequency, non_history_tag, fact_time):
        if current_h is not None:
            sub_emb = current_h[actor1]
            obj_emb = current_h
        else:
            sub_emb = self.entity_embeds[actor1]
            obj_emb = self.entity_embeds

        h = pred_layer(self.dropout(torch.cat((sub_emb, rel_embeds[r]), dim=1)))
        h = torch.tanh(h)

        h2 = pred_layer2(self.dropout(torch.cat((sub_emb, rel_embeds[r]), dim=1)))
        h2 = torch.tanh(h2)

        gate = self.gate(actor1, r)
        preds1 = F.softmax(torch.mm(h, obj_emb.transpose(0, 1)) + history_tag, dim=1)
        preds2 = F.softmax(torch.mm(h2, obj_emb.transpose(0, 1)) + non_history_tag, dim=1)
        preds_two = (torch.mul(gate, preds1) + torch.mul(-gate + 1, preds2))

        nce = torch.sum(torch.gather(torch.log(preds_two), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]
        if torch.any(torch.isnan(nce)):
            nce = torch.tensor(0.0, requires_grad=True)
        return nce, preds_two

    def link_predict(self, preds, actor1, actor2, r, all_triples, pred_known):
        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return ranks

    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden, current_h):
        if current_h is None:
            current_h = self.entity_embeds
        projections = self.contrastive_layer(
            torch.cat((current_h[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return torch.tensor(0.0, requires_grad=True)
        return supervised_contrastive_loss

    def contrastive_layer(self, x):
        x = self.contrastive_hidden_layer(x)
        return x

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def calculate_cand_loss(self, actor1, actor2, r, rel_embeds, preds, cand_layer, decoder, gru, gate, current_h,
                            frequency):
        # ! add a candidate graph based on the prediction result of preds1 and preds2 5&
        if current_h is None:
            current_h = self.entity_embeds
        triples = [actor1, r]
        cand_graph = build_candidate_subgraph(self.num_e, triples, preds, self.k, 1)
        total_feature = cand_layer.forward(cand_graph, current_h, rel_embeds, self.k)
        avg_feature = torch.split_with_sizes(total_feature, cand_graph.batch_num_nodes().tolist())
        neigh_feats = torch.stack(avg_feature, dim=0).mean(dim=0)
        enhanced_ent_emb = neigh_feats
        # enhanced_ent_emb[actor1] = gate(torch.cat([enhanced_ent_emb[actor1], self.fact_time], dim=1))
        enhanced_ent_emb = gru(torch.cat([enhanced_ent_emb, current_h], dim=1), enhanced_ent_emb)

        entity_feature = F.normalize(enhanced_ent_emb)

        score_enhanced = F.softmax(decoder.forward(entity_feature, actor1, rel_embeds[r], self.fact_time), dim=1)
        # score_enhanced = F.softmax(decoder.forward(entity_feature, actor1, rel_embeds[r]), dim=1)

        preds_total = self.gamma * score_enhanced + (1 - self.gamma) * preds
        # preds_total = preds_total * frequency
        # preds_total = 0.001 * preds_total + 0.999 * frequency

        scores_enhanced = torch.log(score_enhanced)
        loss_cand = F.nll_loss(scores_enhanced, actor2) + self.regularization_loss(reg_param=0.01)

        pred_actor2 = torch.argmax(preds_total, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        # print('# Batch accuracy', accuracy)
        if torch.any(torch.isnan(loss_cand)):
            loss_cand = torch.tensor(0.0, requires_grad=True)
        return loss_cand, preds_total

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(
            self.entity_embeds.pow(2))  # + torch.mean(self.time_embeds.pow(2))
        return regularization_loss * reg_param


def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None, dropout=0.0, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation

        self.num_rels = num_rels
        self.emb_rel = rel_emb
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h):
        masked_index = torch.masked_select(
            torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
            (g.in_degrees(range(g.number_of_nodes())) > 0))
        loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        relation = self.emb_rel.index_select(0, edges.data['rel']).view(-1, self.out_feat)
        edge_type = edges.data['rel']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)

        msg = node + relation

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.gnn import GraphAttentionLayer

class GraphAttention(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GraphAttention, self).__init__()
        self.dropout = dropout
 
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
 
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class ASGAT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGAT, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ga = GraphAttention(opt.hidden_dim*2,opt.hidden_dim*2,opt.dropout,opt.alpha,opt.nheads)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        #将句子中的aspect对应的位置设为1，其余位置设为0
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        px = self.position_weight(text_out, aspect_double_idx, text_len, aspect_len)
        x = F.elu(self.ga(px, adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class ASGAT2(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGAT2, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ga = GraphAttention(opt.hidden_dim*2,opt.hidden_dim*2,opt.dropout,opt.alpha,opt.nheads)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        # self.text_embed_dropout = nn.Dropout(opt.dropout)

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        # text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.elu(self.ga(text_out, adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class ASGAT3(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGAT3, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ga = GraphAttention(opt.hidden_dim*2,opt.hidden_dim*2,opt.dropout,opt.alpha,opt.nheads)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(opt.dropout)

    def forward(self, inputs , targets, criterion):
        text_indices, aspect_terms, adj = inputs
        polarity = targets
        text_len = torch.sum(text_indices != 0, dim=-1)
        text = self.embed(text_indices.to(self.opt.device))
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.elu(self.ga(text_out, adj.to(self.opt.device)))
        batches = x.shape[0]
        loss = torch.tensor(0.).to(self.opt.device)
        for i in range(batches):
            aspects = []
            for a in aspect_terms[i]:
                aspects.append(x[i][a[0]:a[1]].mean(dim=0).squeeze())

            predicts = []
            for a in aspects:
                predicts.append(self.fc(a))
            loss_ce = criterion(torch.stack(predicts,dim=0),torch.tensor(polarity[i]).to(self.opt.device))
            loss_neu = self.__get_loss__(aspects,polarity[i])
            loss += loss_ce# + self.opt.lamda * loss_neu

        return loss/batches

    def predict(self,inputs):
        text_indices, aspect_terms, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text = self.embed(text_indices.to(self.opt.device))
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.elu(self.ga(text_out, adj.to(self.opt.device)))
        batches = x.shape[0]
        predicts = []
        for i in range(batches):
            aspects = []
            for a in aspect_terms[i]:
                aspects.append(x[i][a[0]:a[1]].mean(dim=0).squeeze())
            for a in aspects:
                predicts.append(torch.argmax(self.fc(a), -1))
        predicts = torch.tensor(predicts).to(self.opt.device)
        return predicts


    def __get_loss__(self,aspects,polarities):
        # compute loss_neu for every sentence
        loss_neu = torch.tensor(0.).to(self.opt.device)
        count = 0
        neu_set = []
        orther_set = []
        for i in range(len(polarities)):
            if polarities[i] == 0.:
                neu_set.append(i)
            else:
                orther_set.append(i)
        for j in neu_set:
            for k in orther_set:
                loss_neu += self.__dif__(aspects[j],aspects[k])
                count += 1
        return loss_neu/count if count>0 else loss_neu

    def __dif__(self,x,y,norm=True):
        # compute the difference between aj and ak
        # print(F.cosine_similarity(x,y,dim=0))
        return F.cosine_similarity(x,y,dim=0)

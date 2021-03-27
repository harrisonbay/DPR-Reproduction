import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys
from transformers import BertTokenizer, BertModel, BertForMaskedLM  
from transformers import DistilBertTokenizer, DistilBertModel

class BERT_QA(nn.Module):
    def __init__(self, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):

        super(BERT_QA, self).__init__()


        self.tokenizer = tokenizer  
        print("Init")
        # encoder for question
        self.bert_q = BertModel.from_pretrained('./bert-base-uncased')
        print("Q")
        # encoder for passage
        self.bert_p = BertModel.from_pretrained('./bert-base-uncased')
        print("P")

    def forward(self,x_q, x_p):
        if x_q != None:
            # For BERT in Huggingface, the 1-indexed item in the output tuple is
            # actually the embedding of the [CLS] (classification) token for
            # each question in the batch. They make it easy.
            x_q = self.bert_q(x_q)[1]  
            x_q = x_q.view(-1, 768)

        if x_p != None:
            x_p = self.bert_p(x_p)[1] 
            x_p = x_p.view(-1, 768)

        return (x_q, x_p)


    def get_sim(self, q, psg):
    
        # q.shape = (N, 768) 
        # psg.shape = (2N, 768), odds -> negs, evens -> golds
        # gram.shape = (N, 2N)
    
        gram = q @ torch.transpose(psg, 1, 0)
        sim = F.log_softmax(gram, dim=1)
       
        idx = torch.Tensor([2*i for i in range(sim.shape[0])]).long().to(sim.device)
        return sim,idx

    def loss_fn(self, sim, idx, reduction='mean'):
        
        return F.nll_loss(sim, idx, reduction=reduction)


class DISTILBERT_QA(nn.Module):
    def __init__(self, tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')):

        super(DISTILBERT_QA, self).__init__()


        self.tokenizer = tokenizer  
        print("Init")
        # encoder for question
        self.bert_q = DistilBertModel.from_pretrained('distilbert-base-uncased')
        print("Q")
        # encoder for passage
        self.bert_p = DistilBertModel.from_pretrained('distilbert-base-uncased')
        print("P")

    def forward(self,x_q, x_p):
        if x_q != None:
            # For DistilBERT, however, they don't make it as easy. If our
            # (in-GPU) batch size is 8, then the 0-indexed item in the output
            # tuple is the tensor of all of the outputs, so in the case of our
            # questions (which have input length 64 [`train_data_loader.py`]),
            # our shape is [8, 64, 768] since BERT embeds in 768 dimensions. We
            # then want to pull out the embedding for the [CLS] token--this is
            # always located at the beginning of our tokenized question.
            x_q = self.bert_q(x_q)[0][:, 0, :]  
            x_q = x_q.view(-1, 768)

        if x_p != None:
            x_p = self.bert_p(x_p)[0][:, 0, :]
            x_p = x_p.view(-1, 768)

        return (x_q, x_p)


    def get_sim(self, q, psg):
    
        # q.shape = (N, 768) 
        # psg.shape = (2N, 768), odds -> negs, evens -> golds
        # gram.shape = (N, 2N)
    
        gram = q @ torch.transpose(psg, 1, 0)
        sim = F.log_softmax(gram, dim=1)
       
        idx = torch.Tensor([2*i for i in range(sim.shape[0])]).long().to(sim.device)
        return sim,idx

    def loss_fn(self, sim, idx, reduction='mean'):
        
        return F.nll_loss(sim, idx, reduction=reduction)

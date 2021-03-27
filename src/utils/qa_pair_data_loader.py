import pandas as pd
import torch.utils.data
# import numpy as np  # necessary for the testing at the bottom
import torch
import os
from ast import literal_eval

from transformers import BertTokenizer, BertModel, BertForMaskedLM


class QAPairDataset(torch.utils.data.Dataset):

    def __init__(self, path):

        # nq-dev.csv or nq-test.csv or (I think) the triviaQA ones would work
        # here. We didn't actually inspect the triviaQA ones though, just the
        # NQ ones
        self.df = pd.read_csv(path, sep='\t', names=['question', 'answer'],
                              converters={'question': str, 'answer': eval})
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(len(self.df.index))
        
    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        ques = self.df['question'][index]
        ques_token = self.tokenizer.encode(ques, add_special_tokens=True, max_length=64,
                                           padding='max_length', truncation=True)

        return torch.Tensor(ques_token), index


# dataset = QAPairDataset(r"../test/head-test.csv")
# print(len(dataset))
# loader = torch.utils.data.DataLoader(dataset, batch_size=1)

# print(type(loader.dataset.df["answer"][1]))
# print(loader.dataset.df["answer"][1])

# for x, y, z in loader:
#     print(x)
#     print(y)
#     print(z)
#     print(z.shape)
#     print(np.expand_dims(z.numpy(), axis=1).shape)


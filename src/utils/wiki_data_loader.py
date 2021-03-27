import pandas as pd
import torch.utils.data
import torch
import os

from transformers import BertTokenizer, BertModel, BertForMaskedLM

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        
        # psgs_w100.tsv or psgs_w100_subset.tsv
        # There's an article with the title "NaN" that gets parsed to a float...
        self.df = pd.read_csv(path, sep='\t',
                              names=['junk', 'passage', 'title'],
                              converters={'passage': str, 'title': str})
        self.df.drop(columns=['junk'], inplace=True)
        self.df.reset_index(level=0, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def  __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        passage = self.df['passage'][index]
        title = self.df['title'][index]
        passage_token = self.tokenizer.encode(title, text_pair=passage,
                                              add_special_tokens=True, max_length=256,
                                              padding='max_length', truncation=True)

        return torch.Tensor(passage_token), index

# dataset = WikiDataset(r"../test/test.tsv")
# loader = torch.utils.data.DataLoader(dataset, batch_size=2)

# print(type(loader.dataset.df['passage'][[1, 3, 5]][1]))

# for x, y in loader:
#     print(x.shape, "\n")
#     print(x)


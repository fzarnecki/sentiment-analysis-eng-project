from utils.utils import get_tokenizer
from torch.utils.data import Dataset
from collections import Counter
from typing import List
import pandas as pd
import torch

 
class SentimentDataset(Dataset):

    def __init__(self, dataset_path, lang="pl"):
        text, label = self.__load_sentiment_csv(dataset_path)
        data = [text, label]

        self.tokenizer = get_tokenizer(lang)

        # TODO above data field probably useless 
        self.tokens = self.tokenizer(data[0], padding = True, truncation = True, return_tensors='pt')
        self.data = [self.tokens, label, text]
        self.transform = None

        print("Loaded dataset.")
        self.num_classes()


    def num_classes(self):
        print("Classes number: ", Counter(self.data[1]))
               

    def get_num_classes(self):
        return len(Counter(self.data[1]))
            

    def __len__(self):
        return len(self.data[1])
   

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [(self.data[0]['attention_mask'][idx], self.data[0]['input_ids'][idx], self.data[2][idx]), self.data[1][idx]]    
        if self.transform:
            sample = self.transform(sample)
        return sample


    def __delitem__(self, key):
        del self.data[0][key]
        del self.data[1][key]


    def __load_sentiment_csv(self, dataset_path:List[str], verbose:bool=False):
        if not dataset_path:
            raise Exception("__load_sentiment_csv: dataset not provided")
        
        print(f"Loading {dataset_path} dataset.")
        
        data_list = [pd.read_csv(path, header=None, engine='python') for path in dataset_path]
        data = pd.concat(data_list, ignore_index=True)
        data = data.sample(frac=1)
        for l in data_list:
            print(len(l))
        print(len(data))
        print(data.head())

        text = [str(x) for x in data[0].tolist()]
        label = data[1].astype(int).tolist()
        return text, label

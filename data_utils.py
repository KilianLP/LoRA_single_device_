#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:42:34 2024

@author: kilianpreuss
"""

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import random


 
class DataSet(Dataset):
    def __init__(self,data_origin,data_target,tra_token):
        super().__init__()
        
        self.data_origin = data_origin
        self.data_target = data_target
        self.tra_token = tra_token
        
    def __getitem__(self,idx):
        
        return torch.tensor(self.data_origin[idx] + self.tra_token + self.data_target[idx]), torch.tensor([128255 for _ in range(len(self.data_origin[idx] + self.tra_token)-1)] + self.data_target[idx] + [128255] )
    
    def len(self):
        return len(self.data_origin)
    

class DataLoader():
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.idx = [i for i in range(dataset.len())] 
        self.data = []
        
    def __iter__(self):
        random.shuffle(self.idx)
        self.current_idx = 0
        
        return self
        
    def __next__(self):
        if (self.current_idx + 1)*self.batch_size >= len(self.idx):
            raise StopIteration
            
        source_batch  = [self.dataset[i][0] for i in self.idx[self.current_idx*self.batch_size:(self.current_idx + 1)*self.batch_size]]
        target_batch = [self.dataset[i][1] for i in self.idx[self.current_idx*self.batch_size:(self.current_idx + 1)*self.batch_size]]
        
        padded_source = pad_sequence(source_batch, batch_first=True, padding_value=128255)
        padded_target = pad_sequence(target_batch, batch_first=True, padding_value=128255)
        
        self.current_idx += 1
        
        return padded_source, padded_target
    
    
    

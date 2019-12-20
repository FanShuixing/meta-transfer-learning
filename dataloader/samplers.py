##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Sampler for dataloader. """
import torch
import numpy as np

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, label, n_batch, n_cls, n_per):
        '''

        :param label:[0,0,0,1,1,2,3,...]
        :param n_batch:
        :param n_cls:
        :param n_per:

        self.m_ind:[tensor([0,1,2]),tensor([3,4]),tensor([5]),tensor([6])]
        '''
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            #len(self.m_ind)表示数据中的类别数目
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]#只挑选前n_cls个类
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]#只随机挑选n_per个数据
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            #shape:[self.n_cls*self.n_per]
            yield batch #?这样子返回的数据怎么知道其对应的是哪个类别呢
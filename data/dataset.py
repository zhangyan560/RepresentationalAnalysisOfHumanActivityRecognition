import numpy as np
import torch
from scipy import stats
import pandas as pd

class HAR_dataset():
    def __init__(self, datapath,transform,target_transform):
        self.datapath = datapath
        self.transform = transform
        self.target_transform = target_transform
    def build_data(self,window_size=30,step=10,downsample=1):
        self.read_data(downsample=downsample)
        inputs = []
        targets = []
        counter = 0
        while counter+window_size < len(self.data['inputs']):
            inputs += [self.data['inputs'][counter:window_size+counter,:]]
            targets += [stats.mode(self.data['targets'][counter:window_size+counter],axis=None).mode[0]]
            # targets += [self.data['targets'][window_size+counter]]
            counter += step
        self.data = {'inputs': np.asarray(inputs).transpose(0,2,1), 'targets': np.asarray(targets, dtype=int)}
    def __getitem__(self,index):
        x,y = self.transform(self.data['inputs'][index]),self.target_transform(self.data['targets'][index].reshape(1,-1))
        return x,y
    def input_shape(self):
        return self.data['inputs'].shape[1::]
    def nb_classes(self):
        return int(max(self.data['targets'])+1)
    def __len__(self):
        return len(self.data['inputs'])
    def get_data_weights(self):
        class_count = np.zeros((self.nb_classes(),)).astype(float)
        for i in range(self.nb_classes()):
            class_count[i] = len(np.where(self.data['targets'] == i)[0])
        # weights = (1 / torch.from_numpy(class_count).type(torch.DoubleTensor))
        class_weights = [sum(class_count)/c for c in class_count]
        example_weights = [class_weights[e] for e in self.data['targets']]

        return example_weights

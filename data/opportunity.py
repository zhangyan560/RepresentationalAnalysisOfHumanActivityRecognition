import numpy as np
import csv
import h5py
import simplejson as json
from tqdm import tqdm
from scipy import stats
import torch
from data.dataset import HAR_dataset

# default_path = './data_source/OpportunityUCIDataset/'
default_path = './data/data_source/OpportunityUCIDataset/'
default_frequency = 30
default_test_indexes = ['S2-ADL4.dat','S2-ADL5.dat','S3-ADL4.dat','S3-ADL5.dat']

class OpportunityDataset(HAR_dataset):
    def __init__(self,subjects,datapath=default_path,transform=None,target_transform=None):
        super(OpportunityDataset,self).__init__(datapath=datapath,transform=transform,target_transform=target_transform)
        self.files = np.asarray([['S1-ADL1.dat','S1-ADL2.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat'],
                      ['S2-ADL1.dat', 'S2-ADL2.dat','S2-ADL3.dat', 'S2-ADL4.dat','S2-ADL5.dat', 'S2-Drill.dat'],
                      ['S3-ADL1.dat', 'S3-ADL2.dat','S3-ADL3.dat', 'S3-ADL4.dat','S3-ADL5.dat', 'S3-Drill.dat'],
                      ['S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']])
        self.files = [self.files[i,j] for i,j in subjects]
    def read_data(self,downsample=1):
        files = self.files.copy()
        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        label2id = {str(x[0]): i for i, x in enumerate(label_map)}
        id2label = [x[1] for x in label_map]

        cols = [
            38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
        cols = [x-1 for x in cols] # labels for 18 activities (including other)
        self.read_files(files, cols, label2id)
        self.id2label = id2label
        self.label2id = label2id
        if downsample > 1 and downsample != 0:
            self.data['inputs'] = self.data['inputs'][::downsample,:]
            self.data['targets'] = self.data['targets'][::downsample]
    def read_files(self, filelist, cols, label2id):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            nancnt = 0
            with open('/content/drive/MyDrive/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(label2id[elem[-1]])
        self.data = {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


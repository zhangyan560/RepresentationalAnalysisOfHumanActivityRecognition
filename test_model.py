from models import resnets,train, ResNet, FCN, ResNet2
import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import utils
from collections import namedtuple
import json
import _pickle as pickle
from models import test
dataset = 'pamap2'
parser = argparse.ArgumentParser()
parser.add_argument('--saving_folder', type=str, default = '/content/drive/MyDrive/HistoryOf' + dataset)
parser.add_argument('--confusion_matrix',action='store_true',default=True)
parser.add_argument('--read_history',action='store_true',default=False)
parser.add_argument('--f1_mean',action='store_true',default=False)
args = parser.parse_args()

run_info = json.load(open(os.path.join(args.saving_folder,'run_info.json'),'r'))
args = namedtuple('Struct', run_info.keys())(*run_info.values())
print("Loading data")
print(args.dataset)
datasets = utils.get_datasets(args.dataset,validation=True,window_size=args.window_size,
                              step=args.window_step,downsample=args.downsample_factor)
testing_loader = DataLoader(dataset=datasets['testing_set'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                            pin_memory=True)

random_sample = next(iter(testing_loader))[0]
# model = resnets.HAR_ResNet1D(input_channels=random_sample.shape[1],
#                              kernel_size=args.kernel_size,
#                              depth=[int(item) for item in args.architecture_depth.split(',')],
#                              dilated=args.dilated,nb_classes=datasets['training_set'].nb_classes())
# model.build_classifier(random_sample)
model = ResNet2.resnetHAR(layers = [int(item) for item in args.architecture_depth.split(',')],
                kernel_size=args.kernel_size,
                input_channel=random_sample.shape[1],
                nb_classes=datasets['training_set'].nb_classes())
model.build_classifier(random_sample)
# model = ResNet.resnet34(num_classes=datasets['training_set'].nb_classes(), input_channels=random_sample.shape[1])

model.load_state_dict(torch.load(os.path.join(args.saving_folder,'modelbest_.pth.tar')))
# model = FCN.FCN(input_channels=random_sample.shape[1], nb_classes=datasets['training_set'].nb_classes())
# model.build_classifier(random_sample)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
tester = test.Tester(model,id2label=datasets['training_set'].id2label,test_loader=testing_loader,criterion=criterion,
                        verbose=True,saving_folder=args.saving_folder,macro=bool(parser.parse_args().f1_mean))
test_results = tester.test(cm=bool(parser.parse_args().confusion_matrix))
print(test_results)
# if parser.parse_args().read_history:
history = pickle.load(open(os.path.join(args.saving_folder,'history.txt'),'rb'))[0]
print('HIGHEST TEST ON VALIDATION BEST')
print(('TEST F1 : %.2f')%(history['test_f1'][np.argmax(history['val_f1'])]))
print(np.argmax(history['test_f1']))


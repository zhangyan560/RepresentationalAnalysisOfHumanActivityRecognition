from models import train, cnn, test, vgg, FCN, ResNet
import os
import torch
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import utils
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
dataset = "pamap2"
parser.add_argument('-d', '--dataset', type=str, default=dataset)
parser.add_argument('--downsample_factor',type=int,default=1)
parser.add_argument('--dilated', action='store_true', default=False)
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--window_step', type=int, default=45)
parser.add_argument('--svcca_epochs', type = str, default = ','.join([str(i) for i in [10, 11, 20,21, 150,151,250,251,300,301,350,351,397,398]]))
parser.add_argument('--saving_folder', type=str, default='/content/drive/MyDrive/HistoryOf' + dataset)
parser.add_argument('--R_folder', type=str,default='/content/drive/MyDrive/R/')
parser.add_argument('--rep_saving_folder', type=str, default='/content/drive/MyDrive/Rep1_' + dataset)
parser.add_argument('--batch_size', type=int,default=64)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--nesterov', action='store_true',default=False)
parser.add_argument('--weight_decay', type=float,default=10.e-2)
parser.add_argument('--lr_schedule', type=str, default='3,50, 100')
parser.add_argument('--nb_epochs' , type=int,default=200)
parser.add_argument('--verbose', action='store_true',default=True)
parser.add_argument('--kernel_size',type=int,default=15)
parser.add_argument('--architecture_depth', type=str, default='3,4,6,3')
parser.add_argument('--save_best',action='store_true',default=True)
parser.add_argument('--rep_size', type = int, default = 5)
parser.add_argument('--freeze_rate', type = int, default = 0.99)
parser.add_argument('--freeze', type = bool, default = True)
parser.add_argument('--begin_freeze', type = bool, default = 5)
parser.add_argument('--check_epoch', type = str, default = '7,50,100')
args = parser.parse_args()
print(args.freeze)
if len(args.saving_folder) == 0:
    args.saving_folder = None

if args.saving_folder != None:
  with open(os.path.join(args.saving_folder,'run_info.json'), 'w') as write_file:
      json.dump(vars(args), write_file)

print('LOADING DATA')
datasets = utils.get_datasets(args.dataset,validation=True,window_size=args.window_size,
                              step=args.window_step,downsample=args.downsample_factor)

training_weights = datasets['training_weights']
training_sampler = WeightedRandomSampler(training_weights, len(training_weights))

training_loader = DataLoader(dataset=datasets['training_set'],
                             batch_size=args.batch_size,
                            #  sampler = training_sampler,
                             num_workers=2,
                             drop_last=True,
                             shuffle=True,
                             pin_memory=True)
testing_loader = DataLoader(dataset=datasets['testing_set'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                            pin_memory=True)
validation_loader = DataLoader(dataset=datasets['validation_set'],
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=2,
                               drop_last=True,
                               pin_memory=True)

if args.saving_folder != None and not os.path.isdir(args.saving_folder):
    os.mkdir(args.saving_folder)

# repp_data = []
# for i in range(args.rep_size):
#     repp_data.append(next(iter(training_loader)))
X = datasets['training_set'].data['inputs']
y = datasets['training_set'].data['targets']


rep = []
rep_data = []
for j in range(training_loader.dataset.nb_classes()):
    for i in range(2):
        rep_data.append([torch.Tensor(X[np.where(y==j)[0][i * 64:(i+1) * 64]]), torch.Tensor(y[np.where(y==j)[0][i * 64:(i+1) * 64]]).reshape(64,1,1)])
    rep.append(rep_data)
random_training_sample = next(iter(training_loader))[0]
print('Input Shape:')
print(random_training_sample.shape)
#
# model = cnn.simpleCNN(inplanes=random_training_sample.shape[1],
#                              kernel_size=args.kernel_size,
#                              planes=[64, 64, 64, 64, 128, 128, 128, 128],
#                              dilation=1,output_size=training_loader.dataset.nb_classes())
# model.build_classifier(random_training_sample)

# model = vgg.vgg19(input_channels=random_training_sample.shape[1])
# model.build_classifier(random_training_sample, num_classes=training_loader.dataset.nb_classes())
# model = ResNet.resnet34(num_classes=training_loader.dataset.nb_classes(),
#                         input_channels=random_training_sample.shape[1],
#                         zero_init_residual = True)
model = ResNet.resnetHAR(layers = [int(item) for item in args.architecture_depth.split(',')],
                kernel_size=args.kernel_size,
                input_channel=random_training_sample.shape[1],
                nb_classes=training_loader.dataset.nb_classes())
model.build_classifier(random_training_sample)
# model = resnets.HAR_ResNet1D(input_channels=random_training_sample.shape[1],
#                              kernel_size=args.kernel_size,
#                              depth=[int(item) for item in args.architecture_depth.split(',')],
#                              dilated=args.dilated,nb_classes=training_loader.dataset.nb_classes())
# model.build_classifier(random_training_sample)
# model = FCN.FCN(input_channels=random_training_sample.shape[1], nb_classes=training_loader.dataset.nb_classes())
# model.build_classifier(random_training_sample)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,nesterov=bool(args.nesterov),weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

trainer = train.Trainer(model,training_loader,optimizer,criterion,
                        validation_loader=validation_loader,
                        test_loader=testing_loader,
                        verbose=args.verbose,
                        saving_folder=args.saving_folder,
                        nb_outputs=1,
                        nb_classes = training_loader.dataset.nb_classes(),
                        save_best=bool(args.save_best),
                        f1_macro=datasets['mean_metric'],
                        rep_data = rep,
                        rep_saving_folder = args.rep_saving_folder,
                        dataset=args.dataset)
trainer.train(args.nb_epochs,
                drop_learning_rate=[int(item) for item in args.lr_schedule.split(',')],
                save_cca=[int(item) for item in args.svcca_epochs.split(',')],
                freeze_rate = args.freeze_rate, freeze = args.freeze, begin_freeze = args.begin_freeze, check_epoch = [int(i) for i in args.check_epoch.split(',')])



import torch
import torchvision
from torch import nn
from torch.nn.init import kaiming_normal_
import numpy as np
import os
import tqdm
from models.cca import get_cca_similarity
from models.svcca import svcca_distance

def cca(features_x, features_y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

    Returns:
    The mean squared CCA correlations between X and Y.
    """
    qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
        features_x.shape[1], features_y.shape[1])

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def save_representation_resnet(model, epoch, input, device):
    print("Extracting activations from the model at Epoch: ", epoch)
    
    
    for i, (x, y) in enumerate(input):
        x = x.to(device)
        for layer_name, layer in model.named_children():
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children(): 
                    for b_name, b in block.named_children():
                        if isinstance(b, nn.Conv1d):   
                            b.register_forward_hook(get_activation(layer_name + '_' + block_name+'_'+b_name))
            elif isinstance(layer, nn.Conv1d):
                layer.register_forward_hook(get_activation(layer_name))
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        model.fc.register_forward_hook(get_activation('fc'))
        output = model(x)
        if i == 0:
            act = {idx:[] for idx in activation.keys()}
        for idx in activation.keys():
            act[idx] += [activation[idx]]
    for idx in act.keys():
        act[idx] = torch.stack(act[idx])
        if idx != 'fc' :
            act[idx] = act[idx].view(-1, act[idx].size(2), act[idx].size(3))
        else:
            act[idx] = act[idx].view(-1, act[idx].size(2))

    print("Saving all representations at Epoch: ", epoch)
    return act



def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = torch.svd(tensor)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      torch.ones(1).to(ratio.device),
                      torch.zeros(1).to(ratio.device)
                      ).sum()
    return tensor @ right[:, :int(num)]


def get_similarity(epoch1, epoch2, layer,dataset):
    layer1 = torch.load('/content/drive/MyDrive/Rep_'+dataset + '/Epoch'+ str(epoch1) + layer +'.pt')
    layer2 = torch.load('/content/drive/MyDrive/Rep_'+dataset + '/Epoch'+ str(epoch2) + layer +'.pt')
    layer1 = svd_reduction(layer1)
    layer2 = svd_reduction(layer2)
    acts1 = layer1.cpu().detach().numpy()
    acts2 = layer2.cpu().detach().numpy()
    if l1 == 'fc' and l2 == 'fc':
        f_results = get_cca_similarity(acts1.T, acts2.T, epsilon=1e-10, verbose=False, compute_dirns=True)
        return sum(f_results['cca_coef1'])/len(f_results['cca_coef1'])
    elif l2 == 'fc':
        acts1 = np.mean(acts1, axis=2)
        return cca(acts1, acts2)
    elif l1 == 'fc':
        acts2 = np.mean(acts2, axis=2)
        return cca(acts1, acts2)  
    elif acts1.shape[-1] != acts2.shape[-1]:
        num_datapoints, h, channels = acts1.shape
        f_acts1 = acts1.reshape((channels*h, num_datapoints))

        num_datapoints, h, channels = acts2.shape
        f_acts2 = acts2.reshape((channels*h, num_datapoints))

        return cca(f_acts1.T, f_acts2.T)
    else:
        num_datapoints, channels, h = acts1.shape
        f_acts1 = acts1.reshape((num_datapoints*h, channels))

        num_datapoints, channels,h = acts2.shape
        f_acts2 = acts2.reshape((num_datapoints*h, channels))

        f_results = get_cca_similarity(f_acts1.T, f_acts2.T, epsilon=1e-10, verbose=False, compute_dirns=True)
        return sum(f_results['cca_coef1'])/len(f_results['cca_coef1'])



def freezeTraining(epoch1, epoch2, model, dataset, freeze_rate):
    freezed_layers = []

    if model.conv1.weight.requires_grad:
        for param in model.conv1.parameters():
            param.requires_grad = False
        print('Freeze the first layer conv1')
        freezed_layers.append('conv1')
        return freezed_layers

    

    for layer_name, layer in model.named_children():
        if isinstance(layer, nn.Sequential):
            for block_name, block in layer.named_children():
                if block.conv1.weight.requires_grad:
                    for param in block.parameters():
                        param.requires_grad = False
                    print('Freeze layer' + layer_name + '_' + block_name)
                    freezed_layers.append(layer_name + '_' + block_name)
                    return freezed_layers
    
    
    
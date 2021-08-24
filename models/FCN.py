import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

class basicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, stride=1, padding = 1):
        super(basicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FCN(nn.Module):
    def __init__(self, input_channels=None, kernel_size = [8,5,3], nb_classes=None):
        super(FCN, self).__init__()
        self.layer1 = self._make_layer(basicBlock, input_channels, 128, kernel_size=kernel_size[0])
        self.layer2 = self._make_layer(basicBlock, 128, 256, kernel_size=kernel_size[1])
        self.layer3 = self._make_layer(basicBlock, 256, 128, kernel_size=kernel_size[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.nb_classes = nb_classes
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def _make_layer(self, block, inplanes, planes, kernel_size):
        layers = []
        layers.append(block(inplanes, planes, kernel_size))
        return nn.Sequential(*layers)

    def build_classifier(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.fc = nn.Linear(x.size(1), self.nb_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return [x]

if __name__ == "__main__":
    sample = torch.randn(64, 9, 64)
    model = FCN(input_channels=9, nb_classes=2)
    model.build_classifier(sample)
    print(model(sample))
    print(model)

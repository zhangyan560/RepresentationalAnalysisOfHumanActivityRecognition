import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, downsample = None, stride=1, dilation=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2,
                               stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class resnetHAR(nn.Module):
    def __init__(self, block=BasicBlock, kernel_size = 3, layers = None, input_channel=None, nb_classes=None):
        super(resnetHAR, self).__init__()
        self.inplanes = 64
        self.nb_classes = nb_classes
        self.dilation = 1
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride= 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride= 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)

    def build_classifier(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.fc = nn.Linear(x.size(1), self.nb_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv1d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride),
                                       nn.BatchNorm1d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,kernel_size=self.kernel_size))

        return nn.Sequential(*layers)





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return [x]
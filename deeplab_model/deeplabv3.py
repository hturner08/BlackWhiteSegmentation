import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplab_model.resnet import resnet50, resnet34
from deeplab_model.aspp import ASPP, ASPP_Bottleneck

#from resnet import resnet50, resnet34
#from aspp import ASPP, ASPP_Bottleneck

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)
class Bottleneck_custom(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck_custom, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out
    
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

#from model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
#from model.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes): #, model_id, project_dir):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes #as seen from the cityscapes website

        #self.model_id = model_id
        #self.project_dir = project_dir
        #self.create_model_dirs()
        '''
        self.resnet = ResNet34_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        '''
        #self.resnet = ResNet50_OS16() # NOTE! specify the type of ResNet here
        
        resnet = resnet34() #resnet50()
        
        #resnet.load_state_dict(torch.load("/home/kaustavb/6867/model/resnet50-19c8e357.pth")) #needed ResNet50
        resnet.load_state_dict(torch.load("./deeplab_model/resnet34-333f7ec4.pth")) #needed ResNet34
        
        #self.resnet = nn.Sequential(*list(resnet.children())[:-3]) #only the convolutional features are extracted.
        self.encoder = resnet
        
        # replace last conv layer with dilated convolution
        
        #self.layer5 = make_layer(Bottleneck_custom, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2) #needed ResNet50
        self.layer5 = make_layer(Bottleneck_custom, in_channels=256, channels=64, num_blocks=3, stride=1, dilation=2) #needed ResNet34
        
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        
        #self.conv_1x1_1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        #self.bn_conv_1x1_1 = nn.BatchNorm2d(48)
        
        #-------------->
        #ResNet34 : change value from 256 to 64
        #ResNet50 : change value from 64 to 256
        self.aggregate = nn.Sequential(nn.Conv2d(64, 48, kernel_size=1, bias=False),
                                               nn.BatchNorm2d(48))

        #self.conv_3x3_1 = nn.Conv2d(48+256, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        
        #-------------->
        #ResNet34 : change value from 256 to 64
        #ResNet50 : change value from 64 to 256
        self.last_conv = nn.Sequential(nn.Conv2d(48+64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        #self.bn_conv_3x3_1 = nn.BatchNorm2d(num_classes)
        
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map, low_level_features = self.encoder(x) # (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
        
        #feature_map = self.resnet(x)
        #print(low_level_features.shape)
        #print(feature_map.shape)
        feature_map = self.layer5(feature_map)
#         print(feature_map.shape)
        output = self.aspp(feature_map) # (shape: (batch_size, 256, h/4, w/4))
        #print(output.shape)
        
        #print(low_level_features.shape) # (shape: (batch_size, 256, h/4, w/4))
        #print(feature_map.shape) # (shape: (batch_size, 2048, h/16, w/16))
        #print(output.shape) # (shape: (batch_size, 256, h/4, w/4))
        
        #low_level_features = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(low_level_features))) # (shape: (batch_size, 48, h/4, w/4))
        low_level_features = F.relu(self.aggregate(low_level_features)) # (shape: (batch_size, 48, h/4, w/4))
#         print("Features",low_level_features.shape)
#         print("Output",output.shape)
        output = torch.cat([low_level_features, output], 1)
        #print(output.shape)
        output = self.last_conv(output) # (shape: (batch_size, 256, h/4, w/4))
        
        #print(output.shape)
             
        output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        output = F.softmax(output, dim=1) # performs soft-max and outputs probability values for each pixel.
        
        #print(output.shape)
        return output

Dowload the pretrained ResNet34 and ResNet50 models from :

'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

and put the .pth files in the directory 'deeplab_model'

The deeplab v3+ network can be imported into the training code by using the following lines in your code :

from deeplab_model.deeplabv3 import DeepLabV3
net = DeepLabV3(num_classes=7) #based on number of classes you can change this parameter.

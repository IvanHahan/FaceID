import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

# import transforms as transforms
from skimage import io
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class FER:

    def __init__(self):
        self.net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('model', 'fer.t7'), map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
        self.net.eval()

    def predict(self, images):
        inputs = []
        for im in images:
            gray = rgb2gray(im)
            gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]

            img = np.concatenate((img, img, img), axis=2)
            img = ToTensor()(img).unsqueeze(0)
            # img = Variable(img, volatile=True)
            inputs.append(img)
        inputs = torch.cat(inputs, dim=0)
        outputs = self.net(inputs)
        score = F.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs.data, 1).numpy()
        return predicted

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
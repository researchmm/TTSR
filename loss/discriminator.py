import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


class Discriminator(nn.Module):
    def __init__(self, in_size=160):
        super(Discriminator, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.LReLU1 = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(32, 32, 2)
        self.LReLU2 = nn.LeakyReLU(0.2)
        self.conv3 = conv3x3(32, 64)
        self.LReLU3 = nn.LeakyReLU(0.2)
        self.conv4 = conv3x3(64, 64, 2)
        self.LReLU4 = nn.LeakyReLU(0.2)
        self.conv5 = conv3x3(64, 128)
        self.LReLU5 = nn.LeakyReLU(0.2)
        self.conv6 = conv3x3(128, 128, 2)
        self.LReLU6 = nn.LeakyReLU(0.2)
        self.conv7 = conv3x3(128, 256)
        self.LReLU7 = nn.LeakyReLU(0.2)
        self.conv8 = conv3x3(256, 256, 2)
        self.LReLU8 = nn.LeakyReLU(0.2)
        self.conv9 = conv3x3(256, 512)
        self.LReLU9 = nn.LeakyReLU(0.2)
        self.conv10 = conv3x3(512, 512, 2)
        self.LReLU10 = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(in_size//32 * in_size//32 * 512, 1024)
        self.LReLU11 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.LReLU1(self.conv1(x))
        x = self.LReLU2(self.conv2(x))
        x = self.LReLU3(self.conv3(x))
        x = self.LReLU4(self.conv4(x))
        x = self.LReLU5(self.conv5(x))
        x = self.LReLU6(self.conv6(x))
        x = self.LReLU7(self.conv7(x))
        x = self.LReLU8(self.conv8(x))
        x = self.LReLU9(self.conv9(x))
        x = self.LReLU10(self.conv10(x))
        
        x = x.view(x.size(0), -1)
        x = self.LReLU11(self.fc1(x))
        x = self.fc2(x)
        
        return x


if __name__ == '__main__':
    model = discriminator()
    x = torch.ones(1, 3, 160, 160)
    out = model(x)
    print (out.size())
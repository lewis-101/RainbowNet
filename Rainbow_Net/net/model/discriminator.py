import torch.nn as nn
import torch

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=3, output_dim=1):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.input_size = input_size
        #self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        #utils.initialize_weights(self)
        # self.classifier = nn.Conv2d(in_channels=cnum * 8, out_channels=1, kernel_size=5, stride=1, padding=1)


    # def forward(self, input, label):
    #     x = torch.cat([input, label], 1)
    #     x = self.conv(x)
    #     return x

    def forward(self, input):
        x = self.conv(input)
        return x
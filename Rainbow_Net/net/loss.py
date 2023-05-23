import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''
  most codes are borrowed from: 
  https://github.com/knazeri/edge-connect/blob/master/src/loss.py
'''
class VGG19(torch.nn.Module):
  def __init__(self, resize_input=False):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features.cuda()

    self.resize_input = resize_input
    self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    prefix = [1,1, 2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5]
    posfix = [1,2, 1,2, 1,2,3,4, 1,2,3,4, 1,2,3,4]
    names = list(zip(prefix, posfix))
    self.relus = []
    for pre, pos in names:
      self.relus.append('relu{}_{}'.format(pre, pos))
      self.__setattr__('relu{}_{}'.format(pre, pos), torch.nn.Sequential())

    nums = [[0,1], [2,3], [4,5,6], [7,8],
     [9,10,11], [12,13], [14,15], [16,17],
     [18,19,20], [21,22], [23,24], [25,26],
     [27,28,29], [30,31], [32,33], [34,35]]

    for i, layer in enumerate(self.relus):
      for num in nums[i]:
        self.__getattr__(layer).add_module(str(num), features[num])

    # don't need the gradients, just want the features 预训练的VGG19模型进行特征提取时，不会更新模型的权重参数
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    # resize and normalize input for pretrained vgg19
    x = (x+1)/2
    x = (x-self.mean.view(1,3,1,1)) / (self.std.view(1,3,1,1))
    if self.resize_input:
      x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
    features = []
    for layer in self.relus:
      x = self.__getattr__(layer)(x)
      features.append(x)
    out = {key: value for (key, value) in list(zip(self.relus, features))}
    return out




class AdversarialLoss(nn.Module):
  r"""
  Adversarial loss
  https://arxiv.org/abs/1711.10337
  """

  def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
    r"""
    type = nsgan | lsgan | hinge
    """
    super(AdversarialLoss, self).__init__()
    self.type = type
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))

    if type == 'nsgan':
      self.criterion = nn.BCELoss()
    elif type == 'lsgan':
      self.criterion = nn.MSELoss()
    elif type == 'hinge':
      self.criterion = nn.ReLU()

  def patchgan(self, outputs, is_real=None, is_disc=None):
    if self.type == 'hinge': #ReLU
      if is_disc: # True 来自判别器
        if is_real:  # True表示输出来自真实图像的判别结果
          outputs = -outputs # 真实图像 取反
        return self.criterion(1 + outputs).mean() # 1 代表真实标签 True[1+(-真实)] False[1 + 合成]
      else: # False 来自生成器
        return (-outputs).mean() #[-合成]
    else:
      labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
      loss = self.criterion(outputs, labels)
      return loss

  def __call__(self, outputs, is_real=None, is_disc=None):
    return self.patchgan(outputs, is_real, is_disc)

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch) #矩阵相乘 在归一化

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

import numpy as np
import glob
import torch, time, os, cv2
import torch.nn as nn
import torch.nn.functional as F
from Rainbow_Net.net.model.transformer import FeatureTransformer, FeatureFlowAttention
from dataloader import dataloader
from model.vgg import Vgg16
from model.generator_net import generator_net
from model.generator_trans import generator_trans
from model.discriminator import discriminator
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils.utils import Progbar,set_device
from loss import AdversarialLoss, PerceptualLoss, StyleLoss


class RainNet(nn.Module):
    def __init__(self,
                 config,
                 args,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(RainNet, self).__init__()

        self.config = config
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        self.iteration = 0
        self.epoch = 0

        self.dataset = args.dataset
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        # 日志
        self.D_writer = None
        self.G_writer = None
        self.summary = {}
        self.D_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))  # 对抗模型的日志
        self.G_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))  # 生成模型的日志

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)

        # networks init !!!!
        # self.G = generator_net(3, 3) # cnn!!!
        self.G = generator_trans(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            **kwargs) # trans
        self.D = discriminator(input_dim=3, output_dim=1) # 6-> 3
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config['Rainbow']['lrG'], betas=(config['Rainbow']['beta1'], config['Rainbow']['beta2']))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config['Rainbow']['lrG'], betas=(config['Rainbow']['beta1'], config['Rainbow']['beta2']))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.l1loss = nn.L1Loss().cuda()
            self.loss_mse = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        # Gan 的初始化
        def weight_init(m):
          classname = m.__class__.__name__
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

        # self.G.apply(weight_init)
        self.D.apply(weight_init)

        # swim 的初始化
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.G.apply(_init_weights)


        # set up losses and metrics
        self.adversarial_loss = set_device(AdversarialLoss(type=self.config['losses']['gan_type']))  # 对抗损失 增强影像的纹理
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()  # 感知损失
        self.style_loss = StyleLoss()  # 风格损失

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # 加载
        self.load()
    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):  # 加载模型参数文件
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None  # 取最大的.pth文件
        if latest_epoch is not None:  # 参数加载进当前模型和优化器中
            G_path = os.path.join(model_path, 'G_{}.pth'.format(str(latest_epoch).zfill(5)))
            D_path = os.path.join(model_path, 'D_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(G_path))
            data = torch.load(G_path, map_location=lambda storage, loc: set_device(storage))
            self.G.load_state_dict(data['G'])
            data = torch.load(D_path, map_location=lambda storage, loc: set_device(storage))
            self.D.load_state_dict(data['D'])
            data = torch.load(opt_path, map_location=lambda storage, loc: set_device(storage))
            self.G_optimizer.load_state_dict(data['optimG'])
            self.D_optimizer.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            G_path = os.path.join(self.config['save_dir'], 'G_{}.pth'.format(str(it).zfill(5)))
            D_path = os.path.join(self.config['save_dir'], 'D_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(G_path))
            # if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
            #     netG, netD = self.netG.module, self.netD.module
            # else:
            #     netG, netD = self.netG, self.netD
            G, D = self.G, self.D
            torch.save({'G': G.state_dict()}, G_path)
            torch.save({'D': D.state_dict()}, D_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.G_optimizer.state_dict(),
                        'optimD': self.D_optimizer.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # get current learning rate
    def get_lr(self, type='G'):
        if type == 'G':
            return self.G_optimizer.param_groups[0]['lr']  # 表示获取第一个参数组中的学习率
        return self.D_optimizer.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):  # 动态调整优化器的学习率
        # 衰减系数
        decay = 0.1 ** (
                    min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay  # 新的学习率
        if new_lr != self.get_lr():
            for param_group in self.G_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.D_optimizer.param_groups:
                param_group['lr'] = new_lr

    def add_summary(self, writer, name, val):#TensorBoard 摘要
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0


    # 可以不用
    vgg = Vgg16().type(torch.cuda.FloatTensor)

    # process input and calculate loss every training epoch
    def _train_epoch(self):
        self.epoch += 1
        # vgg = Vgg16().type(torch.cuda.FloatTensor)
        # self.iteration += 1
        self.adjust_learning_rate()
        self.D.train() # 设置模型为训练状态
        # print('training start!!')
        start_time = time.time()
        lenth = self.data_loader.dataset.__len__()

        progbar = Progbar(lenth, width=20, stateful_metrics=['epoch', 'iter'])  # 进度条 len(self.train_dataset)
        mae = 0

        for iter, (x_, y_, mask) in enumerate(self.data_loader):
            self.iteration += 1
            print("iter= %d" % iter)
            self.G.train()  # 设置模型为训练状态

            if self.gpu_mode:
                x_, y_ = x_.cuda(), y_.cuda()

            G_loss = 0
            D_loss = 0

            # x_ 是合成的 y_ 是gt
            # update D network
            # 先计算D loss D input_dim = 3
            pyramid_feat, G_ = self.G(x_) # 去条纹 x_进入swim 再上采样对G_  后来想想,我好像不需要这个特征,那我可以拿来计算loss
            D_real_feat = self.D(y_)
            D_fake_feat = self.D(G_.detach()) # 上采样之后再这个
            D_real_loss = self.adversarial_loss(D_real_feat, True, True)  # 真实图像特征 需要反向传播
            D_fake_loss = self.adversarial_loss(D_fake_feat, False, True)  # 合成图像特征 需要反向传播
            D_loss += (D_real_loss + D_fake_loss) / 2

            self.add_summary(self.D_writer, 'loss/D_loss', D_loss.item())

            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            # update G network
            # 先计算G loss

            # generator adversarial loss
            G_fake_feat = self.D(G_)
            G_fake_loss = self.adversarial_loss(G_fake_feat, True, False) # is_real=True, is_disc=False
            G_loss += G_fake_loss * self.config['losses']['adversarial_weight']
            self.add_summary(self.G_writer, 'loss/G_fake_loss', G_fake_loss.item())

            # generator l1 loss
            L1_loss = self.l1_loss(G_, y_)  # 生成图像和真实图像之间的L1损失，即像素级别的绝对误差。 惩罚生成图像中的细节和噪声
            G_loss += L1_loss * self.config['losses']['l1_weight']
            self.add_summary(self.G_writer, 'loss/L1_loss', L1_loss.item())

            # perceptual loss / content loss
            G_content_loss = self.perceptual_loss(G_, y_)  # 在高层次特征表示中的差异。 相似性
            G_loss += G_content_loss * self.config['losses']['content_loss_weight']
            self.add_summary(self.G_writer, 'loss/content_loss', G_content_loss.item())

            # style loss
            G_style_loss = self.style_loss(G_, y_)  # 通过计算生成图像和原始图像在风格提取网络中的特征的差异来计算损失值。
            G_loss += G_style_loss * self.config['losses']['style_loss_weight']
            self.add_summary(self.G_writer, 'loss/style_loss', G_style_loss.item())


            self.add_summary(self.G_writer, 'loss/G_loss',G_loss.item())

            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            new_mae = (torch.mean(torch.abs(y_ - G_))).item()  # 模型预测结果与真实结果之间的平均绝对误差
            mae = new_mae

            logs = [("epoch", self.epoch), ("iter", self.iteration), ("lr", self.get_lr()),
                    ('mae', mae), ('gen_loss', G_fake_loss.item()), ('L1_loss', L1_loss.item()),
                    # ('pyramid_loss', pyramid_loss.item()),
                    ('content_loss', G_content_loss.item()),
                    ('style_loss', G_style_loss.item())]

            # if self.config['global_rank'] == 0:
            progbar.add(len(x_) * self.config['world_size'], values=logs \
                if self.config['trainer']['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

            # saving and evaluating
            if self.iteration % self.config['trainer']['save_freq'] == 0:
                self.save(int(self.iteration // self.config['trainer']['save_freq']))
            if self.iteration > self.config['trainer']['iterations']:
                break





















    def train(self):
        start_time = time.time()
        print('training start!!')
        while True:
            # self.epoch += 1
            # if self.config['distributed']:
            #     self.train_sampler.set_epoch(self.epoch)
            self._train_epoch()
            if self.iteration > self.config['trainer']['iterations']:
                break
        print('\nEnd training....')




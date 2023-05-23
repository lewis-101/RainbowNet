# -*- coding: utf-8 -*-
import cv2
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import math
import os
import argparse
import importlib
import datetime
import json

### My libs
from utils.utils import set_device, postprocess
# from core.utils import postprocess
# from core.dataset import Dataset

from model.generator_trans import generator_trans
from dataloader import dataloader_test

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", default='configs/places2.json', type=str, required=False)
parser.add_argument("-l", "--level",  type=int, required=False)
parser.add_argument("-n", "--model_name", default='pennet4', type=str, required=False)
parser.add_argument("-m", "--mask", default='square', type=str)
parser.add_argument("-s", "--size", default=256, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 4

def main_worker(gpu, ngpus_per_node, config, **kwargs):
  torch.cuda.set_device(gpu)
  # set_seed(config['seed'])

  # Model and version
  # net = importlib.import_module('model.'+args.model_name)
  # model = set_device(net.InpaintGenerator())
  G = generator_trans(in_chans=3,
                           patch_size=4,
                           window_size=7,
                           embed_dim=96,
                           depths=(2, 2, 6, 2),
                           num_heads=(3, 6, 12, 24),
                           **kwargs)
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'G_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage))
  G.load_state_dict(data['G'])
  G.eval()

  # prepare dataset
  # dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  # step = math.ceil(len(dataset) / ngpus_per_node)
  # dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  # dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)


  batch_size = 1
  data_loader = dataloader_test(batch_size)


  # path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  path = config['save_dir']
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  out_path = 'G:/Image_Decomposition/RainbowNet-main/Rainbow_Net/net/release_model/result'
  # for idx, (images, gt, img_name, gt_name) in enumerate(dataloader):
  for iter, (images, gt, mask, img_id) in enumerate(data_loader):
    print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      gpu, iter, len(data_loader), img_id))
    images, gt = set_device([images, gt])
    with torch.no_grad():
      _, output = G(images)
    orig_imgs = postprocess(images)
    gt_imgs = postprocess(gt)
    pred_imgs = postprocess(output)
    # numpy().astype(np.uint8) * 255
    for i in range(len(orig_imgs)):
      # Image.fromarray(pred_imgs[i]).save(os.path.join(path, '{}_pred.png'.format(img_name[i].split('.')[0])))
      # Image.fromarray(pred_imgs[i]).save(os.path.join(path, '{}_pred.png'.format(img_id)))
      cv2.imwrite(os.path.join(out_path, '{}_pred.png'.format(img_id[0])), pred_imgs[i])
      # Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(img_name[i].split('.')[0])))
      # Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(img_id)))
      cv2.imwrite(os.path.join(out_path, '{}_orig.png'.format(img_id[0])), orig_imgs[i])
      # Image.fromarray(gt_imgs[i]).save(os.path.join(path, '{}_gt.png'.format(img_name[i].split('.')[0])))
      # Image.fromarray(gt_imgs[i]).save(os.path.join(path, '{}_gt.png'.format(img_id)))
      cv2.imwrite(os.path.join(out_path, '{}_gt.png'.format(img_id[0])), gt_imgs[i])

  print('Finish in {}'.format(out_path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  config['model_name'] = args.model_name
  # 可以换sava_dir 这个路径指向自己练的 不用这个路径指向他给的预训练模型
  # config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'],
  #   config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))


  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
  gpu = 0
  main_worker(gpu, ngpus_per_node, config)

 

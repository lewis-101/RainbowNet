import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os.path as osp
import os, cv2


def dataloader(dataset, input_size, batch_size):
	data_loader = DataLoader(
            Getdata(),
            batch_size = batch_size, shuffle=True, num_workers=0)
	return data_loader

def dataloader_test(batch_size):
	data_loader = DataLoader(
            Getdata_test(),
            batch_size = batch_size, shuffle=True, num_workers=0)
	return data_loader


class Getdata(torch.utils.data.Dataset):
	def __init__(self):
		self.transform_norm = transforms.Compose([transforms.ToTensor()])
		self.transform_tensor = transforms.ToTensor()
		root = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/test/'
		self.imageR_path = osp.join(root, 'image', '%s.png')
		self.imageF_path = osp.join(root, 'image_free', '%s.png')
		self.mask_path = osp.join(root, 'mask_out', '%s.png')

		self.root = root
		self.transform = transforms
		self.ids = list()
		# 原
		# for file in os.listdir(root+'/Watermarked_image'):
		# 	#if(file[:-4]=='.jpg'):
		# 	self.ids.append(file.strip('.jpg'))
		dir_path = root + '/image'
		files = os.listdir(dir_path)
		sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
		self.ids = [f.split('.')[0] for f in sorted_files if f.endswith('.png')]

		# print(self.ids)

	def __getitem__(self,index):
		imag_R, image_F, mask = self.pull_item(index)
		return imag_R, image_F, mask
	def __len__(self):
		return len(self.ids)
	def pull_item(self,index):
		img_id = self.ids[index]
		# PIL RGB
		# img_R = Image.open(self.imageR_path % img_id)
		# img_F = Image.open(self.imageF_path % img_id)
		# mask = Image.open(self.mask_path % img_id)
		# cv2 打开图像 BGR
		img_R = cv2.imread(self.imageR_path % img_id)
		img_F = cv2.imread(self.imageF_path % img_id)
		mask = cv2.imread(self.mask_path % img_id, 0) # 灰度图

		# 对合成的要用mask进行矩形框，再输入到生成网络中
		img_R_crop, mask = self.rectangle(img_R, mask)
		img_F_crop, mask = self.rectangle(img_F, mask)

		img_source = self.transform_norm(img_R_crop)
		image_target = self.transform_norm(img_F_crop)
		# mask = self.transform_norm(mask)
		return img_source, image_target, mask

	# 找目标区域
	def rectangle(self, image, mask):
		# 将彩色图像转换为灰度图像
		# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		# 查找非零值的位置
		# B, C, H, W = mask.shape
		# mask = mask.view(-1, W)
		nz = np.nonzero(mask)
		xmin = np.min(nz[1])
		xmax = np.max(nz[1])
		ymin = np.min(nz[0])
		ymax = np.max(nz[0])
		# 使用boundingRect函数计算mask的最小边界框 x是竖的方向 y是横的方向
		x, y, w, h = cv2.boundingRect(np.column_stack(nz))

		# 裁剪图像，获取矩形框内的像素
		image_crop = image[x:x + w, y:y + h]
		# PIL
		# image_crop = image.crop((x, y, x+w, y+h))

		# 显示结果图像
		# cv2.imshow('image', image)
		# cv2.imshow('image_crop', image_crop)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return image_crop, mask


class Getdata_test(torch.utils.data.Dataset):
	def __init__(self):
		self.transform_norm = transforms.Compose([transforms.ToTensor()])
		self.transform_tensor = transforms.ToTensor()
		root = 'G:/Image_Decomposition/Grounded-Segment-Anything-outputs/test/'
		self.imageR_path = osp.join(root, 'image', '%s.png')
		self.imageF_path = osp.join(root, 'image_free', '%s.png')
		self.mask_path = osp.join(root, 'mask_out', '%s.png')

		self.root = root
		self.transform = transforms
		self.ids = list()
		# 原
		# for file in os.listdir(root+'/Watermarked_image'):
		# 	#if(file[:-4]=='.jpg'):
		# 	self.ids.append(file.strip('.jpg'))
		dir_path = root + '/image'
		files = os.listdir(dir_path)
		sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
		self.ids = [f.split('.')[0] for f in sorted_files if f.endswith('.png')]

		# print(self.ids)

	def __getitem__(self,index):
		imag_R, imag_F, mask, img_id = self.pull_item(index)
		return imag_R, imag_F, mask, img_id
	def __len__(self):
		return len(self.ids)
	def pull_item(self,index):
		img_id = self.ids[index]
		# PIL RGB
		# img_R = Image.open(self.imageR_path % img_id)
		# img_F = Image.open(self.imageF_path % img_id)
		# mask = Image.open(self.mask_path % img_id)
		# cv2 打开图像 BGR
		img_R = cv2.imread(self.imageR_path % img_id)
		img_F = cv2.imread(self.imageF_path % img_id)
		mask = cv2.imread(self.mask_path % img_id, 0) # 灰度图

		# 对合成的要用mask进行矩形框，再输入到生成网络中
		img_R_crop, mask = self.rectangle(img_R, mask)
		img_F_crop, mask = self.rectangle(img_F, mask)

		img_source = self.transform_norm(img_R_crop)
		image_target = self.transform_norm(img_F_crop)
		# mask = self.transform_norm(mask)
		return img_source, image_target, mask, img_id

	# 找目标区域
	def rectangle(self, image, mask):
		# 将彩色图像转换为灰度图像
		# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		# 查找非零值的位置
		# B, C, H, W = mask.shape
		# mask = mask.view(-1, W)
		nz = np.nonzero(mask)
		xmin = np.min(nz[1])
		xmax = np.max(nz[1])
		ymin = np.min(nz[0])
		ymax = np.max(nz[0])
		# 使用boundingRect函数计算mask的最小边界框 x是竖的方向 y是横的方向
		x, y, w, h = cv2.boundingRect(np.column_stack(nz))

		# 裁剪图像，获取矩形框内的像素
		image_crop = image[x:x + w, y:y + h]
		# PIL
		# image_crop = image.crop((x, y, x+w, y+h))

		# 显示结果图像
		# cv2.imshow('image', image)
		# cv2.imshow('image_crop', image_crop)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return image_crop, mask
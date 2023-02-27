import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import skimage.io
import torch
import pickle
import cv2
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T


# import pdb

class LowLight_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt['noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        if train:
            self.dark_root = f'{self.root}/input/train/1'
            self.light_root = f'{self.root}/gt/train'
        else:
            self.dark_root = f'{self.root}/input/test/1'
            self.light_root = f'{self.root}/gt/test'

        self.dark_paths = [f for f in os.listdir(self.dark_root) if os.path.isfile(os.path.join(self.dark_root, f))]
        self.light_paths = [f for f in os.listdir(self.light_root) if os.path.isfile(os.path.join(self.light_root, f))]

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.dark_root)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)
        pairs = []
        for idx, f_name in enumerate(low_list):
            pairs.append([
                 cv2.imread(os.path.join(folder_path, 'low', f_name)),  # [:, 4:-4, :],
                 cv2.imread(os.path.join(folder_path, 'high', f_name)),
                                            f_name.split('.')[0]])
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        dark_path = f'{self.dark_root}/{self.dark_paths[item]}'

        lr = skimage.io.imread(dark_path)

        light_path = f'{self.light_root}/{dark_path.split("/")[-1][0:5]}_00_10s.png'
        if not os.path.exists(light_path):
            light_path = f'{self.light_root}/{dark_path.split("/")[-1][0:5]}_00_30s.png'

        hr = skimage.io.imread(light_path)
        f_name = self.dark_paths[item][:-4]
        his = self.hiseq_color_cv2_img(lr)
        # lr, hr, f_name, his = self.pairs[item]
        if self.histeq_as_input:
            lr = his

        if self.use_crop:
            hr, lr, his = random_crop(hr, lr, his, self.crop_size)

        if self.center_crop_hr_size:
            hr = center_crop(hr, self.center_crop_hr_size)
            lr = center_crop(lr, self.center_crop_hr_size)
            his = center_crop(his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, his = random_flip(hr, lr, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()
        if self.concat_histeq:
            his = self.to_tensor(his)
            lr = torch.cat([lr, his], dim=0)

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}



class SonyTif_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt['noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)

        self.dark_root = f'{self.root}/short_tif_crop_test'
        self.light_root = f'{self.root}/long_tif_crop_test'


        if train:
            file_path = f'{self.root}/Sony_train_list.txt'
        else:
            file_path = f'{self.root}/Sony_val_list.txt'

        assert os.path.isfile(file_path)
        with open(file_path) as file:
            lines = []
            for line in file:
                lines.append(line.rstrip())
            # while (line := file.readline().rstrip()):
            #     lines.append(line)

        input_fnames = []
        gt_fnames = []
        for i in range(len(lines)):
            items = lines[i].split(' ')
            input_fnames.append(items[0].split('/')[-1][:-4])
            gt_fnames.append(items[1].split('/')[-1][:-4])

        self.input_files = sorted([f for f in os.listdir(self.dark_root) if os.path.isfile(os.path.join(self.dark_root, f))])
        self.gt_files = sorted([f for f in os.listdir(self.light_root) if os.path.isfile(os.path.join(self.light_root, f))])
        self.dark_paths = [k for k in self.input_files if f'{("_").join(k.split("_")[:-1])}' in input_fnames]
        self.light_paths = [k for k in self.gt_files if f'{("_").join(k.split("_")[:-1])}' in gt_fnames]

        # self.pairs = self.load_pairs(self.dark)
        # self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.dark_root)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)
        pairs = []
        for idx, f_name in enumerate(low_list):
            pairs.append([
                 cv2.imread(os.path.join(folder_path, 'low', f_name)),  # [:, 4:-4, :],
                 cv2.imread(os.path.join(folder_path, 'high', f_name)),
                                            f_name.split('.')[0]])
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def to_tensor(self, x):
        return torch.from_numpy(x.copy()).permute(2, 0, 1).type(torch.float32)

    def __getitem__(self, item):

        lr = skimage.io.imread(f'{self.dark_root}/{self.dark_paths[item]}') #[0,1] H,W,3
        hr = skimage.io.imread(f'{self.light_root}/{self.light_paths[item]}')

        lr = np.flip(lr, axis=-1)
        hr = np.flip(hr, axis=-1)

        f_name = self.dark_paths[item][:-4]
        his = self.hiseq_color_cv2_img((lr * 255).astype(np.uint8)) / 255.0

        if self.histeq_as_input:
            lr = his

        if self.use_crop:
            hr, lr, his = random_crop(hr, lr, his, self.crop_size)

        if self.center_crop_hr_size:
            hr = center_crop(hr, self.center_crop_hr_size),
            lr = center_crop(lr, self.center_crop_hr_size)
            his = center_crop(his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, his = random_flip(hr, lr, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        hr = self.to_tensor(hr.copy())
        lr = self.to_tensor(lr.copy())

        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()
        if self.concat_histeq:
            his = self.to_tensor(his)
            lr = torch.cat([lr, his], dim=0)

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


def random_flip(img, seg, his_eq):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()
    if his_eq is not None:
        his_eq = his_eq if random_choice else np.flip(his_eq, 1).copy()
    return img, seg, his_eq


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg, his):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    if his is not None:
        his = np.rot90(his, random_choice, axes=(0, 1)).copy()
    return img, seg, his


def random_crop(hr, lr, his_eq, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    if his_eq is not None:
        his_eq_patch = his_eq[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]
    return hr_patch, lr_patch, his_eq_patch


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]

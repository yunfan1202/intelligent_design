from torch.utils import data
import torchvision.transforms as transforms
import os
from pathlib import Path
from PIL import Image
import numpy as np


def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.JPG')or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.ppm')]


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


# def fold_files(foldname):
#     """All files in the fold should have the same extern"""
#     allfiles = os.listdir(foldname)
#     if len(allfiles) < 1:
#         raise ValueError('No images in the data folder')
#         return None
#     else:
#         return allfiles

class Custom_Loader_test(data.Dataset):
    """
    Custom Dataloader
    """
    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # self.imgList = fold_files(os.path.join(root))
        self.imgList = get_imgs_list(root)


    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):
        # with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
        with open(self.imgList[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename


class Custom_Loader_train(data.Dataset):
    """
    Custom Dataloader
    """
    def __init__(self, train_image_dir='data/custom', train_label_dir='data/custom', threshold=0.3, use_uncertainty=True):
        self.train_image_dir = train_image_dir
        self.train_label_dir = train_label_dir
        self.threshold = threshold * 256

        self.use_uncertainty = use_uncertainty

        print('Threshold for ground truth: %f on Custom' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # self.imgList = fold_files(os.path.join(root))
        # self.imgList = get_imgs_list(train_image_dir)
        self.imgList = get_imgs_list(train_label_dir)

    def __len__(self):
        return len(self.imgList)

    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)
        return img

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path)
        lb = np.array(lb_data, dtype=np.float32)
        if lb.ndim == 3:
            lb = np.squeeze(lb[:, :, 0])
        assert lb.ndim == 2
        threshold = self.threshold
        lb = lb[np.newaxis, :, :]

        lb[lb == 0] = 0
        # ---------- important ----------
        if self.use_uncertainty:
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
        else:
            lb[np.logical_and(lb > 0, lb < threshold)] /= 255.
        lb[lb >= threshold] = 1
        return lb

    def __getitem__(self, index):
        image_path = fit_img_postfix(self.imgList[index])
        filename = Path(image_path).stem
        lb_path = os.path.join(self.train_label_dir, filename + ".png")

        img = self.read_img(image_path)
        lb = self.read_lb(lb_path)

        return img, lb
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from models.convert_pidinet import convert_pidinet
from utils import *
from edge_dataloader import Custom_Loader_test
from torch.utils.data import DataLoader
import cv2
import torch
import torchvision
from PIL import Image
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

print("torch.cuda.is_available():", torch.cuda.is_available())
parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='results/savedir',
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data',
        help='dir to the dataset')

parser.add_argument('--dataset', type=str, default='BSDS',
        help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='baseline',
        help='model to train the dataset')
parser.add_argument('--sa', action='store_true',
        help='use CSAM in pidinet')
parser.add_argument('--dil', action='store_true',
        help='use CDCM in pidinet')
parser.add_argument('--config', type=str, default='carv4',
        help='model configurations, please refer to models/config.py for possible configurations')

parser.add_argument('--gpu', type=str, default='',
        help='gpus available')
parser.add_argument('--checkinfo', action='store_true',
        help='only check the informations about the model: model size, flops')


parser.add_argument('-j', '--workers', type=int, default=4,
        help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3,
        help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--lmbda', type=float, default=1.1,
        help='weight on negative pixels (the beta parameter in the paper)')

parser.add_argument('--print-freq', type=int, default=10,
        help='print frequency')
parser.add_argument('--save-freq', type=int, default=1,
        help='save frequency')
parser.add_argument('--evaluate', type=str, default=None,
        help='full path to checkpoint to be evaluated')
parser.add_argument('--evaluate-converted', action='store_true',
        help='convert the checkpoint to vanilla cnn, then evaluate')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main(running_file):

    global args

    ### Refine args - yyf
    args.model = 'pidinet'
    args.config = 'carv4'
    args.sa = True
    args.dil = True
    args.gpu = '0'
    args.dataset = ['Custom']
    # ------------------------------改以下部分的路径------------------------------------
    args.datadir = "example_image"      # 模型会检测这个路径下的图片

    # args.evaluate = "BSDS_refine_dice.pth"      # 用这个模型出来的边缘更细
    #args.savedir = "example_results"

    args.evaluate = "BSDS_raw.pth"            # 这是原模型，出来的边缘粗一些
    args.savedir = "example_results_raw"
    # ------------------------------改以上部分的路径------------------------------------

    os.makedirs(args.savedir, exist_ok=True)

    args.evaluate_converted = False
    args.checkinfo = True
    ### Refine args
    args.use_cuda = torch.cuda.is_available()
    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    ### Output its model size, flops and bops
    if args.checkinfo:
        count_paramsM = get_model_parm_nums(model)
        print('Model size: %f MB' % count_paramsM)
        print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))

    ### Define optimizer

    ### Transfer to cuda devices
    if args.use_cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    if 'Custom' == args.dataset[0]:
        test_dataset = Custom_Loader_test(root=args.datadir)
    else:
        raise ValueError("unrecognized dataset setting")

    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    args.start_epoch = 0
    ### Evaluate directly if required

    checkpoint = load_checkpoint(args, running_file)
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] + 1
        if args.evaluate_converted:
            model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))
        else:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('no checkpoint loaded')

    normal_test(test_loader, model, args.start_epoch, running_file, args)
    # multiscale_test(test_loader, model, args.start_epoch, running_file, args)
    print('##########Time########## %s' % (time.strftime('%Y-%m-%d %H:%M:%S')))

    return


def normal_test(test_loader, model, epoch, running_file, args):
    model.eval()
    # img_dir = os.path.join(args.savedir, 'imgs_epoch_%03d' % (epoch - 1))
    img_dir = args.savedir

    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    os.makedirs(img_dir, exist_ok=True)
    for idx, (image, img_name) in enumerate(test_loader):

        img_name = img_name[0]
        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1]).cpu().numpy()

        # results_all = torch.zeros((len(results), 1, H, W))
        # for i in range(len(results)):
        #     results_all[i, 0, :, :] = results[i]

        # torchvision.utils.save_image(1-results_all, os.path.join(img_dir, "%s.jpg" % img_name))
        result = Image.fromarray(((1-result) * 255).astype(np.uint8))
        # result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        if (idx + 1) % 25 == 0:
            print(img_name)
            print(runinfo)

def crisp_test(test_loader, model, epoch, running_file, args):
    from PIL import Image
    import scipy.io as sio
    model.eval()

    # img_dir = os.path.join(args.savedir, 'imgs_epoch_%03d' % (epoch - 1))
    img_dir = args.savedir
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    os.makedirs(img_dir, exist_ok=True)
    for idx, (image, img_name) in enumerate(test_loader):
        img_name = img_name[0]

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        scale = 1.5
        _, H, W = image.shape
        # multi_fuse = np.zeros((H, W), np.float32)

        with torch.no_grad():
            im_ = cv2.resize(image_in, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)

            im_ = image_in.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_raw = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            final = cv2.bitwise_and(fuse_raw, fuse)

        result = Image.fromarray(((1-final) * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))


def multiscale_test(test_loader, model, epoch, running_file, args):

    from PIL import Image
    import scipy.io as sio
    model.eval()

    # img_dir = os.path.join(args.savedir, 'imgs_epoch_%03d' % (epoch - 1))
    img_dir = args.savedir
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))

    os.makedirs(img_dir, exist_ok=True)

    for idx, (image, img_name) in enumerate(test_loader):
        img_name = img_name[0]

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))

        scale = [0.5, 1, 1.5]

        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)

        with torch.no_grad():
            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)

                im_ = im_.transpose((2, 0, 1))
                results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
                result = torch.squeeze(results[-1].detach()).cpu().numpy()
                fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse
            multi_fuse = multi_fuse / len(scale)

        # sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': multi_fuse})
        result = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        # result = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)
        running_file.write('%s\n' % runinfo)
    running_file.write('\nDone\n')

if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' \
            % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))
    with open(running_file, 'w') as f:
        main(f)
    print('done')

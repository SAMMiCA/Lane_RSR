import argparse
import os
import time

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.nn import DataParallel

from models.model import define_G
from models.model_ENET_SAD import ENet_SAD

from data_loader.CULane2 import CULane
from utils import getLane
from utils.transforms import *

import time

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/jisukang/LaneDetection/CULane', help='dataset path')
    parser.add_argument('--type', type=str, default='ae', help='type of the gan loss/ ae, naive, wgangp, hinge')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--old_model', type=str, default='./saved_models/Painting_KD_no_guidance_tanh_threshold_0_lamb_1_exp3/epoch_10.pth')
    parser.add_argument('--lane_model', type=str, default='./saved_models/exp3_best.pth')
    parser.add_argument('--model_path', type=str, default='./saved_models/', help='Location to save to')
    parser.add_argument('--output_folder', type=str, default='./output/')
    args = parser.parse_args()
    return args


# ------------ config ------------
args = parse_args()
exp_name = 'Painting_KD_no_guidance_tanh_threshold_0_lamb_1_exp3'
device = torch.device('cuda')
resize_shape = (800, 288)


class GenWrapper(nn.Module):
    def __init__(self, norm='batch', use_dropout=False):
        super(GenWrapper, self).__init__()
        self.encoder, self.decoder = define_G(input_nc=3, output_nc=3, ngf=64,
                                              netG='resnet_9blocks', norm=norm,
                                              use_dropout=use_dropout)

    def forward(self, image):
        latent, att = self.encoder(image)
        output = self.decoder(latent)
        return output, att


def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


# ------------ data and model ------------
test_set = CULane(args.data_path, 'test', (800, 288))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                         collate_fn=test_set.collate, num_workers=6)

print("\nloading Lane Detection Model ......")
lane_model = ENet_SAD(input_size=[800, 288], sad=True).to(device)
checkpoint_file = torch.load(args.lane_model)
lane_model.load_state_dict(checkpoint_file['net'])
lane_model = DataParallel(lane_model)
lane_model.eval()

print("\nloading Lane Generation Model ......")
paint_model = GenWrapper(norm="batch")
paint_model = DataParallel(paint_model)
checkpoint_file = torch.load(args.old_model)
paint_model.load_state_dict(checkpoint_file['state_dict'])
paint_model.eval()

# ------------ test ------------
out_path = os.path.join(args.output_folder, exp_name, "coord_output")
evaluation_path = os.path.join(args.output_folder, exp_name, "evaluate")
if not os.path.exists(os.path.join(args.output_folder, exp_name)):
    os.mkdir(os.path.join(args.output_folder, exp_name))
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(evaluation_path):
    os.mkdir(evaluation_path)

progressbar = tqdm(range(len(test_loader)))
with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        img = sample['img'].to(device)
        img_name = sample['img_name']

        # start = time.time()
        gen_img, _ = paint_model(img)
        # print("Gen time: ", time.time() - start)
        # gen_img = gen_img.detach()
        # gen_img = img

        # start = time.time()
        seg_pred, exist_pred = lane_model(gen_img)[:2]
        # print("Detect time: ", time.time() - start)

        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()

        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] > 0.1 else 0 for i in range(4)]
            lane_coords = getLane.prob2lines_CULane(seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18, thresh=0.1)

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

        progressbar.update(1)
progressbar.close()

# ---- evaluate ----
os.system("sh utils/lane_evaluation/CULane/Run.sh " + exp_name)

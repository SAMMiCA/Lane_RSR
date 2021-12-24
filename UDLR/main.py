import argparse
import os
import torch
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from data_loader.CULane import CULane
from data_loader.Carla import Carla
from inpainting import Painting


def main():
    parser = argparse.ArgumentParser(description='KMFace')
    parser.add_argument('--type', type=str, default='ae', help='type of the gan loss/ ae, naive, wgangp, hinge')
    parser.add_argument('--mode', type=str, default='test', help='Train mode or test mode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--old_model', type=str, default='')
    parser.add_argument('--old_model', type=str, default='./saved_models/Painting_KD_tanh_threshold_0_lamb_1_exp3/epoch_10.pth')
    parser.add_argument('--lane_model', type=str, default='./saved_models/exp3_best.pth')
    parser.add_argument('--results_folder', type=str, default='./logs/')
    parser.add_argument('--output_folder', type=str, default='./output/')
    parser.add_argument('--model_path', type=str, default='./saved_models/', help='Location to save to')

    opt = parser.parse_args()
    if opt.old_model == '':
        opt.old_model = None

    # opt.model = "Painting_KD_guidance_tanh_threshold_0_lamb_1_exp3_recursive"
    # opt.model = "Painting_no_KD_no_guidnace_exp3"
    # opt.model = 'Carla_test_output_image'
    opt.model = 'test'
    use_kd = True
    use_guidance = True
    opt.num_views = 1

    model_name = '{}'.format(opt.model)

    opt.results_folder = os.path.join(opt.results_folder, model_name)
    opt.output_folder = os.path.join(opt.output_folder, model_name)
    opt.model_path = os.path.join(opt.model_path, model_name)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    path = '../CULane'
    mode = opt.mode
    # ------------ train data ------------
    # # CULane mean, std
    # mean = (0.3598, 0.3653, 0.3662)
    # std = (0.2573, 0.2663, 0.2756)
    transform = Compose(Resize((800, 288)), ToTensor())
    # Normalize(mean=mean, std=std))
    # train_set = CULane(path, 'train', (800, 288))
    # test_set = CULane(path, 'test', (800, 288))
    train_set = CULane(path, 'train', transform)
    test_set = CULane(path, 'test', transform)
    model = Painting(model_path=opt.model_path, results_folder=opt.results_folder, lr=opt.lr, loss_type=opt.type, lane_model=opt.lane_model, old_model=opt.old_model, device=device)

    if opt.mode == 'train':
        model.train(train_set, opt.batch_size, opt.epochs, use_kd, use_guidance)
        # print("train")
    elif opt.mode == 'test':
        model.test(test_set, 1, opt.output_folder)
    else:
        print("You can choose only train or test")


if __name__ == '__main__':
    main()

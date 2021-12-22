# CRRN f1_score: 0.8112233240414933	 # UDLR f1_score: 0.7072364069321339 -- 500 / 2304
import os
import time
import torch
import argparse
from torch import optim
import torch.utils.data
import torchvision
# from model.ConvRRN import ST_EncDec as Model
from model.ConvRRN import Wrapper as Model
from model.CNN import CNN
import numpy as np
import cv2
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'): ## mode must be train, val, test
        super(Dataset, self).__init__() 
        print('Dataset initialize starts')
        self.mode = mode
        self.data_dir_path = '../Dataset/udlr+bicrrn/'
        self.size = (400,144)
        self.list_file = open(os.path.join(self.data_dir_path,"list", f"{mode}.txt"), 'r')
        self.data_list = []
        self.label_list = []
        prev_subdir_name = None
        set_counter = 0
        self.weight = torch.tensor([0.03905, 0.96095]) ## calculated by get_weight()
        while True:
            line = self.list_file.readline()
            while '90frame' in line:
                line = self.list_file.readline()
                if not line: break
            if not line: break

            ############################################################################################
            subdir_name = line[:-10]		####Make Data Multiple of 10
            if subdir_name ==  prev_subdir_name:
                set_counter += 1
            else:
                if set_counter%10 != 9 or set_counter!=0:
                    self.data_list = self.data_list[:-1*(set_counter%10+1)]
                    self.label_list = self.label_list[:-1*(set_counter%10+1)]

                set_counter = 0
            ############################################################################################

            data_name = f"{self.data_dir_path}/udlr_ENet_whiten{line[:-4]}jpg"
            label_name = f"{self.data_dir_path}/label{line[:-4]}png"
            self.data_list.append(data_name)
            self.label_list.append(label_name)
            prev_subdir_name = subdir_name
        print('Dataset initialize ends')

    def get_weight(self):
        ratio = 0
        count = 0
        for label in tqdm(self.label_list):
            label_img = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label_size = np.size(label_img)
            num_of_one = int(label_img.sum()/255)
            count += 1
            if count == 0:
                ratio = num_of_one/label_size
            else:
                ratio = count/(count+1)*ratio + num_of_one/label_size/(count+1)
                if count%1000 == 0:
                    print(ratio)
        return ratio
            

    def __getitem__(self, index):
        data_list = self.data_list[index*10:(index+1)*10]
        label_list = self.label_list[index*10:(index+1)*10]

        inputs = []
        targets = []
        for i in range(10):
            input_numpy = torch.round(torch.Tensor(cv2.resize(cv2.imread(data_list[i], cv2.IMREAD_GRAYSCALE),self.size))/255)
            target_img = torch.Tensor(cv2.resize(cv2.imread(label_list[i], cv2.IMREAD_GRAYSCALE),self.size))/255
            h,w = input_numpy.size()
            h = h+1
            w = w+1
            padded_input = torch.zeros(h, w, dtype=torch.float) 
            padded_input[:input_numpy.size(0), :input_numpy.size(1)] = input_numpy       
            padded_target = torch.zeros(h, w, dtype=torch.float) 
            padded_target[:input_numpy.size(0), :input_numpy.size(1)] = target_img
            inputs.append(padded_input)
            targets.append(padded_target)

        inputs = torch.stack(inputs,0)
        targets = torch.stack(targets,0)
        return inputs, targets, data_list

    def __len__(self): 	
        return int(len(self.data_list)/10)

def split(targets, inputs, outputs, i):
    target = targets.squeeze()[i]
    input = inputs.squeeze()[i]
    output = outputs.squeeze()[i]
    return target, input, output

def gt_lane_existance(gt_map, gt_th):
    if np.sum(gt_map) > gt_th:
        return True
    else:
        return False

def pred_lane_existance(pred_map, pred_th):
    if np.sum(pred_map) > pred_th:
        return True
    else:
        return False

def gt_pred_IOU(gt, pred, iou_th):
    assert(pred.shape == gt.shape)
    summation = pred+gt
    intersection = np.sum(summation>1)
    union = np.sum(summation>0)
    iou = intersection/union
    if iou > iou_th:
        return True
    else:
        return False

def f1_counter(targets, inputs, outputs, gt_th, pred_th, iou_th): #call after inference
    tp,fp,tn,fn = 0,0,0,0

    for i in range(10):
        if i < 9: ##last frame only
            continue
        target, input, output = split(targets, inputs, outputs, i)
        gt_existance = gt_lane_existance(target, gt_th)
        pred_existance = pred_lane_existance(output, pred_th)
        if gt_existance and pred_existance:
            if gt_pred_IOU(target, output, iou_th):
                tp+=1
            else:
                fp+=1
        elif not gt_existance and pred_existance:
            fp+=1
        elif not gt_existance and not pred_existance:
            tn+=1
        elif gt_existance and not pred_existance:
            fn+=1
    return tp,fp,tn,fn

        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PCB anomaly detection')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=501)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    #	parser.add_argument('--init_tr', type=float, default=0.25)
    #	parser.add_argument('--final_tr', type=float, default=0.0)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--is_attn', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--mode', type=str, default='ConvConjLSTM', help='ConvLSTM, ConvConjLSTM, CNN')
    parser.add_argument('--test_mode', type=str, default='CRRN', help='CRRN or UDLR')

    args = parser.parse_args()

    if not os.path.isdir('visualization'):
        os.mkdir('visualization')

    if args.mode == 'CNN':
        vis_dir = 'visualization/CNN'
    else:
        vis_dir = 'visualization/CRRN'

    if not os.path.isdir(vis_dir):
        os.mkdir(vis_dir)

    # train_dataset = Dataset(mode='test')
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                                num_workers=0)

    valid_dataset = Dataset(mode='test')
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=6)

    if args.mode == 'CNN':
        model = CNN([3, 64, 64], [5, 5])
    else:
        model = Model(args.mode, [1, 64, 64], [5, 5], args.n_layers, args.is_attn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if torch.cuda.device_count() > 1:
        if args.gpu_ids == None:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            device = torch.device('cuda:0')
        else:
            print("Let's use", len(args.gpu_ids), "GPUs!")
            device = torch.device('cuda:' + str(args.gpu_ids[0]))

    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('args.gpu_ids', args.gpu_ids)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)

    start_time = time.time()
    state = torch.load('parameters/290_parameter')
    model.load_state_dict(state['state_dict'])
    model.eval()
    with torch.no_grad():
        threshold = 0.99 # [0.95, 0.99]

        precision = [0,0,0,0,0,0,0,0,0]
        recall = [0,0,0,0,0,0,0,0,0]
        cnt = 0
        out_dir = './output'
        outputs_list = []
        inputs_list = []
        targets_list = []
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        for inputs, targets, data_list in tqdm(valid_dataloader):
            
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(2)
            targets = targets.to(device)

            if (len(inputs) != args.batch_size):
                break
            if args.test_mode == 'CRRN':
                outputs = model(inputs)  # batch by 10 by 2 by h by w
                outputs = torch.nn.Sigmoid()(outputs)
                outputs = outputs.squeeze()[:,1].detach().cpu().numpy()
                inputs = inputs.squeeze().detach().cpu().numpy()
                targets = targets.squeeze().detach().cpu().numpy()
            else:
                inputs = inputs.squeeze().detach().cpu().numpy()
                targets = targets.squeeze().detach().cpu().numpy()
                outputs = inputs
            
            outputs = (outputs>threshold)
            # outputs = torch.tensor(outputs)

        # for i in range(len(targets_list)):
        #     final_result = torch.cat((inputs_list[i].squeeze(), targets_list[i].float(), outputs_list[i][:,1]),0).unsqueeze(1)
        #     scaled_result = torchvision.utils.make_grid(final_result, nrow=10, padding=2, pad_value=1)
        #     os.makedirs('./testnoth', exist_ok=True)
        #     torchvision.utils.save_image(scaled_result, f'./testtest/{i}.png')
        # print('a')
                
            iou_th = 0.3
            # l1_th_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            # for i, l1_th in enumerate(l1_th_list):
            tp,fp,tn,fn = f1_counter(targets, inputs, outputs, gt_th=0, pred_th=2304, iou_th=iou_th)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        precision = total_tp/(total_tp+total_fp)
        recall = total_tp/(total_tp+total_tn)
        print(f"tp: {total_tp} \nfp: {total_fp} \ntn: {total_tn}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {2*precision*recall/(precision+recall)}")
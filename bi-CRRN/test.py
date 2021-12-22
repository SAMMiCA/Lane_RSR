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
        return inputs, targets

    def __len__(self): 	
        return int(len(self.data_list)/10)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PCB anomaly detection')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=501)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    #	parser.add_argument('--init_tr', type=float, default=0.25)
    #	parser.add_argument('--final_tr', type=float, default=0.0)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--is_attn', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--mode', type=str, default='ConvConjLSTM', help='ConvLSTM, ConvConjLSTM, CNN')

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

    #	criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(device))

    start_time = time.time()
    each_train_loss = []

    param = []
    #param = [50, 60, 70, 80, 90]
    #param = [50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
    param = [280]

    # param = torch.linspace(0, 800, 17).tolist()
    # param.append(60)

    f1score_file = open('f1score.txt', 'w+')
    f1score = []

    for i in param:
        param_num = int(i)
        state = torch.load('parameters/%d_parameter' %param_num)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        model.eval()

        origin_inputs = []
        origin_outputs = []
        out_prob = [];
        each_valid_loss = []
        threshold = torch.linspace(0.9, 1, 10).tolist()

        frame_temp = []
        tp, fp, fn, tn, f1 = {}, {}, {}, {}, {}
        predicted_framelevel = {}
        output_array = []
        target_array = []

        for th in threshold:
            tp[th] = 0;
            fp[th] = 0;
            fn[th] = 0;
            tn[th] = 0;
            f1[th] = 0;
            predicted_framelevel[th] = []


        # Validation
        with torch.set_grad_enabled(False):
            start = time.time()
            print(f"epoch {i}")
            for inputs, targets in tqdm(valid_dataloader):
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(2)
                targets = targets.to(device)

                if (len(inputs) != args.batch_size):
                    break

                origin_inputs.append(inputs.cpu())

                outputs = model(inputs)  # batch by 10 by 2 by h by w
                outputs = torch.nn.Sigmoid()(outputs)

                if args.mode == 'CNN':
                    temp_outputs = outputs.view(-1, 10, *temp_outputs.size()[1:])
                    out_prob.append(temp_outputs)
                else:
                    outputs = outputs.squeeze()
                    out_prob.append(outputs.cpu())

                origin_outputs.append(targets.cpu())  # batch by 10 by h by w

                if args.mode != 'CNN':
                    outputs = outputs.view(-1, *outputs.size()[2:])
                targets = targets.view(-1, *targets.size()[2:])

                # err = criterion(outputs, targets.long())
                # each_valid_loss.append(err.item())


                output_np = outputs[:, 1].cpu()
                output_array.append(output_np)
                target_np = targets.cpu()
                target_array.append(target_np)



                for th in threshold:
                    th_code = targets.int() * 2 + (outputs[:, 1] >= th).int()


                    tp[th] += (th_code == 3).sum().item()
                    fp[th] += (th_code == 1).sum().item()
                    fn[th] += (th_code == 2).sum().item()
                    tn[th] += (th_code == 0).sum().item()

                    # put output frame-level prediction
                    th_temp = (outputs[:, 1] >= th).int()
                    th_temp = th_temp.view(-1, 10, 145, 401)
                    for i in range(len(inputs)):

                        total_pixel = 1
                        # for j in range(origin_outputs[i].element_size()):
                        for j in range(3):
                            total_pixel = total_pixel * th_temp[i].size()[j]

                        defect_ratio = (th_temp[i] == 1).sum().item() / total_pixel

                        if (defect_ratio > 0.01):
                            predicted_framelevel[th].append(1)
                        else:
                            predicted_framelevel[th].append(0)


	# # pt파일을 받아놓은 후 sklearn 라이브러리 이용하여 pr curve등 뽑을 수 있음
    #     torch.save(torch.stack(output_array), 'output.pt')
    #     torch.save(torch.stack(target_array), 'target.pt')



        precision, recall, f1 = [], [], []
        accuracy = []
        for th in threshold:
            p = tp[th] / (tp[th] + fp[th] + 1e-7)
            r = tp[th] / (tp[th] + fn[th] + 1e-7)
            a = (tp[th] + tn[th]) / (tp[th] + tn[th] + fp[th] + fn[th] + 1e-7)
            if p != 0 and r != 0:
                precision.append(p)
                recall.append(r)
                accuracy.append(a)
                f1_each = 2 / (1 / p + 1 / r)
                f1.append(f1_each)
		

        print(max(f1))
        index_best = np.argmax(f1)
        print(index_best,'  ', threshold[index_best])
        print( 'accuracy:', accuracy[index_best], 'precision:', precision[index_best], ' recall:', recall[index_best], 'f1score:', f1[index_best] )
        f1score.append(str(i) + ':  ' + str(max(f1)) + '\n')
        # epoch_valid_loss = sum(each_valid_loss) / len(each_valid_loss)


        # save out_prob
        origin_inputs = torch.cat(origin_inputs, 0)  # N by 10 by 3 by h by w
        origin_outputs = torch.cat(origin_outputs, 0)
        out_prob = torch.cat(out_prob, 0)  # 10*k by 10 by h by w



        ##################### frame-level evaluation ###########################
   
        
        eval_framelevel_target = []
        for i in range(len(out_prob)):

            total_pixel = 1
            # for j in range(origin_outputs[i].element_size()):
            for j in range(3):
                total_pixel = total_pixel * origin_outputs[i].size()[j]

            defect_ratio = (origin_outputs[i] == 1).sum().item() / total_pixel

            # if (origin_outputs[i] == 1).sum().item() > threshold_framelevel:
            if (defect_ratio > 0.01):
                eval_framelevel_target.append(1)
            else:
                eval_framelevel_target.append(0)

        # evaluate the performance / calculate accuracy on 10 frame-level

        p_frame = []; r_frame = []; f1_frame = [] ; a_frame = []

        for th in threshold:

            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i in range(len(eval_framelevel_target)):
                if (predicted_framelevel[th][i] == 0 and eval_framelevel_target[i] == 0):
                    tn = tn + 1
                elif (predicted_framelevel[th][i] == 0 and eval_framelevel_target[i] == 1):
                    fn = fn + 1
                elif (predicted_framelevel[th][i] == 1 and eval_framelevel_target[i] == 0):
                    fp = fp + 1
                elif (predicted_framelevel[th][i] == 1 and eval_framelevel_target[i] == 1):
                    tp = tp + 1

            precision_framelevel = tp / (tp + fp + 1e-7)
            recall_framelevel = tp / (tp + fn + 1e-7)
            f1score_framelevel = 2 * precision_framelevel * recall_framelevel / (precision_framelevel + recall_framelevel + 1e-7)
            accuracy_framelevel = (tp + tn) / (tp + fp + tn + fn + 1e-7)

            # p_frame.append(precision_framelevel)
            # r_frame.append(recall_framelevel)
            # print('-> framelevel precision:', precision_framelevel, 'recall: ', recall_framelevel, 'f1score: ', f1score_framelevel, 'accuracy: ', accuracy_framelevel)

        # with open('plot_p_frame.txt', 'w') as p_f:
        #     for listitem in p_frame:
        #         p_f.write('%s\n' % listitem)

        # with open('plot_r_frame.txt', 'w') as r_f:
        #     for listitem in r_frame:
        #         r_f.write('%s\n' % listitem)


 
        ###########################################################################



        for i in range(len(out_prob)):

            if not os.path.isdir('%s/iter_%03d' % (vis_dir, state['epoch'])):
                os.mkdir('%s/iter_%03d' % (vis_dir, state['epoch']))

            scaled_file = '%s/iter_%03d/%04d.png' % (vis_dir, state['epoch'], i)

            scaled_input = torchvision.utils.make_grid(origin_inputs[i], nrow=10, padding=2, pad_value=1)
            output_and_target = torch.cat([origin_outputs[i].float(), out_prob[i][:, 1]], 0).unsqueeze(1)

            scaled_result = torchvision.utils.make_grid(output_and_target, nrow=10, padding=2, pad_value=1)
            scaled = torch.cat([scaled_input, scaled_result], 1)

            torchvision.utils.save_image(scaled, scaled_file)

        # print('-> Valid loss: %f' % (epoch_valid_loss))
        # print(' -> Confusion matrix')


    f1score_file.writelines(f1score)
    f1score_file.close()

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
from tqdm import tqdm
import cv2
import numpy as np

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
        
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):

    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PCB anomaly detection')
    parser.add_argument('--batch_size', type=int, default=26)
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

    train_dataset = Dataset(mode='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=12)

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


    filename = 'parameters/200_parameter'
    model, optimizer, epoch = load_checkpoint(model, optimizer, filename)

    #	criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(device))

    if not os.path.isdir('parameters'):
        os.mkdir('parameters')

    loss_file = open('loss.txt', 'w+')

    if epoch == '':
        num_epochs = 0
    else:
        num_epochs = epoch

    for epoch in range(num_epochs, args.num_epochs):
        start_time = time.time()
        each_train_loss = []
        model.train()

        origin_inputs = []
        origin_outputs = []
        out_prob = []
        with torch.set_grad_enabled(True):
            for inputs, targets in tqdm(train_dataloader):

                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(2)
                targets = targets.to(device)

                if (len(inputs) != args.batch_size):
                    break

                optimizer.zero_grad()
                outputs = model(inputs)


                if args.mode != 'CNN':
                    outputs = torch.nn.Sigmoid()(outputs.view(-1, *outputs.size()[2:]))


                targets = targets.view(-1, *targets.size()[2:])

                err = criterion(outputs, targets.long())

                err.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                each_train_loss.append(err.item())

        epoch_train_loss = sum(each_train_loss) / len(each_train_loss)
        print('epoch: ', epoch, 'loss: ', epoch_train_loss, ', time: %4.2f'%(time.time()-start_time))
        context = str(epoch) + ': ' + str(epoch_train_loss) + '\n'
        loss_file.writelines(context)
        if epoch % 10==0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            par_filepath = 'parameters/%d_parameter' % (epoch)
            torch.save(state, par_filepath)

    loss_file.close()

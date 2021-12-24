import os
import time
import timeit
from utils import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import DataParallel

from models.model import define_G
from models.model_ENET_SAD import ENet_SAD
from models.loss import Perceptual, Wgangp, Naive, Hinge
from tensorboardX import SummaryWriter

time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))


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


class DiscWrapper(nn.Module):
    def __init__(self):
        super(DiscWrapper, self).__init__()
        pass

    def forward(self):
        pass


class Painting:
    def __init__(self, model_path, results_folder, lr, loss_type, lane_model,
                 old_model=None, device=torch.device("cpu")):

        self.start_epoch = 0
        self.model_path = model_path
        self.loss_type = loss_type
        self.use_discriminator = False if loss_type == 'ae' else True
        self.results_folder = results_folder
        self.writer = None
        self.device = device

        self.lane_model = ENet_SAD(input_size=[800, 288], sad=True).to(self.device)
        checkpoint_file = torch.load(lane_model, map_location='cuda:0')
        self.lane_model.load_state_dict(checkpoint_file['net'])
        self.lane_model = DataParallel(self.lane_model)

        print("*" * 20)
        print("* Model name                 : Inpainting")
        print("* Loss type                  :", loss_type)
        print("*" * 20)

        self.paint_model = GenWrapper(norm="batch")
        self.paint_model = DataParallel(self.paint_model)

        if self.use_discriminator:
            self.discriminator = DiscWrapper()
            self.discriminator = DataParallel(self.discriminator)

            if loss_type == 'wgangp':
                self.gan_loss = Wgangp(self.discriminator, lambda_val=10)
            elif loss_type == 'naive':
                self.gan_loss = Naive(self.discriminator)
            elif loss_type == 'hinge':
                self.gan_loss = Hinge(self.discriminator)
            else:
                print("GAN Loss type is wrong")

        self.recon_loss = nn.L1Loss()
        self.perceptual_loss = Perceptual()

        self.optimizer = torch.optim.SGD(self.paint_model.parameters(), lr=lr, momentum=0.9)
        if self.use_discriminator:
            self.disc_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=lr, momentum=0.9)

        if old_model is not None:
            checkpoint_file = torch.load(old_model)
            old_model_name = os.path.split(old_model)[-1]
            old_model_name = old_model_name.split('.')[0]
            self.start_epoch = int(old_model_name.split('_')[-1])
            self.paint_model.load_state_dict(checkpoint_file['state_dict'])
            self.optimizer.load_state_dict(checkpoint_file['optimizer'])
            if self.use_discriminator:
                self.discriminator.load_state_dict(checkpoint_file['discriminator'])
                self.disc_optimizer.load_state_dict(checkpoint_file['disc_optimizer'])
            del checkpoint_file

    def optimize(self, image, segLabel, exist, use_kd=True, use_guidance=True, use_generated_image=False):
        output, att_bef = self.paint_model(image)
        r_loss = self.recon_loss(image, output)
        lane_loss = 0
        att_loss = 0

        if use_generated_image:
            if use_kd or use_guidance:
                # print("Use generated image")
                seg_pred, exist_pred, loss_seg, loss_exist, lane_loss, att_aft = self.lane_model(output, segLabel, exist, False)
        else:
            if use_kd:
                # print("Use original image")
                _, _, _, _, _, att_aft = self.lane_model(image, segLabel, exist, False)
            if use_guidance:
                seg_pred, exist_pred, loss_seg, loss_exist, lane_loss, _ = self.lane_model(output, segLabel, exist, False)

        mask = None
        mask1 = None
        mask2 = None

        if use_kd:
            for i in range(len(att_bef)):
                ones = torch.ones(att_bef[i].size()).cuda()
                zeros = torch.zeros(att_bef[i].size()).cuda()
                mask = torch.where(att_aft[i] > 0, ones, zeros)
                att_loss += self.recon_loss(att_bef[i]*mask, att_aft[i])

            mask1 = att_aft[-1]
            mask2 = att_bef[-1]

        # loss = r_loss + 10*lane_loss
        loss = r_loss + lane_loss + att_loss
        loss_list = [loss, r_loss, lane_loss, att_loss]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_list, output, mask, mask1, mask2

    def optimize_disc(self, image):
        output = self.discriminator(image)
        loss = 0

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        return loss

    def save_model(self, epoch):
        info = {'state_dict': self.paint_model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        if self.use_discriminator:
            info['discriminator'] = self.discriminator.state_dict()
            info['disc_optimizer'] = self.disc_optimizer.state_dict()
        model_out_path = "epoch_{}.pth".format(epoch)

        if not (os.path.exists(self.model_path)):
            os.makedirs(self.model_path)
        model_out_path = os.path.join(self.model_path, model_out_path)
        torch.save(info, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

        for i in range(0, epoch - 30, 10):
            if os.path.exists(os.path.join(self.model_path, "epoch_{}.pth".format(i))):
                os.remove(os.path.join(self.model_path, "epoch_{}.pth".format(i)))

    def train(self, train_set, batch_size, epochs, use_kd=True, use_guidance=True):
        self.writer = SummaryWriter(self.results_folder)
        self.lane_model.eval()
        self.paint_model.train()
        print("Starting the training process ... ")
        print("Using Knowledge Distillation Loss: ", use_kd)
        print("Using Guidance Loss: ", use_guidance)

        step = 0
        training_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                          collate_fn=train_set.collate, num_workers=6)

        for epoch in range(self.start_epoch, epochs):
            start = timeit.default_timer()  # record time at the start of epoch

            sum_loss = 0
            sum_r_loss = 0
            sum_l_loss = 0
            sum_att_loss = 0
            if self.use_discriminator:
                sum_g_loss = 0
                sum_d_loss = 0

            print("\nEpoch: %d" % epoch)

            for (num, batch) in enumerate(training_data_loader, 1):

                step += 1

                img = batch['img'].to(self.device)
                segLabel = batch['segLabel'].to(self.device, dtype=torch.int64)
                exist = batch['exist'].to(self.device)

                ##############################
                # Actual training part
                ##############################
                if self.use_discriminator:
                    for i in range(1):
                        disc_loss = self.optimize_disc(img)
                if epoch < 5:
                    loss, output, mask, mask1, mask2 = self.optimize(img, segLabel, exist, use_kd, use_guidance, False)
                else:
                    loss, output, mask, mask1, mask2 = self.optimize(img, segLabel, exist, use_kd, use_guidance, True)
                ##############################

                sum_loss += loss[0]
                sum_r_loss += loss[1]
                sum_l_loss += loss[2]
                sum_att_loss += loss[3]
                average_loss = sum_loss / num
                avg_r_loss = sum_r_loss / num
                avg_l_loss = sum_l_loss / num
                avg_att_loss = sum_att_loss / num

                if self.use_discriminator:
                    sum_d_loss += disc_loss
                    sum_g_loss += loss[-1]
                    avg_g_loss = sum_g_loss / num
                    avg_d_loss = sum_d_loss / num

                if num % 100 == 0:
                    self.writer.add_scalars('Train/loss', {'train': average_loss}, step)
                    self.writer.add_scalars('Train/r_loss', {'train': avg_r_loss}, step)
                    self.writer.add_scalars('Train/l_loss', {'train': avg_l_loss}, step)
                    self.writer.add_scalars('Train/att_loss', {'train': avg_att_loss}, step)
                    if self.use_discriminator:
                        self.writer.add_scalars('Train/g_loss', {'train': avg_g_loss}, step)
                        self.writer.add_scalars('Train/d_loss', {'train': avg_d_loss}, step)

                    print("===> Train Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, num,
                                                                             len(training_data_loader),
                                                                             average_loss))
                    if self.use_discriminator:
                        print("                                G: {:.4f} D: {:.4f}".format(avg_g_loss,
                                                                                           avg_d_loss))

                if num % 1000 == 0:
                    self.writer.add_image('Train/input', img.data.cpu(), step)
                    self.writer.add_image('Train/result', output.data.cpu(), step)
                    if use_kd:
                        self.writer.add_image('Train/mask', mask.data.cpu(), step)
                        self.writer.add_image('Train/mask1', mask1.data.cpu(), step)
                        self.writer.add_image('Train/mask2', mask2.data.cpu(), step)
                    self.save_model(epoch)

            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

        print("Training completed ...")

    def run(self, image):
        output, _ = self.paint_model(image)
        # output = image
        seg_pred, exist_pred, _, _, _, _ = self.lane_model(output, None, None, False)
        return output, seg_pred, exist_pred

    def test(self, test_set, batch_size, output_path):
        from torchvision import transforms
        import time
        self.paint_model.eval()
        self.lane_model.eval()
        print("Starting the training process ... ")

        test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size,
                                      shuffle=False, collate_fn=test_set.collate)

        for (num, batch) in enumerate(test_data_loader, 1):
            start = time.time()
            img = batch['img'].to(self.device)
            name = batch['img_name']

            
            output, seg_pred, exist_pred = self.run(img)
            

            for i in range(batch_size):
                output_image = output.permute(0, 2, 3, 1)[i].detach().cpu().numpy()
                output_image = (output_image * 255).astype('uint8')
                # output_image = transforms.ToPILImage()(output_image).convert("RGB")
                # output_image = np.array(output_image)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

                os.makedirs(output_path, exist_ok=True)
                folder_list = name[i].split('/')
                image_output_path = os.path.join(output_path, folder_list[-3])
                os.makedirs(image_output_path, exist_ok=True)
                image_output_path = os.path.join(image_output_path, folder_list[-2])
                os.makedirs(image_output_path, exist_ok=True)

                udlr_path = './udlr'
                udlr_output_path = os.path.join(udlr_path, folder_list[-3], folder_list[-2])
                os.makedirs(udlr_output_path, exist_ok=True)
                

                image_name = os.path.basename(name[i])
                # image_path = os.path.join(image_output_path, image_name)
                # output_image.save(image_path)

                # """ Lane Detection Result """
                seg_pred = seg_pred.detach().cpu().numpy()
                exist_pred = exist_pred.detach().cpu().numpy()
                output = output.squeeze().detach().cpu().numpy()


                img = img.permute(0,2,3,1).squeeze().cpu().numpy()*255
                lane_img = np.zeros_like(img)

                color = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype='uint8')

                coord_mask = np.argmax(seg_pred[i], axis=0)
                for j in range(0, 4):
                    if exist_pred[i, j] > 0.5:
                        lane_img[coord_mask == (j + 1)] = color[j]
                img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
                # cv2.putText(lane_img, "{}".format([1 if exist_pred[i, j] > 0.3 else 0 for j in range(4)]), (20, 20),
                            # cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                print("Inference time: ", time.time() - start)
                cv2.imwrite(os.path.join(image_output_path, image_name), lane_img)
                # cv2.imwrite(os.path.join('painted', image_name), output_image)
                
                # cv2.imwrite(result_lane_path, lane_img)

            del output


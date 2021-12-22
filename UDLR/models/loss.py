import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import numpy as np

np.random.seed(42)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        from torchvision import models
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Perceptual(nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        self.vgg = Vgg19().cuda()  # to(torch.device('cuda:1'))
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, fake_samples, real_samples):
        x_vgg, y_vgg = self.vgg(fake_samples), self.vgg(real_samples)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Wgangp(nn.Module):
    def __init__(self, discriminator, lambda_val):
        super(Wgangp, self).__init__()
        self.lambda_val = lambda_val
        self.discriminator = discriminator

    def calculate_gradient_penalty(self, real_samples, fake_samples):
        eta = torch.FloatTensor(real_samples.size(0), 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(real_samples.size(0), real_samples.size(1), real_samples.size(2), real_samples.size(3))
        eta = eta.cuda()  # to(torch.device('cuda:1'))
        interpolated = eta * real_samples + ((1 - eta) * fake_samples)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                  # to(torch.device('cuda:1')),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean() * self.lambda_val
        return grad_penalty

    def forward(self, fake_samples, real_samples=None, generator=True):
        fake = torch.mean(self.discriminator(fake_samples))
        if generator:
            loss = -fake
        else:
            real = torch.mean(self.discriminator(real_samples))
            gp = self.calculate_gradient_penalty(real_samples, fake_samples)
            loss = fake - real + gp
        return loss


class Naive(nn.Module):
    def __init__(self, discriminator):
        super(Naive, self).__init__()
        from torch.nn import BCEWithLogitsLoss

        self.discriminator = discriminator
        self.criterion = BCEWithLogitsLoss()

    def forward(self, fake_samples, real_samples=None, generator=True):
        fake = torch.squeeze(self.discriminator(fake_samples))
        if generator:
            real_loss = self.criterion(
                fake,
                torch.ones(fake.shape).cuda())
            loss = real_loss
        else:
            real = torch.squeeze(self.discriminator(real_samples))
            real_loss = self.criterion(
                real,
                torch.ones(real.shape).cuda())
            fake_loss = self.criterion(
                fake,
                torch.zeros(fake.shape).cuda())
            loss = (real_loss + fake_loss) / 2
        return loss


class Hinge(nn.Module):

    def __init__(self, discriminator):
        super(Hinge, self).__init__()

        self.discriminator = discriminator

    def forward(self, fake_samples, real_samples=None, generator=True, label=None):
        if label is None:
            fake = self.discriminator(fake_samples)
        else:
            fake = self.discriminator(fake_samples, label)

        if generator:
            loss = -torch.mean(fake)
        else:
            if label is None:
                real = self.discriminator(real_samples)
            else:
                real = self.discriminator(real_samples, label)

            loss = (torch.mean(torch.nn.ReLU()(1 - real)) +
                    torch.mean(torch.nn.ReLU()(1 + fake)))
        return loss


class Metrics:
    '''Compute tpr, fpr, fpr, fnr and balanced accuracy'''

    @classmethod
    def compute_tpr(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @staticmethod
    def _compute_tpr(y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @classmethod
    def compute_tnr(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @staticmethod
    def _compute_tnr(y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @classmethod
    def compute_ppv(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tp / (tp + fp)

    @classmethod
    def compute_npv(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_neg = 1 - y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tn / (tn + fn)

    @classmethod
    def balanced_accuracy(cls, y_true, y_pred):
        tpr = cls._compute_tpr(y_true, y_pred)
        tnr = cls._compute_tnr(y_true, y_pred)
        return (tpr + tnr) / 2


def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        """
        IoU Loss for individual examples
        inputs - N x Classes x H x W
        target_oneHot - N x Classes x H x W
        """

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        # return 1 - loss.mean()
        return -(loss.mean() - 1.)


'''
def IoULoss(pred, target):
    epsilon = 0.0001
    inter = torch.dot(pred.view(-1), target.view(-1))
    union = torch.sum(pred) + torch.sum(target)
    l_iou = 1 - inter / (union + epsilon)
    l_iou = torch.mean(l_iou)
    return l_iou
'''


class DCLoss(torch.autograd.Function):
    """Dice coeff for individual examples"""

    @staticmethod
    def forward(ctx, pred, target):
        #target = target.type(torch.FloatTensor).to(cfg.device)
        #target = target.type(torch.FloatTensor).cuda()
        target = target.type(torch.FloatTensor).to(pred.device)
        pred = torch.abs(pred)
        eps = 0.0001
        inter = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps
        ctx.save_for_backward(pred, target, inter, union)
        t = (2 * inter + eps) / union
        return t

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pred, target, inter, union = ctx.saved_variables
        grad_input = grad_output * 2 * (target * union - inter) / (union * union)
        return grad_input, None












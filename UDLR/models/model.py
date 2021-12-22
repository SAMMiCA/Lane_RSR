from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        encoder = ResnetEncoder(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        decoder = ResnetDecoder(output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        encoder = ResnetEncoder(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
        decoder = ResnetDecoder(output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    encoder = init_net(encoder, init_type, init_gain, gpu_ids)
    decoder = init_net(decoder, init_type, init_gain, gpu_ids)
    return encoder, decoder


class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2

        encoder1 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        encoder1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * 2),
                     nn.ReLU(True)]

        encoder2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * 4),
                    nn.ReLU(True)]

        encoder3 = []
        for i in range(n_blocks//3):  # add ResNet blocks
            encoder3 += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias)]
        encoder4 = []
        for i in range(n_blocks//3, n_blocks):  # add ResNet blocks
            encoder4 += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder1 = nn.Sequential(*encoder1)
        self.encoder2 = nn.Sequential(*encoder2)
        self.encoder3 = nn.Sequential(*encoder3)
        self.encoder4 = nn.Sequential(*encoder4)

        self.connection1 = nn.Sequential(
            nn.Conv2d(ngf * 2, 1, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Upsample(scale_factor=0.5),
            nn.InstanceNorm2d(1), nn.Tanh())

        self.connection2 = nn.Sequential(
            nn.Conv2d(ngf * 4, 1, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Upsample(scale_factor=0.5),
            nn.InstanceNorm2d(1), nn.Tanh())

        self.connection3 = nn.Sequential(
            nn.Conv2d(ngf * 4, 1, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Upsample(scale_factor=0.5),
            nn.InstanceNorm2d(1), nn.Tanh())

        self.connection4 = nn.Sequential(
            nn.Conv2d(ngf * 4, 1, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Upsample(scale_factor=0.5),
            nn.InstanceNorm2d(1), nn.Tanh())

    def forward(self, image):
        x1 = self.encoder1(image)
        at1 = self.connection1(x1)
        x2 = self.encoder2(x1)
        at2 = self.connection2(x2)
        x3 = self.encoder3(x2)
        at3 = self.connection3(x3)
        x4 = self.encoder4(x3)
        at4 = self.connection4(x4)
        return x4, [at1, at2, at3, at4]


class ResnetDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2

        decoder = []

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, latent):
        return self.decoder(latent)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.last = last

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

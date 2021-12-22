import torch
import torch.nn as nn 
from model.core_module.ConvLSTM import ConvLSTM
from model.SpatialAttention import SpatialAttention 
class StackedConvLSTM(nn.Module):
	def __init__(self, mode, channel, kernel, n_layers, is_attn=False):
		super(StackedConvLSTM, self).__init__()
		self.channel = channel
		self.kernel = kernel
		self.n_layers = n_layers
		self.is_attn = is_attn

		if mode=='ConvLSTM':
			from model.core_module.ConvLSTM import ConvLSTM
		elif mode=='ConvConjLSTM':
			from model.core_module.ConvConjLSTM import ConvConjLSTM as ConvLSTM
		elif mode=='ConvSTLSTM':
			from model.core_module.ConvSTLSTM import ConvSTLSTM as ConvLSTM
		elif mode=='ConvGridLSTM':
			from model.core_module.ConvGridLSTM import ConvGridLSTM as ConvLSTM

		conv_lstm = [ConvLSTM(channel, kernel, 2)]
		for _ in range(1, n_layers):
			conv_lstm.append(ConvLSTM(channel, kernel))

		self.conv_lstm = nn.Sequential(*conv_lstm)

		if is_attn:
			s_attn = [SpatialAttention(channel, kernel) for _ in range(n_layers)]
			self.s_attn = nn.Sequential(*s_attn)

	def init_cell_weight(self):
		for i in range(len(self.conv_lstm)):
			self.conv_lstm[i].reset_cell_weight()

	def forward(self, inputs, hiddens=None):
		#print('inputs in stackedconvLSTM', inputs.size())
		steps = inputs.size(1)
		#print('steps',steps)

		if hiddens is None:
			hiddens = [None for i in range(self.n_layers)]

		xm = [[inputs[:,j], None] for j in range(steps)]
		#print('xm', xm[0])
		attns = None
		if self.is_attn:
			attns = []

		# each layer
		for i in range(self.n_layers):

			ym = []
			if self.is_attn:
				attn = []

			h_mask = None
			outputs = []
			# each time step
			for j in range(steps):
				h, c = self.conv_lstm[i](xm[j], hiddens[i])

				if type(h) is tuple:
					h_up, h = h
				else:
					h_up = h

				if type(c) is tuple:
					c, m = c
				else:
					m = c

				if self.is_attn:
					h, a = self.s_attn[i](h)
					attn.append(a)

				if self.training:
					if h_mask is None:
						h_mask = h.new(*h.size()).bernoulli_(0.8).div(0.8)
						h = h*h_mask

				ym.append([h_up,m])
				hiddens[i] = [h,c]
				outputs.append(h)
			if self.is_attn:
				attns.append(attn)

			xm = ym

		return hiddens, attns, outputs


class StackedBiConvLSTM(nn.Module):
	def __init__(self, mode, channel, kernel, n_layers, is_attn=False):
		super(StackedBiConvLSTM, self).__init__()
		self.channel = channel
		self.kernel = kernel
		self.n_layers = n_layers
		self.is_attn = is_attn

		if mode == 'ConvLSTM':
			from model.core_module.ConvLSTM import ConvLSTM
		elif mode == 'ConvConjLSTM':
			from model.core_module.ConvConjLSTM import ConvConjLSTM as ConvLSTM
		elif mode == 'ConvSTLSTM':
			from model.core_module.ConvSTLSTM import ConvSTLSTM as ConvLSTM
		elif mode == 'ConvGridLSTM':
			from model.core_module.ConvGridLSTM import ConvGridLSTM as ConvLSTM

		conv_lstm = [ConvLSTM(channel, kernel, 2)]
		conv_lstm_backward = [ConvLSTM(channel, kernel, 2)]
		# for _ in range(1, n_layers):
		# 	conv_lstm.append(ConvLSTM(channel, kernel))
		# 	conv_lstm_backward.append(ConvLSTM(channel, kernel))

		for _ in range(1, n_layers):
			conv_lstm.append(ConvLSTM(channel, kernel, layer=1))
			conv_lstm_backward.append(ConvLSTM(channel, kernel, layer=1))

		self.conv_lstm = nn.Sequential(*conv_lstm)
		self.conv_lstm_backward = nn.Sequential(*conv_lstm_backward)

		if is_attn:
			s_attn = [SpatialAttention(channel, kernel) for _ in range(n_layers)]
			self.s_attn = nn.Sequential(*s_attn)

	def init_cell_weight(self):
		for i in range(len(self.conv_lstm)):
			self.conv_lstm[i].reset_cell_weight()
			self.conv_lstm_backward[i].reset_cell_weight()

	def forward(self, inputs, hiddens_comb=[None, None]):
		# print('inputs in stackedconvLSTM', inputs.size())
		steps = inputs.size(1)

		hiddens = hiddens_comb[0]
		hiddens_rev = hiddens_comb[1]
		# print('steps',steps)
		# if hiddens is None:
		# 	hiddens = [None for i in range(self.n_layers)]
		#
		# if hiddens_rev is None:
		# 	hiddens_rev = [None for i in range(self.n_layers)]




		xm = [[inputs[:, j], None] for j in range(steps)]
		# print('xm', xm[0])
		xm_rev = xm
		xm_rev.reverse()

		attns = None
		if self.is_attn:
			attns = []

		for i in range(self.n_layers):

			ym = []
			ym_rev = []
			if self.is_attn:
				attn = []

			h_mask = None
			h_mask_rev = None
			outputs = []
			outputs_rev = []
			hiddens = [None for i in range(self.n_layers)]
			hiddens_rev = [None for i in range(self.n_layers)]

			for j in range(steps):
				h, c = self.conv_lstm[i](xm[j], hiddens[i])
				h_rev, c_rev = self.conv_lstm_backward[i](xm_rev[j], hiddens_rev[i])

				if type(h) is tuple:
					h_up, h = h
				else:
					h_up = h

				# reverse
				if type(h_rev) is tuple:
					h_up_rev, h_rev = h_rev
				else:
					h_up_rev = h_rev

				if type(c) is tuple:
					c, m = c
				else:
					m = c
				# reverse
				if type(c_rev) is tuple:
					c_rev, m_rev = c_rev
				else:
					m_rev = c_rev

				if self.is_attn:
					h, a = self.s_attn[i](h)
					attn.append(a)

				if self.training:
					if h_mask is None:
						h_mask = h.new(*h.size()).bernoulli_(0.8).div(0.8)
						h = h * h_mask
						h_mask_rev= h_rev.new(*h_rev.size()).bernoulli_(0.8).div(0.8)
						h_rev = h_rev * h_mask_rev

				ym.append([h_up, m])
				hiddens[i] = [h, c]
				outputs.append(h)

				ym_rev.append([h_up_rev, m_rev])
				hiddens_rev[i] = [h_rev, c_rev]
				outputs_rev.append(h_rev)

			if self.is_attn:
				attns.append(attn)

			xm = ym
			xm_rev = ym_rev

			xm_comb = []

			for k in range(steps):
				temp = []
				for l in range(2):
					temp.append(torch.cat((xm[k][l], xm_rev[9-k][l]), 1))
				xm_comb.append(temp)

			xm = xm_comb
			xm_rev = xm_comb

		hidden_comb = [hiddens, hiddens_rev]

		outputs_comb = []
		for i in range(steps):
			outputs_comb.append(torch.cat((outputs[i], outputs_rev[9-i]), 1))

		return hidden_comb, attns, outputs_comb

class S_EncDec(nn.Module): 
	def __init__(self, channels, kernels, encoder=True): 

		super(S_EncDec, self).__init__()
		layer = []

		if not encoder: 
			channels.reverse() 
			kernels.reverse() 

		for i in range(len(kernels)):
		#for i in range(2):
			if encoder: 
#				conv = nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernels[i], stride=2, bias=True)
#			conv = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=kernels[i], stride=2, bias=True)
				conv = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=1, bias=True)
				layer.append(conv) 
				layer.append(nn.ReLU())
				layer.append(nn.BatchNorm2d(25))
#				layer.append(nn.BatchNorm2d(channels[i+1]))
				conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=5, stride=2, bias=True)#
				# conv2 = nn.Conv2d(in_channels=25, out_channels=32, kernel_size=5, stride=2, bias=True)#
				layer.append(conv2)
				layer.append(nn.ReLU())
				layer.append(nn.BatchNorm2d(64))
				# layer.append(nn.BatchNorm2d(32))

			if not encoder: 
				#conv = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], 2, bias=True)
#				conv = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=13, stride=4, bias=True)
#				conv = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=17, stride=4, bias=True)
#				conv = nn.ConvTranspose2d(in_channels=64, out_channels=25, kernel_size=17, stride=4, bias=True)
				conv = nn.ConvTranspose2d(in_channels=64, out_channels=25, kernel_size=5, stride=2, bias=True)
				layer.append(conv)
				if i<len(kernels)-1: 
					layer.append(nn.ReLU())
#					layer.append(nn.BatchNorm2d(channels[i+1]))
					layer.append(nn.BatchNorm2d(25))
				conv2 = nn.ConvTranspose2d(in_channels=25, out_channels=2, kernel_size=9, stride=2, bias=True)
				layer.append(conv2)
				if i < len(kernels) - 1:
					layer.append(nn.ReLU())
					layer.append(nn.BatchNorm2d(2))

		self.layer = nn.Sequential(*layer)
		self.tmp = encoder

	def forward(self, inputs): 

		inputs_size = inputs.size()
		inputs = inputs.view(-1, *inputs_size[-3:])
		outputs = self.layer(inputs)   # inputs (100,2,93,161) # outputs (100,64,45,79)
		return outputs.view(*inputs_size[:-3], *outputs.size()[-3:])


class ST_Encoder(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers, is_attn): 
		super(ST_Encoder, self).__init__()

		self.channels = channels 
		self.kernels = kernels

		# self.conv_lstm = StackedConvLSTM(mode, channels[-1], kernels[-1], n_layers, is_attn)
		self.conv_lstm = StackedBiConvLSTM(mode, channels[-1], kernels[-1], n_layers, is_attn)

	def forward(self, inputs):
		# inputs = self.s_enc(inputs)
		hidden, attns, outputs = self.conv_lstm(inputs)
		outputs = torch.stack(outputs, 1)
		return hidden, attns, outputs

	def init_cell_weight(self): 
		self.conv_lstm.init_cell_weight()


class ST_Decoder(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers): 
	
		super(ST_Decoder, self).__init__() 

		self.channels = channels 
		self.kernels = kernels
		self.conv_lstm = StackedConvLSTM(mode, channels[-1], kernels[-1], n_layers)
		channels[0] = 2

	def forward(self, hiddens, targets, attns=None, teacher_ratio=1.0): 

		steps = targets.size(1) 
		timesteps = range(steps-1, 0, -1) 

		if attns is not None:
			for i in range(len(attns)):
				hiddens[i][0] = hiddens[i][0]-(attns[i][-1].expand_as(hiddens[i][0])) 

		# outputs = [self.s_dec(hiddens[-1][0])]
		outputs = [hiddens[-1][0]]

		for step in timesteps:
#			x_empty = hiddens[0][0].new(hiddens[0][0].size(0), 10, 64, 45, 79).zero_()
#			x_empty = hiddens[0][0].new(hiddens[0][0].size(0), 10, 64, 21, 38).zero_()
			x_empty = hiddens[0][0].new(hiddens[0][0].size(0), 10, 64, 43, 77).zero_()
			# print('hidden in ST_Decoder', hiddens[0][0].size())
			# print('x_empty in ST_Decoder', x_empty.size())

			hiddens, _ = self.conv_lstm(x_empty, hiddens)
			if attns is not None: 
				for i in range(len(attns)): 
					hiddens[i][0] = hiddens[i][0]-(attns[i][step-1].expand_as(hiddens[i][0])) 
			# print('hiddens in ST_Decoder',hiddens[2][1].size(), hiddens[9][1].size())

			# print('hiddens in ST_Decoder', hiddens[-1][0].size())

			# outputs.append(self.s_dec(hiddens[-1][0]))
			outputs.append(hiddens[-1][0])
			# print('outputs in for', outputs[0].size())

		outputs = torch.stack(outputs[::-1], 1) # [::-1 ] : reverse value sequence
		return outputs

	def init_cell_weight(self): 
		self.conv_lstm.init_cell_weight() 

		
	
class ST_EncDec(nn.Module): 
	def __init__(self, mode, channels, kernels, n_layers, is_attn=False): 
		super(ST_EncDec, self).__init__()

		self.st_encoder = ST_Encoder(mode, channels, kernels, n_layers, is_attn) 
		self.st_decoder = ST_Decoder(mode, channels, kernels, n_layers)

	def forward(self, inputs, teacher_ratio=1.0): 

		self.st_encoder.init_cell_weight() 
		self.st_decoder.init_cell_weight() 

		# input

		hidden, attns, outputs = self.st_encoder(inputs)
		# print("hidden in ST_ENCDEC", hidden[0][0].size(), hidden[0][1].size())

		# outputs = self.st_decoder(hidden, inputs, attns, teacher_ratio=teacher_ratio)

		return outputs


class Wrapper(nn.Module):
	def __init__(self, mode, channels, kernels, n_layers, is_attn=False):
		super(Wrapper, self).__init__()

		# self.channels = []
		# self.kernels = []
		self.channels = channels
		self.kernels = kernels
		self.s_enc = S_EncDec(channels[:-1], kernels[:-1], encoder=True)
		self.st_encdec = ST_EncDec(mode, channels, kernels, n_layers, is_attn)
		self.st_encdec_bi = ST_EncDec(mode, [1,64,64], [5,5], n_layers, is_attn)   # add bi-directional
		self.s_dec = S_EncDec(channels[:-1], kernels[:-1], encoder=False)
		self.conv = nn.Conv2d(128, 64, [1,1], 1, bias=True)

	def forward(self, inputs):

		# step1. S-Encoder

		outputs_senc = self.s_enc(inputs)
		outputs_senc_flip = torch.flip(outputs_senc, [1])  # make bi-directional input

		# outputs_senc = torch.cat((outputs_senc, outputs_senc_flip),2)

		# step2. ST-Encoder and ST-Decoder

		outputs_st = self.st_encdec(outputs_senc)
		# outputs_st_bi = self.st_encdec_bi(outputs_senc_flip)

		# step3. concat outputs_senc
		# inputs_st = torch.cat((outputs_st, outputs_st_bi),2)

		# step4. one by one convolution to shrink dim
		inputs_size = outputs_st.size()
		outputs_st = outputs_st.view(-1, *inputs_size[-3:])
		outputs = self.conv(outputs_st)  # inputs (100,2,93,161) # outputs (100,64,45,79), 1 by 1 convolution
		outputs_st = outputs.view(*inputs_size[:-3], *outputs.size()[-3:])

		# step5. S-Decoder
		outputs_sdec= self.s_dec(outputs_st)




		return outputs_sdec

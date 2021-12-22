import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter 

class S_Attention(nn.Module): 
	def __init__(self, channel, kernel): 
		super(S_Attention, self).__init__() 
		pad = int((kernel-1)/2) 
		self.attn = nn.Conv2d(channel, 1, kernel, 1, pad, bias=True) 
		
	def forward(self, input): 
		attn = 0.5*nn.Sigmoid()(self.attn(input))+0.5 
		output = input*attn.expand_as(input) 
		return output, attn 


class ConvLSTM(nn.Module): 
	def __init__(self, channel, kernel): 

		super(ConvLSTM, self).__init__() 

		self.channel = channel 

		pad = int((kernel-1)/2) 
		self.h2h = nn.Conv2d(channel, 4*channel, kernel, 1, pad) 

		self.Wci = None 
		self.Wcf = None 
		self.Wco = None 

	def forward(self, input, hidden=None, drop_mask=None): 

		if hidden is None: 
			hidden = self.init_hidden(input.size()) 
			self.init_cell_weight(input.size()) 

		if self.Wci is None: 
			self.init_cell_weight(input.size()) 

		hx, cx = hidden 

		# add horizontal dropout 
		if self.training and drop_mask is not None: 
			hx = hx*drop_mask

		gates = input + self.h2h(hx) 
		ingate, cellgate, outgate, forgetgate = gates.chunk(4, 1) 
		cellgate = F.tanh(cellgate) 

		ingate = F.sigmoid(ingate + cx*self.Wci) 
		forgetgate = F.sigmoid(forgetgate + cx*self.Wcf) 
		cy = (forgetgate*cx) + (ingate*cellgate) 

		outgate = F.sigmoid(outgate + cy*self.Wco) 
		hy = outgate * F.tanh(cy) 

		return hy, cy 

	def init_hidden(self, sz): 

		sz = [sz[0], self.channel, sz[2], sz[3]] 
		weight = next(self.parameters()).data 
		
		return (weight.new(*sz).zero_().requires_grad_(), 
			    weight.new(*sz).zero_().requires_grad_()) 

	def init_cell_weight(self, sz): 
		sz = [sz[0], self.channel, sz[2], sz[3]] 
		weight = next(self.parameters()).data 

		self.Wci = weight.new(*sz).zero_().requires_grad_() 
		self.Wcf = weight.new(*sz).zero_().requires_grad_() 
		self.Wco = weight.new(*sz).zero_().requires_grad_() 


class S_EncDec(nn.Module): 
	def __init__(self, channels, kernels, encoder=True): 

		super(S_EncDec, self).__init__()
		layer = [] 		

		if not encoder: 
			channels.reverse() 
			kernels.reverse() 

		for i in range(len(kernels)):
			
			if not encoder: 
				conv = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], 2, bias=True) 
			else: 
				out_channel = channels[i+1] if i<len(kernels)-1 else 4*channels[i+1]
				conv = nn.Conv2d(channels[i], out_channel, kernels[i], 2, bias=True) 

			layer.append(conv) 
			if i<len(kernels)-1: 
				layer.append(nn.ReLU())
				layer.append(nn.BatchNorm2d(channels[i+1])) 

		self.layer = nn.Sequential(*layer) 

	def forward(self, inputs): 

		inputs_size = inputs.size() 
		inputs = inputs.view(-1, *inputs_size[-3:])
		outputs = self.layer(inputs) 
		return outputs.view(*inputs_size[:-3], *outputs.size()[-3:]) 


class ST_Encoder(nn.Module): 
	def __init__(self, channels, kernels): 
	
		super(ST_Encoder, self).__init__() 

		self.channels = channels 
		self.kernels = kernels 
		self.s_enc = S_EncDec(channels, kernels, encoder=True) 
		self.conv_lstm = ConvLSTM(channels[-1], kernels[-1]) 

	def forward(self, inputs): 
	
		inputs = self.s_enc(inputs) 

		steps = inputs.size(1) 
		hidden = None 

		drop_mask = None 
		if self.training: 
			drop_mask = inputs.new(inputs.size(0), self.channels[-1], *inputs.size()[-2:]).bernoulli_(0.8).div(0.8)

		for step in range(steps): 
			hy, cy = self.conv_lstm(inputs[:,step], hidden, drop_mask=drop_mask)
			hidden = hy, cy 
	
		return hidden


#class ST_Decoder(nn.Module): 
#	def __init__(self, channels, kernels, is_attn=False): 
#		
#		super(ST_Decoder, self).__init__() 
#
#		self.channels = channels 
#		self.kernels = kernels 
#		self.is_attn = is_attn 
#		self.s_enc = S_EncDec(channels, kernels, encoder=True) 
#		self.conv_lstm = ConvLSTM(channels[-1], kernels[-1]) 
#		self.s_dec = S_EncDec(channels, kernels, encoder=False) 
#
#	def forward(self, hidden, targets, teacher_ratio=1.0, attn_scores=None): # targets for teacher forcing 
#
#		steps = targets.size(1) 
#		timesteps = range(steps-1,0,-1) 
#
#		hx, cx = hidden 
#
#		if self.is_attn: 
#			hx = hx*attn_scores[-1].expand_as(hx)
#
#		outputs = [self.s_dec(hx)]  
#
#		drop_mask = None 
#		if self.training: 
#			drop_mask = hx.new(*hx.size()).bernoulli_(0.8).div(0.8) 
#
#		for step in timesteps: 
#
#			if self.training: 
#				x = targets[:, step]
#				teacher_mask = x.new(x.size(0), 1, 1, 1).bernoulli_(teacher_ratio)
#				teacher_mask = teacher_mask.expand_as(x) 
#
#				x = teacher_mask*x + (1-teacher_mask)*outputs[-1] 
#				x = x.detach() 
#
#			else: 
#				x = outputs[-1]
#
#			x = self.s_enc(x)
#			hx, cx = self.conv_lstm(x, (hx,cx), drop_mask=drop_mask) 
#
#			if self.is_attn: 
#				hx = hx*attn_scores[step-1].expand_as(hx) 
#
#			outputs.append(self.s_dec(hx))
#
#		outputs = torch.stack(outputs[::-1], 1) 
#		return outputs 


#class ST_EncDec(nn.Module): 
#	def __init__(self, channels, kernels, is_attn=False): 
#		super(ST_EncDec, self).__init__() 
#
#		self.st_encoder = ST_Encoder(channels, kernels, is_attn) 
#		self.st_decoder = ST_Decoder(channels, kernels, is_attn) 
#
#	def forward(self, inputs, teacher_ratio=1.0): 
#
#		hidden, attn_scores = self.st_encoder(inputs) 
#		outputs = self.st_decoder(hidden, inputs, teacher_ratio=teacher_ratio, attn_scores=attn_scores)
#		return outputs

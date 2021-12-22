import torch 
import torch.nn as nn 

class SpatialAttention(nn.Module): 
	def __init__(self, channel, kernel): 
		super(SpatialAttention, self).__init__() 
		pad = int((kernel-1)/2) 
		self.attn = nn.Conv2d(channel, 1, kernel, 1, pad, bias=True) 
		
	def forward(self, input): 
		attn = nn.Tanh()(self.attn(input))
		output = input+attn.expand_as(input) 
		return output, attn 


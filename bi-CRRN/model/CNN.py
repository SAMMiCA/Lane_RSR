import torch
import torch.nn as nn 
		
	
class CNN(nn.Module): 
	def __init__(self, channels, kernels): 
	# channels: [3,64,64], kernels: [5,5] 
		super(CNN, self).__init__() 
	
		layer = [] 	
		for i in range(len(kernels)): 
			conv = nn.Conv2d(channels[i], channels[i+1], kernels[i], 2, bias=True) 
			layer.append(conv)
			layer.append(nn.ReLU()) 
			layer.append(nn.BatchNorm2d(channels[i+1])) 

		self.encoder = nn.Sequential(*layer) 

		self.core = nn.Conv2d(channels[-1], channels[-1], kernels[-1], 1, int((kernels[-1]-1)/2)) 

		channels.reverse()
		kernels.reverse() 

		channels[-1] = 2

		layer = []
		for i in range(len(kernels)): 
			conv = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], 2, bias=True) 
			layer.append(conv)
			if i<len(kernels)-1: 
				layer.append(nn.ReLU()) 
				layer.append(nn.BatchNorm2d(channels[i+1])) 

		self.decoder = nn.Sequential(*layer) 


	def forward(self, inputs): 

		# N by 10 by 3 by H by W -> 10N by 3 by H by W

		inputs = inputs.view(-1, *inputs.size()[2:]) 

		encoded = self.encoder(inputs) 

		hidden = self.core(encoded)
		decoded = self.decoder(hidden) 

		return decoded

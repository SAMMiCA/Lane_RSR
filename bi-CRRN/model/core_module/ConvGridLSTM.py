import torch
import torch.nn as nn 
import torch.nn.functional as F 

class ConvGridLSTM(nn.Module): 
	def __init__(self, channel, kernel, stride=1): 
		super(ConvGridLSTM, self).__init__() 
		self.channel = channel
		self.kernel = kernel

		if stride==1: 
			self.i2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2))
			self.i2m = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2))
		else: 
			self.i2h = nn.Conv2d(channel, 4*channel, kernel, stride)
			self.i2m = nn.Conv2d(channel, 4*channel, kernel, stride)

		self.i2h.bias[3*channel:].data.fill_(1) 
		self.i2m.bias[3*channel:].data.fill_(1) 

		self.h2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2)) 
		self.h2m = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2)) 

		self.Wci = None 
		self.Wcf = None 
		self.Wco = None 

		self.Wmi = None 
		self.Wmf = None 
		self.Wmo = None 


	def forward(self, input, hidden): 
		x, m = input 

		gates_c = self.i2h(x) 
		gates_m = self.i2m(x) 
	
		if hidden is None:
			hidden = self.init_hidden(gates_c.size()) 
			self.init_cell_weight(gates_c.size()) 

		if self.Wci is None: 
			self.init_cell_weight(gates_c.size()) 

		hx,cx = hidden 
		gates_c = gates_c + self.h2h(hx) 
		if m is not None: 
			gates_m = gates_m + self.h2h(m) 

		ggate_c, ingate_c, outgate_c, forgetgate_c = gates_c.chunk(4,1) 
		ggate_m, ingate_m, outgate_m, forgetgate_m = gates_m.chunk(4,1) 
		
		ggate_c = F.tanh(ggate_c)
		ingate_c = F.sigmoid(ingate_c + cx*self.Wci)
		forgetgate_c = F.sigmoid(forgetgate_c + cx*self.Wcf)
		next_c = (forgetgate_c*cx) + (ingate_c*ggate_c) 
		outgate_c = F.sigmoid(outgate_c + next_c*self.Wco) 
		next_h = outgate_c*F.tanh(next_c) 
	
		if m is None: 
			m = 0 	
		ggate_m = F.tanh(ggate_m)
		ingate_m = F.sigmoid(ingate_m + m*self.Wmi)
		forgetgate_m = F.sigmoid(forgetgate_m + m*self.Wmf)
		next_m = (forgetgate_m*m) + (ingate_m*ggate_m) 
		outgate_m = F.sigmoid(outgate_m + next_m*self.Wmo) 
		next_x = outgate_m*F.tanh(next_m) 
		
		return (next_x, next_h), (next_c, next_m) 



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

		self.Wmi = weight.new(*sz).zero_().requires_grad_() 
		self.Wmf = weight.new(*sz).zero_().requires_grad_() 
		self.Wmo = weight.new(*sz).zero_().requires_grad_() 

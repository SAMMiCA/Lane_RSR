import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvLSTM(nn.Module): 
	def __init__(self, channel, kernel, stride=1, layer=0):

		super(ConvLSTM, self).__init__() 
		self.channel = channel
		self.kernel = kernel
		self.layer = layer

		if layer == 0:
			if stride == 1:
				self.i2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2))
			else:
				self.i2h = nn.Conv2d(channel, 4*channel, kernel, stride)

			self.i2h.bias[3*channel:].data.fill_(1)
			self.h2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2))
		else:
			if stride == 1:
				self.i2h = nn.Conv2d(2*channel, 4*channel, kernel, 1, int((kernel - 1) / 2))
			else:
				self.i2h = nn.Conv2d(2*channel, 4*channel, kernel, stride)

			self.i2h.bias[3 * channel:].data.fill_(1)
			self.h2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel - 1) / 2))

		self.Wci = None 
		self.Wcf = None 
		self.Wco = None 

	def reset_cell_weight(self): 
		self.Wci = None 
		self.Wcf = None 
		self.Wco = None 

	def forward(self, input, hidden): 
		#print('input', len(input))
		x,m = input 
		#print("abc")
		#print("x in convLSTM", x.size())
		gates = self.i2h(x) 
		if hidden is None:
			hidden = self.init_hidden(gates.size()) 
			self.init_cell_weight(gates.size()) 

		if self.Wci is None: 
			self.init_cell_weight(gates.size()) 

		hx,cx = hidden
		# print("hidden", hidden.size())
		#print("gates in convLSTM", gates.size())
		gates = gates + self.h2h(hx) 
		#print('h2h gates', self.h2h(hx).size())
		ingate, cellgate, outgate, forgetgate = gates.chunk(4, 1) 
		cellgate = F.tanh(cellgate) 

		ingate = F.sigmoid(ingate + cx*self.Wci) 
		forgetgate = F.sigmoid(forgetgate + cx*self.Wcf)

		cy = (forgetgate*cx) + (ingate*cellgate) 

		outgate = F.sigmoid(outgate + cy*self.Wco) 
		hy = outgate * F.tanh(cy)
		'''
		print('ingate', ingate.size())
		print('outgate', outgate.size())
		print('cellgate', cellgate.size())
		print('forgetgate', forgetgate.size())
		
		print('hy', hy.size())
		print('cy', cy.size())
		'''
		return hy, cy
	

	def init_hidden(self, sz):

		sz = [sz[0], self.channel, sz[2], sz[3]]
		weight = next(self.parameters()).data 
		
		return (weight.new(*sz).zero_().requires_grad_(), 
			    weight.new(*sz).zero_().requires_grad_()) 

	def init_cell_weight(self, sz):
		# if self.layer == 0:
		# 	channel = self.channel
		# else:
		# 	channel = int(self.channel/2)


		sz = [sz[0], self.channel, sz[2], sz[3]]
		weight = next(self.parameters()).data 
		
		self.Wci = weight.new(*sz).zero_().requires_grad_() 
		self.Wcf = weight.new(*sz).zero_().requires_grad_() 
		self.Wco = weight.new(*sz).zero_().requires_grad_() 

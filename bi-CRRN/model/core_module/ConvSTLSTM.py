import torch
import torch.nn as nn 
import torch.nn.functional as F 

class ConvSTLSTM(nn.Module): 
	def __init__(self, channel, kernel, stride=1): 
		super(ConvSTLSTM, self).__init__() 
		self.channel = channel
		self.kernel = kernel

		if stride==1: 
			self.i2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2))
			self.i2m = nn.Conv2d(channel, 3*channel, kernel, 1, int((kernel-1)/2))
		else: 
			self.i2h = nn.Conv2d(channel, 4*channel, kernel, stride)
			self.i2m = nn.Conv2d(channel, 3*channel, kernel, stride)

		self.i2h.bias[3*channel:].data.fill_(1) 
		self.i2m.bias[2*channel:].data.fill_(1) 

		self.h2h = nn.Conv2d(channel, 4*channel, kernel, 1, int((kernel-1)/2)) 
		self.h2m = nn.Conv2d(channel, 3*channel, kernel, 1, int((kernel-1)/2)) 

		self.btneck = nn.Conv2d(2*channel, channel, 1, 1) 

		self.Wco = None 
		self.Wmo = None 


	def forward(self, input, hidden): 
		x, m = input 

		gates = self.i2h(x) 
		m_gates = self.i2m(x) 
	
		if hidden is None:
			hidden = self.init_hidden(gates.size()) 
			self.init_cell_weight(gates.size()) 

		if self.Wco is None: 
			self.init_cell_weight(gates.size()) 

		hx,cx = hidden 
		gates = gates + self.h2h(hx) 
		if m is not None: 
			m_gates = m_gates + self.h2m(m) 

		ig, cg, og, fg = gates.chunk(4,1)
		ig_, mg_, fg_ = m_gates.chunk(3,1) 

		cg = F.tanh(cg) 
		ig = F.sigmoid(ig) 
		fg = F.sigmoid(fg) 

		mg_ = F.tanh(mg_) 
		ig_ = F.sigmoid(ig_) 
		fg_ = F.sigmoid(fg_) 

		cg_next = (fg*cx) + (ig*cg) 
		
		if m is not None: 
			mg_next = (fg_*m) + (ig_*mg_) 
		else: 
			mg_next = ig_*mg_ 

		cg = F.sigmoid(og + self.Wco*cg_next + self.Wmo*mg_next) 

		cm = torch.cat([cg_next, mg_next], 1) 
		hy = og * F.tanh(self.btneck(cm)) 
		
		return hy, (cg_next, mg_next) 

	def init_hidden(self, sz): 
		sz = [sz[0], self.channel, sz[2], sz[3]] 
		weight = next(self.parameters()).data 
		
		return (weight.new(*sz).zero_().requires_grad_(), 
			    weight.new(*sz).zero_().requires_grad_()) 

	def init_cell_weight(self, sz): 
		sz = [sz[0], self.channel, sz[2], sz[3]] 
		weight = next(self.parameters()).data 
		
		self.Wco = weight.new(*sz).zero_().requires_grad_() 
		self.Wmo = weight.new(*sz).zero_().requires_grad_() 

import json
import os
import time
import argparse
import shutil
import functools
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function


#---> these functions are used for resetting the object attribute.
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

class AQuantizer(Function):    
    @staticmethod
    def forward(ctx, tensor, shift_v, N, a_sgn):#-->a_sgn:1X1 tensor
        #tensor1 = torch.unsqueeze(tensor,0).repeat(N,1,1,1,1)
        tensor1 = torch.cat(N*[torch.unsqueeze(tensor,0)]) #--> same as the above but the above giving issue with backward pass.
        #shift_v = shift_v.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # shape : NX1X1X1X1
        shift_v = shift_v.unsqueeze(1).unsqueeze(3).unsqueeze(3) # shape : NX1XCX1X1

        x = tensor1-shift_v
        ctx.save_for_backward(x, a_sgn)
        y = a_sgn[0]*torch.sign(x) + (1-a_sgn[0])*x
        return y
    
    @staticmethod
    def backward(ctx, grad_output):        
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        
        g_o = grad_output.clone() #this is a 5D tensor : Nxbatch-sizexchannelxRxC
        #print(ctx.saved_tensors)
        t, a_sgn = ctx.saved_tensors #--> otherwise reyurning a tuple
        
        t1 = t.clone()          #--->new
        if a_sgn[0]>0:
            t1 = -((1+a_sgn[0])/a_sgn[0])*torch.abs(t1)+2 #--->new
        else:
            t1=0*torch.abs(t1) #--> just make it zero.
        
        t[torch.abs(t)<=a_sgn[0]] = 1.0
        t[torch.abs(t)>a_sgn[0]] = 0.0 
        
        t = t1*t                #--->new
        
        grad_input=grad_output*(1-a_sgn[0]) + grad_output*(a_sgn[0])*t
        grad_input=(1/grad_output.size()[0])*torch.sum(grad_input, dim=0)
        
        #average the gradient along the batch-size dimension
        #grad_a = (1/t.size()[1])*(torch.sum(torch.sum(torch.sum(torch.sum(g_o,dim=1),dim=1),dim=1),dim=1))*-1.0
        grad_a = (1/t.size()[1])*(torch.sum(torch.sum(torch.sum(g_o,dim=1),dim=2),dim=2))*-1.0 #--> NXC shift_v parameter grad
        #N-element tensor returned.
        return grad_input , grad_a, None, None

#-----------------------> (added on 27=07-2020)
class ActQuantizer(nn.Module) :
    def __init__(self, *args, **kwargs):
        super(ActQuantizer, self).__init__()
        self.shift_init = kwargs['shift_init']
        self.N = kwargs['N']
        #self.a_sgn = kwargs['a_sgn']
        #self.shift_v = nn.Parameter(torch.from_numpy(np.array(self.shift_init)).float()) #initial clip_v value
        self.shift_v = nn.Parameter(self.shift_init.float()) #initial clip_v value
        #self.register_buffer('shift_v', torch.from_numpy(np.array(self.shift_init)).float())        
        
        #self.register_backward_hook(self.backward_hook) #---> This is not called when backward_hook() is not called.
    
    def forward(self, input, a_sgn):
        x = AQuantizer.apply(input,self.shift_v,self.N,a_sgn)#-->new addition
        return x

class WQuantizer(nn.Module):
    
    def __init__(self, *kargs, **kwargs):
        super(WQuantizer, self).__init__()
        self.M = kwargs['M']
        self.register_buffer('u', torch.tensor(np.zeros( (self.M,1,1,1,1) ) ) )
        for i in range(self.M):
            self.u[i,0,0,0,0] = -1+2*(i-1)/(self.M-1)        
        data = kwargs['data']
        
    def quantize(self, data):        
        data = torch.unsqueeze(data,0) 
        B_concat = torch.sign(data-torch.mean(data) + self.u*torch.std(data)).float() #-->new (all Bi's along 0 th dimension)       
        #calculate 'a'        
        # the .float() was added to ensure all operations in float() mode. Otherwise, it was giving error saying input is float and weight double()        
        W1 = torch.reshape(data,(-1,1))           #-->added .float()
        B1 = torch.reshape(B_concat,(self.M,-1)) #-->new
        B = torch.transpose(B1,0,1)              #-->new
        a = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(torch.transpose(B,0,1),B)),torch.transpose(B,0,1)),W1).float()
        
        return a,B_concat

class QConv2d(nn.Conv2d):
    
    def __init__(self, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QConv2d, self).__init__(*kargs, **kwargs)
        # ....................................................weight quantization
        self.weight.data = init_args['weight_data']
        if kwargs['bias'] == True:
            self.bias.data = init_args['bias_data']
        self.M = init_args['M']
        w_qargs = {'M':self.M}
        self.quantizer = WQuantizer (data = self.weight.data, **w_qargs)
        
        a_copy = np.zeros((self.M,1)) #--> new
        a_copy[0][0]=1.0 #--> new
        self.register_buffer('a', torch.tensor(a_copy)) #--> new
        #self.register_buffer('a', torch.tensor([[1],[0],[0]])) 
        
        qB_copy = torch.unsqueeze(self.weight.clone(),0) #--> new
        qB1_copy = qB_copy #--> new
        for i in range(self.M-1) : #--> new
            qB_copy = torch.cat((qB_copy,qB1_copy),0) #--> new
        self.qB = nn.Parameter(qB_copy) #--> new
        
        
        # .....................................................input quantization 
        self.N = init_args['N']
        
        if self.N > 0 :
            self.shift_v = init_args['shift_v']
            a_sgn = init_args['a_sgn'] #--> scalar value
            i_qargs = {'shift_init': self.shift_v,'N': self.N} #, 'a_sgn': torch.from_numpy(np.array([self.a_sgn]))}
            self.register_buffer('a_sgn',  torch.from_numpy(np.array([a_sgn]))) #--> new (1X1) tensor
            self.input_quantizer = ActQuantizer(**i_qargs) 
            self.b = (1/self.N)*torch.ones(self.N,1)
        
    #call it after loss.backward()
    #---> specifically added for DeepLabv3+ (as last layer of ResNet-18 is not used by the code, so no gradient propagation from there)#KB(added on 01-08-2020)
    def update_grads(self):
        if self.qB.grad is not None :    
            w_grad = 0.0
            for i in range(self.M):
                w_grad  += self.a[i][0]*self.qB.grad[i]
            self.weight.grad = w_grad
            
    def update_a_sgn(self, epoch):
        if self.N > 0:
            self.a_sgn[0] = 1#1 - math.exp(-1*epoch/10) #self.a_sgn[0] + 0.05
            if self.a_sgn[0] > 1.0:
                self.a_sgn[0] = 1.0
                
    def update_a_sgn_val(self, epoch):
        if self.N > 0:
            self.a_sgn[0] = 1#1 - math.exp(-1*epoch/10) #1.0
            if self.a_sgn[0] > 1.0:
                self.a_sgn[0] = 1.0
    
    def forward(self, input):       
 # ----------------------------------------------------------------------------N=3(number of bases for input activations.)  
        if self.N == 0 :    
            self.a.data, self.qB.data = self.quantizer.quantize(self.weight)
            out = 0.0
            for i in range(self.M):
                out  += self.a[i][0]*F.conv2d(input, self.qB[i], self.bias, self.stride, self.padding, self.dilation, self.groups)
        else :
            x = self.input_quantizer(input, self.a_sgn) # --> Input quantization
            self.a.data, self.qB.data = self.quantizer.quantize(self.weight)
            out = 0.0
            for j in range(self.N):
                out_temp = 0.0
                for i in range(self.M):
                    out_temp += self.a[i][0]*F.conv2d(x[j], self.qB[i], self.bias, self.stride, self.padding, self.dilation, self.groups)
                out += self.b[j][0]*out_temp
        return out

class PReLU(Function):    
    @staticmethod
    def forward(ctx, tensor, gamma, eta, beta): 
        print(tensor.shape)
        print(gamma.shape)
        x = tensor-gamma
        ctx.save_for_backward(x, beta)
        
        y = x.clone() #-->.clone() is necessary, otherwise on changing y, x also changes and we don't want that.
        z = x.clone()
        y[y<=0]=0
        z[z>0]=0
        
        #z = torch.tensor([0.0]).cuda()
        #y = torch.max(x,z)[0] + beta*torch.min(x,z)[0]
        y = y + beta*z
        y = y + eta
        return y

    @staticmethod
    def backward(ctx, grad_output):        
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        g_o = grad_output.clone()
        t,b = ctx.saved_tensors                                  
        x = t.clone() #-->.clone() is necessary, otherwise on changing t, x also changes and we don't want that.
        t[t<=0.0] = -1.0
        t[t>0.0] = 0.0 
        t = -1.0*t
        grad_b_i = g_o*(x*t)
        grad_g_i = g_o*(-t*b - (1.0 - t))  
                                  
        grad_input=g_o*(t*b + (1.0 - t))
                                  
        grad_g = (1/t.size()[0])*(torch.sum(torch.sum(torch.sum(grad_g_i,dim=0),dim=1),dim=1))
        grad_gamma = grad_g.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                                  
        grad_e = (1/t.size()[0])*(torch.sum(torch.sum(torch.sum(g_o,dim=0),dim=1),dim=1))
        grad_eta = grad_e.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                                  
        grad_b = (1/t.size()[0])*(torch.sum(torch.sum(torch.sum(grad_b_i,dim=0),dim=1),dim=1))
        grad_beta = grad_b.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        return grad_input , grad_gamma, grad_eta, grad_beta
                                                                            
#-----------------------> (added on 17-08-2020)
class PReLU_ActQuantizer(nn.ReLU) :
    def __init__(self, *args, **kwargs):
        super(PReLU_ActQuantizer, self).__init__()
        gamma = kwargs['gamma'] #this is a 1XC torch array.
        gamma = gamma.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        self.gamma = nn.Parameter(gamma.float()) #initial clip_v value
                                  
        eta = kwargs['eta'] #this is a 1XC torch array.
        eta = eta.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        self.eta = nn.Parameter(eta.float()) #initial clip_v value
                                  
        beta = kwargs['beta'] #this is a 1XC torch array.
        beta = beta.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        self.beta = nn.Parameter(beta.float()) #initial clip_v value                          
    
    def forward(self, input):
        x = PReLU.apply(input, self.gamma, self.eta, self.beta)#-->new addition
        return x
    
def update_model_grads(net):
    for n,m in net.named_modules():
        if isinstance(m, QConv2d) :#or isinstance(m, QLinear):
            m.update_grads()
            
def update_model_a_sgn(net, epoch):
    for n,m in net.named_modules():
        if isinstance(m, QConv2d) :#or isinstance(m, QLinear):
            m.update_a_sgn(epoch)
            a_sgn = m.a_sgn #-->
    return a_sgn #-->

def update_model_a_sgn_val(net, epoch):
    for n,m in net.named_modules():
        if isinstance(m, QConv2d) :#or isinstance(m, QLinear):
            m.update_a_sgn_val(epoch)
            a_sgn = m.a_sgn #-->
    return a_sgn #-->

def quantize_model(net,skip_list=['conv1']):
    n_channels = -1 #--->
    N_val = 8
    for n,m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            n_channels = m.weight.size()[0] #---->
            if n in skip_list: 
                #the first convlution layer remains as full-precision (first layer of ResNet-18 encoder) #it is called backbone.conv1 in DeepLabv3+ code.
                continue
            else :
                #layer_id = int(n.partition('.')[-1].partition('.')[0]) #090719, AB: layer number for the conv layer
                bias = False
                if m.bias is not None:
                    bias = True
                init_args = {'weight_data': m.weight.data,'bias_data': m.bias.data if bias else None, 'M':5, 'N':N_val, 'shift_v': torch.randn(N_val,m.weight.data.size()[1]*m.groups), 'a_sgn':1.0} #added the 'alpha' variable which will be initialized from previously learned values.
                conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias, 'dilation': m.dilation}
                conv = QConv2d(init_args = init_args, **conv_args)
                rsetattr(net,n, conv)
                print('CONV layer '+ n+ ' quantized using '+ 'ABC-Net method')

        elif isinstance(m, nn.ReLU):#---->
            i_qargs = {'gamma' : torch.randn(n_channels,), 'eta' : torch.randn(n_channels,), 'beta' : 0.25*torch.ones(n_channels,)}
            relu = PReLU_ActQuantizer(**i_qargs)
            rsetattr(net,n, relu)
            print('RELU layer '+ n+ ' replaced using '+ 'ReAct-Net method')
    return net
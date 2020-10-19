import torch
import numpy as np
# init tensor
mytensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device='cpu', requires_grad=False)
print(mytensor)
print(mytensor.dtype, mytensor.device, mytensor.shape, mytensor.requires_grad)

x = torch.empty(size=(3,3))
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3))
print(x)
x = torch.ones((3,3))
print(x)
x = torch.eye(3,3)
print(x)
x = torch.arange(start=3,end=8,step=1)
print(x)
x = torch.linspace(start=.01,end=1,steps=10)
print(x)

x = torch.empty(size=(1,5)).normal_(mean=0,std=1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# inti tensors to diff types and convert to diff types
x = torch.arange(4)
print(x)
print(x.bool())
print(x.short())
print(x.long())
print(x.half())
print(x.float())
print(x.double())

nparr = np.zeros((5,5))
x = torch.from_numpy(nparr)
print(x)
backnp = x.numpy()
print(backnp)

# Tensor math and comparison

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x,y,out=z1)
print(z1)
z2 = torch.add(x,y)
print(z2)
z = x+y
print(z)




z =  x-y

z = torch.true_divide(x,y)
print(z)

# inplace ops

t = torch.zeros(3)
t.add_(x)
print(t)
t+=x
print(t)


z = x.pow(2)
print(z)
z = x**2
print(z)


z = x>0
print(z)

#matrix mult#
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
print(x3)
x3 = x1.mm(x2)
print(x3)


# matrix exponent
me = torch.rand(5,5)
me = me.matrix_power(2)
print(me)

# element wise mult
z = x*y
print(z)

# dot porduct
z = torch.dot(x,y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch,n,m))
t2 = torch.rand((batch,m,p))
outbmm = torch.bmm(t1,t2)
print(outbmm)

# broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
x = x1 - x2
print(x)
z = x1 ** x2
print(z)


# others
sum_x = torch.sum(x,dim=1)
print(sum_x)
vals, indices  = torch.max(x,dim=0)
print(vals, indices )
vals,indices = torch.min(x,dim=0)
print(vals, indices )
print(torch.abs(x))
print(torch.argmax(x,dim=0))
print(torch.argmin(x,dim=0))
mean_x = torch.mean(x.float(),dim=0)
print(mean_x)

print(torch.eq(x,x))

print(torch.sort(y,dim=0,descending=False))

z= torch.clamp(x,min=0,max=1)
print(z)

x= torch.tensor([1,0,1,1,1],dtype=torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)

# indexing in tensor

batch_size = 10
features = 25
x = torch.rand(batch_size,features)
print(x[0].shape)
print(x[:,0].shape)
print(x[2,0:10])
x[0,0] = 100

# fancy indexing
x = torch.arange(10)
indices = [2,5,8]

print(x[indices])

x= torch.rand((5,5))
rows = torch.tensor([1,3])
cols = torch.tensor([4,3])
print(x[rows,cols])


# advance indexing

x = torch.arange(10)
print(x[(x<2) | (x>8)])

print(x[x.remainder(2) == 0])


# more 
print(torch.where(x>5,x,x*2))
print(torch.tensor([0,0,1,1,2,2]).unique())
print(x.ndimension())
print(x.numel())



# reshaping

x = torch.arange(9)
x_33 = x.view(3,3) # needs contiguous memory
print(x_33)
print(x.reshape(3,3)) # safer


x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0))
print(torch.cat((x1,x2), dim=1))


z = x1.view(-1)
print(z)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

z = x.permute(0,2,1) # transpose batch
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0))
print(x.unsqueeze(1))
print(x.unsqueeze(0).unsqueeze(1))



























from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import generalQ as genq
import discrete_rnn as drnn
import matplotlib.pyplot as plt
import pickle
import math
import collections

from sympy.solvers import solve , nsolve
from sympy import Symbol
from sympy import sin, limit

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def examine(results,steps=100,test=0):
    layers = []
    states = []
    for (k1,k2), val in results.items():
        if k1 not in layers:
            layers.append(k1)
        if k2 not in states:
            states.append(k2)
    layers.sort()
    states.sort()
    res=np.zeros([len(layers),len(states)])
    for (k1,k2), val in results.items():
        i = layers.index(k1)
        j = states.index(k2)
        res[i][j] = val[steps][test]
    return res, layers, states

def showgrid(name="grid_results",steps=100,test=0,fig = plt.figure(), subplot=111):
    results, layers, states = examine(load_obj(name),steps,test)
    ax = fig.add_subplot(subplot)
    cax = ax.matshow(results)
    plt.gcf().set_size_inches(5, 5)
    ##fig.colorbar(cax)
    strlist = [l.__str__() for l in layers]
    ax.set_yticklabels(strlist)
    strlist = [l.__str__() for l in states]
    ax.set_xticklabels(strlist)
    ax.set_yticks( range(len(layers)), minor=False )
    ax.set_xticks( range(len(states)), minor=False )
    plt.xlabel("States")
    plt.ylabel("Layers")
    return fig

def showContour(name="grid_results",steps=100,test=0,fig = plt.figure(), subplot=111, levels = 7, scan = 0):
    results, layers, states = examine(load_obj(name),steps,test)

    l = len(layers)
    s = len(states)
    S, L = np.meshgrid(states,layers)
    ax = fig.add_subplot(subplot)
    plt.gcf().set_size_inches(5, 5)

    ax.set_yscale("log") 
    cmap=plt.get_cmap("hot")
    CS = ax.contourf(S, L, results,levels, colors = ['black','darkred','firebrick','red'])
    ##CS.cmap.set_over('red')
    ##CS.cmap.set_under('black')
    plt.xlabel("# States", fontsize=18)
    if (subplot%10==1):
        plt.ylabel("Layers", fontsize=18)
    ##ax.set_xticks( range(len(states)), minor=False )
    DS = np.asarray([genq.DepthScale(s) for s in states])
    ax.set_ylim((layers[0],layers[-1]))
    strlist = [l.__str__() for l in states]
    ##ax.set_xticklabels(strlist)
    ##ax.set_xticks( range(s), minor=False )
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        
    if int(subplot/10)>11:
        plt.title("%d Training Steps" % steps)
        
    if (scan==0):   
        midstate = states[int(len(states)/2)]
        idx = int(len(states)/2)
        if subplot==111:
            fig.colorbar(CS)
        ##if subplot%10==2:
        ##    fig.colorbar(CS)
        for a,c in [(4,'springgreen'),(6,'y'),(9,'w')]:
            plt.semilogy(states,a*DS,lineStyle='--',color=c)
            point = (midstate,a*DS[idx])
            while (idx> 0 and a*DS[idx] > 0.8*layers[-1]):
                idx= idx-1
                midstate = states[idx]
            
            plt.annotate("%d$\\xi$" % a , (midstate,a*DS[idx]), color=c ,fontSize=18, 
                         textcoords="offset points",xytext=(-7,7), ha='center')
            idx= idx-1
            midstate = states[idx]

    else:    

            
        results_misc, layers, sigmaWs = examine(load_obj(name),steps=steps,test=test)
        midstate = sigmaWs[int(len(sigmaWs)*3/4)]
        idx = int(len(sigmaWs)*3/4)
        colors = ['k','k','r','w']

        offsets, scale, W, Qu = genq.optimal_spacing(scan)

        Chis_for_DS = np.asarray([genq.Estimate(drnn.RnnArgs(SigmaW = w, SigmaU=0.0 ,SigmaB=0.001) 
                                                ,offsets,scale,steps=100)   for w in sigmaWs])
        
        DS = np.asarray([-1.0/np.log(chi) for chi in Chis_for_DS])
        plt.xlabel("$\sigma_w$")
        ##plt.title("Mnist Training- %d States Actvation" % scan)
        
        perct = 0.4
        for a,c in [(4,'c'),(6,'y'),(9,'w')]:
            plt.semilogy(sigmaWs,a*DS,lineStyle='--',color=c)
            point = (midstate,a*DS[idx])
            while (idx> 0 and a*DS[idx] > perct*layers[-1]):
                idx= idx-1
                midstate = sigmaWs[idx]
            perct+=0.2
            plt.annotate("%d$\\xi$" % a , (midstate,a*DS[idx]), color=c ,fontSize=16, 
                         textcoords="offset points",xytext=(0,10), ha='center')

    

def heaviside(x):
    res = 1/2+torch.sign(x)/2
    return res

class QuantF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, offsets=torch.Tensor([0.0]), count = torch.Tensor([1]) ,scale=torch.Tensor([1.0]), slope = torch.Tensor([1.0])):
        if input.is_cuda:
            output = (-scale*torch.ones(input.size()).cuda())
            factor = ((2*scale)/count).cuda()
        else:
            output = -scale*torch.ones(input.size())
            factor = ((2*scale)/count)

        for offset in offsets:
            output += factor*heaviside(input-offset)
        ctx.save_for_backward(slope)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        slope, = ctx.saved_tensors
        return (slope*grad_output), None, None, None, None

class Quant(nn.Module):
    __constants__ = ['offsets','scale']
    def __init__(self,offsets=torch.Tensor([0.0]), count=torch.Tensor([1]) 
                 ,scale=torch.Tensor([1.0]), ste_threshold=1.0, slope =torch.Tensor([1.0])):
        super(Quant, self).__init__()
        self.offsets= offsets
        self.scale= scale
        self.count = count
        self.border = ste_threshold
        self.slope = slope
    def forward(self,x):
        return QuantF.apply(x.clamp(min=-self.border, max = self.border),self.offsets,self.count,self.scale,self.slope)

def gen_quant(states,scale, lazy_grad = True):
    offsets = torch.linspace(-scale,scale,states-1).cuda()
    count = torch.Tensor([offsets.numel()]).cuda()
    scale = torch.Tensor([scale]).cuda()
    if lazy_grad:
        return Quant( offsets ,count,scale)
    else:
        return Quant2( offsets ,count,scale)

def opt_quant(states):
    offsets, scale , W, Qu = genq.optimal_spacing(states)
    offsets = offsets.cuda()
    count = torch.Tensor([offsets.numel()]).cuda()
    scale = torch.Tensor([scale]).cuda()
    ##thresh = float(scale.cpu().numpy())
    thresh=1.0
    ##print(offsets)
    slope = genq.optimize_ste(Qu,W, thresh, samples=1000)
    slope = torch.Tensor([1.0/slope]).cuda()
    ##print("Thresh: %.05f, Slope: %.05f" % (thresh,slope))
    return Quant(offsets ,count,scale, thresh,slope), W

def opt_quantW(states, w):
    offsets, scale , W, Qu = genq.optimal_spacing(states)
    Qu = Qu*((w/W)**2)
    offsets = offsets.cuda()
    count = torch.Tensor([offsets.numel()]).cuda()
    scale = torch.Tensor([scale]).cuda()
    ##thresh = float(scale.cpu().numpy())
    thresh=1.0
    ##print(offsets)
    slope = genq.optimize_ste(Qu,w, thresh, samples=1000)
    slope = torch.Tensor([1.0/slope]).cuda()
    ##print("Thresh: %.05f, Slope: %.05f" % (thresh,slope))
    return Quant(offsets ,count,scale, thresh,slope), w
  
  
def opt_quant2(states):
    offsets, scale , W = genq.optimal_spacing2(states)
    offsets = offsets.cuda()
    count = torch.Tensor([offsets.numel()]).cuda()
    scale = torch.Tensor([scale]).cuda()
    thresh = float(scale.cpu().numpy())
    slope = genq.optimize_ste(1.0,W, thresh, samples=1000)
    slope = torch.Tensor([1.0/slope]).cuda()
    print("Thresh: %.05f, Slope: %.05f" % (thresh,slope))
    return Quant(offsets ,count,scale, thresh,slope), W
      
def init_layer(op, sigma_w,sigma_b,standard):
    if standard:
        stdv = 1. / math.sqrt(op.weight.size(1)) 
        op.weight.data.uniform_(-stdv, stdv) 
        if op.bias is not None: 
            op.bias.data.uniform_(-stdv, stdv)
            op.bias.data.fill_(0.0)
    else:
        nn.init.normal_(op.weight,0.0,sigma_w)
        if (sigma_b>0):
            nn.init.normal_(op.bias,0.0,sigma_b)
        else:
            op.bias.data.fill_(0.0)

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
   
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size,-1) 
          
def gen_model(act, num_layers =5, N =28*28,W=1.3,B=0.0, standard=False):
    ##super(Net, self).__init__()
    dict = collections.OrderedDict()
    L = num_layers
    M = 28*28
    sigma_w = W/np.sqrt(N)
    sigma_b = B/np.sqrt(N)
    flat = Flatten()
    dict["flat"] = flat
    fc= nn.Linear(M, N)
    init_layer(fc, W/np.sqrt(M),B/np.sqrt(M),standard)
    dict["fc0"] = fc
    dict["act0"] = act
        
    for l in range(1,L-1):
        fc= nn.Linear(N, N)
        init_layer(fc,sigma_w,sigma_b,standard)
        dict["fc%d" % l] = fc
        dict["act%d" % l] = act
    l = L-1
    fc= nn.Linear(N, 10)
    init_layer(fc,sigma_w,sigma_b,standard)
    dict["fc%d" % l] = fc
    dict["act%d" % l] = act
    dict["final"] = nn.LogSoftmax(dim=1)
    model = nn.Sequential(dict)
    return model
        
    ##def forward(self, x):
    ##    for op in self.dict.values():
    ##        print(op)
    ##        x = op(x)
    ##    print("hi!")
    ##    return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    lossVec=[]
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1,1,784).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx>0:
            lossVec.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return lossVec

def train2(args, model, device, train_loader, train_test_loader , test_loader, optimizer, counter, train_log, reports):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1,1,784).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        counter = counter + 1
        if counter in reports:
            train_acc = test(args,model,device, train_test_loader)
            test_acc = test(args,model,device, test_loader)
            train_log[counter] = (train_acc,test_acc)
    return counter
            
def getTestLoss(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = (100. * correct / len(test_loader.dataset))
    return test_loss
    
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = (100. * correct / len(test_loader.dataset))
    return test_acc

  
    ##print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    ##    test_loss, correct, len(test_loader.dataset),
    ##    100. * correct / len(test_loader.dataset)))
    

def set_seed(seed, fully_deterministic=False):
   np.random.seed(seed)
   ##random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
       if fully_deterministic:
           torch.backends.cudnn.deterministic = True


def main(cmd=None, act = nn.ReLU(), sigmaW =1.0, reports=[100]):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
##    parser.add_argument('--Weights','-W', type=float, default=1.0, dest='sigma_w',
##                        help='Parameter to initialize weights with')
    
    parser.add_argument('--Biases','-B', type=float, default=0.0, dest='sigma_b',
                        help='Parameter to initialize biases with')
    
    parser.add_argument('--Layers','-L', type=int, default=5, dest='num_layers',
                        help='Number of layers (Minimum 2)')
    
    parser.add_argument('-N','--LayerSize', dest='size_of_layer', default=784, \
                        type=int, help='Size of hidden layers (height/width) (default: 784 )')
    
    parser.add_argument('--continious', action='store_true', default=False,
                        help='continious run')
    
    parser.add_argument('--standard', action='store_true', default=False,
                        help='standard initialization')
    
    
    if cmd is not None:
        args = parser.parse_args(cmd.split(" "))
    else:
        args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    ##set_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    train_loader_check = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    
    
    model = gen_model(act, args.num_layers, args.size_of_layer, sigmaW, args.sigma_b, args.standard).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    if (args.continious):
        train_accuracy= []
        test_accuracy = []
        lossVec = []
        testLoss = []
        for epoch in range(1, args.epochs + 1):
            start=time.time()
            tmpLossVec = train(args, model, device, train_loader, optimizer, epoch)
            lossVec.append(tmpLossVec)
            train_acc = test(args, model, device, train_loader_check)
            train_accuracy.append(train_acc)
            
            test_acc = test(args, model, device, test_loader)
            test_accuracy.append(test_acc)
            test_loss = getTestLoss(args, model, device, train_loader_check)
            testLoss.append(test_loss)
            duration  = time.time() -start
            print("train accuracy: %.02f, test accuracy: %.02f, time: %.02f" % (train_acc, test_acc, duration))
        return lossVec, testLoss
    else:
        train_log= {}
        counter = 0
        if counter in reports:
            train_acc = test(args,model,device, train_loader_check)
            test_acc = test(args,model,device, test_loader)
            train_log[counter] = (train_acc,test_acc)
        
        while counter < max(reports):
            start=time.time()
            counter = train2(args, model, device, train_loader, train_loader_check, test_loader, optimizer, counter, train_log,reports)
            duration  = time.time() - start
            #if len(train_log.values())>0:
            #    cur_step = max(train_log.values())
            #    (train_acc, test_acc) = train_log[cur_step]
            #    print("train accuracy: %.02f, test accuracy: %.02f, Duration: %.02f" % ( train_acc, test_acc,duration))
            ##else:
            ##print("Epoch Completed. Duration: %.02f" % (duration))
            
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
    return train_log

def pick_lr2(L):
  if L < 100:
    lr = 0.001
  elif L < 180:
    lr = 0.0005
  else:
    lr = 0.0004
  return lr

def pick_lr(L):
  if L < 100:
    lr = 0.001
  else:
    lr = 0.0005
  return lr

def run_part_lin(name, start, end, gap , allstates ,deviceId, num_devices, reports = [100,200,400,800,1600], N = 2048):
  x = np.linspace(start,end,int((end+gap-start)/gap))
  x = np.round(x)
  x = np.unique(x)
  x = x[deviceId::num_devices]
  allLayers = x.astype(np.int)
  print("Iterating over layers: %s" % allLayers)
  part_name = "%s_%d" % (name, deviceId)
  run_grid(part_name, allLayers, allstates ,deviceId , reports, N)

def run_part(name, start, end, sample_layers , allstates ,deviceId, num_devices, reports = [100,200,400,800,1600], N = 2048):
  x = np.linspace(np.log(start),np.log(end),sample_layers)
  x = np.exp(x)
  x = np.round(x)
  x = np.unique(x)
  x = x[deviceId::num_devices]
  allLayers = x.astype(np.int)
  print("Iterating over layers: %s" % allLayers)
  part_name = "%s_%d" % (name, deviceId)
  run_grid(part_name, allLayers, allstates ,deviceId , reports, N)

def run_grid(name, allLayers, allstates ,deviceId , reports = [100,200,400,800,1600], N = 2048):  
  torch.cuda.set_device(deviceId)
  results={}
  for L in allLayers:
    for states in allstates:
      torch.cuda.empty_cache()
      offsets, scale, W, Qu  = genq.optimal_spacing(states)
      slope = genq.optimize_ste(1.0,W, scale, samples=1000)
      chi = genq.Estimate(drnn.RnnArgs(SigmaW = W, SigmaU=0.0 ,SigmaB=0.001) ,offsets,scale,steps=100)
      mjj = genq.Mjj(Qu, W, scale, 1/slope)        
      start=time.time()
      print("%d states, %d layers, chi: %.05f, Mjj: %.05f" % (states,L,chi, mjj))
      quant, W = opt_quant(states)
      lr = pick_lr(L)
      res = main('--batch-size 32 --epochs 5 --lr %f -N %d -L %d' % (lr ,N, L) , act = quant, sigmaW = W,reports=reports)
      duration = time.time()-start
      print("Duration: %.02f" % duration)
      results[(L,states)] = res
      save_obj(results,name)

def run_partW(name, start, end, gap , states ,deviceId, num_devices, reports = [100,200,400,800,1600], N = 2048, adapt_slope=False):
  x = np.linspace(start,end,int((end+gap-start)/gap))
  x = np.round(x)
  x = np.unique(x)
  x = x[deviceId::num_devices]
  allLayers = x.astype(np.int)
  print("Iterating over layers: %s" % allLayers)
  if (adapt_slope):
    part_name = "%s_W_AdaptiveSlope_%d" % (name, deviceId)
    run_gridW2(part_name, allLayers, states ,deviceId , reports, N)
  else:
    part_name = "%s_W_%d" % (name, deviceId)
    run_gridW(part_name, allLayers, states ,deviceId , reports, N)

  
  
def run_gridW(name, allLayers, states ,deviceId , reports = [100,200,400,800,1600], N = 2048):  
  torch.cuda.set_device(deviceId)
  results={}
  for L in allLayers:
    torch.cuda.empty_cache()
    offsets, scale, W, Qu  = genq.optimal_spacing(states)
    Ws = torch.linspace(W*0.5,W*1.5,16)
    for w in Ws:
      slope = genq.optimize_ste(1.0,w, scale, samples=1000)
      chi = genq.Estimate(drnn.RnnArgs(SigmaW = w, SigmaU=0.0 ,SigmaB=0.001) ,offsets,scale,steps=100)
      start=time.time()
      print("%d states, %d layers, chi: %.05f, W = %.02f" % (states,L,chi,w))
      quant, W = opt_quant(states)
      lr = pick_lr(L)
      res = main('--batch-size 32 --epochs 5 --lr %f -N %d -L %d' % (lr ,N, L) , act = quant, sigmaW = w,reports=reports)
      duration = time.time()-start
      print("Duration: %.02f" % duration)
      results[(L,w)] = res
      save_obj(results,name)
      
  
def run_gridW2(name, allLayers, states ,deviceId , reports = [100,200,400,800,1600], N = 2048):  
  torch.cuda.set_device(deviceId)
  results={}
  for L in allLayers:
    torch.cuda.empty_cache()
    offsets, scale, W, Qu  = genq.optimal_spacing(states)
    Ws = torch.linspace(W*0.5,W*1.5,16)
    for w in Ws:
      start=time.time()
      print("%d states, %d layers, W = %.02f" % (states,L,w))
      quant, W = opt_quantW(states,w)
      lr = pick_lr(L)
      res = main('--batch-size 32 --epochs 5 --lr %f -N %d -L %d' % (lr ,N, L) , act = quant, sigmaW = w,reports=reports)
      duration = time.time()-start
      print("Duration: %.02f" % duration)
      results[(L,w)] = res
      save_obj(results,name)
      

def mnisthon(states, device, L=10, N=2048, seeds = 10, standard = False, maximum = 10000, lr = None):
    torch.cuda.set_device(device)
    quant, W = opt_quant(states)
    if lr is None:
        lr = pick_lr(L)
    all_results={}
    
    name = "test_acc_%d_states_%d_layers" % (states,L)
    if (standard):
      name = name + "_std"
    print(name)
    for seed in range(seeds):
        
        torch.cuda.empty_cache()
        if (standard):
            res = main('--batch-size 32 --epochs 10 --seed %d --lr %f -N %d -L %d --standard --test-batch-size 100' % (seed,lr ,N, L) , act = quant, sigmaW = W,reports=range(0,maximum,100))
        else:
            res = main('--batch-size 32 --epochs 10 --seed %d --lr %f -N %d -L %d --test-batch-size 100' % (seed,lr ,N, L) , act = quant, sigmaW = W,reports=range(0,maximum,100))
        save_obj(res, "%s_%d" % (name,seed))
        all_results[seed] = res
        print("seed %d completed" % seed)
        save_obj(all_results, name)
      
      
if __name__ == '__main__':
    mnist.main('--batch-size 32 --epochs 10 -L 10',act = nn.ReLU())
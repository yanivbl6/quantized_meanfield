import numpy as np
import argparse
import re
import math
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt
import discrete_rnn as drnn


class Subplot:
  def __init__(self,x,y,autoC=True):
    self.x = int(x)
    self.y = int(y)
    self.n = 0
    fig, axs = plt.subplots(self.y, self.x, figsize=( 16, 16),subplot_kw={"projection": "3d"})
    

    self.fig = fig
    self.axs = axs
    if (autoC):
      fig2 , axs2 = plt.subplots(1, self.x, figsize=( 8, 4))
      self.axs2 = axs2
      self.fig2 = fig2

  def new(self):
    self.n = self.n%self.y+1
  def pos(self):
    return int(100*self.y+10*self.x+self.n)
  
  def next(self):
    ny,nx = divmod (self.n,self.y)
    self.n = self.n+self.y
    if (self.x==1):
      return self.axs[ny]
    elif (self.y==1):
      return self.axs[nx]
    else:
      return self.axs[ny][nx]
  def set_col_title(self,x,s,xmax=100):
    self.axs[0][x].set_title(s,fontsize=24)
    self.axs2[x].set_title(s,fontsize=24)


  def set_line_title(self,y,s):
    self.axs[y][0].set_zlabel(s,fontsize=20)
    
  def add(self,C):
    ny,nx = divmod (self.n,self.y)
    if (self.x==1):
      ax = self.axs2
    else:
      ax = self.axs2[nx]
    
    ax.matshow(C)
    xmax = len(C[0])
    ax.set_xlabel(r'$\Delta\theta$',fontsize=16)
    if (nx==0):
      ax.set_ylabel("Layers",fontsize=16)

    ax.xaxis.set_ticks(np.arange(0, 6*xmax/5, xmax/4))
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(scale=xmax)))
    
    ax.set_aspect('auto')
        
    
    
def Entropy(L, bins=10000):
  P=torch.histc(L.cpu(),bins=bins)/(L.numel())
  lP = torch.log(P)
  H = P*lP
  H[H != H] = 0
  return -torch.sum(H)

def PCA(Z,n=3):
  zm = torch.mean(Z,1,keepdim=True)
  Z2 = Z - zm
  zm = torch.mean(Z2,0,keepdim=True)
  zc = torch.matmul(torch.transpose(zm, 0, -1),zm)
  e, v = torch.symeig(zc, eigenvectors=True)
  r=range(len(e)-1,len(e)-n-1,-1)
  v = v[r]
  return torch.matmul(Z ,torch.transpose(v, 0, -1))

def gen_colors(n):
  theta= torch.linspace(0,2*np.pi,n).reshape([n,1])
  col = [(0.5+float(np.cos(theta[i])/2),0.0,0.5+float(np.sin(theta[i]))/2) for i in range(n)]
  return col

def display(Z,col=None,gradient=False,title="PCA display",subplot=None):
  x = Z[:,0].cpu().numpy()
  y = Z[:,1].cpu().numpy()
  z = Z[:,2].cpu().numpy()
  n = len(x)
  ##print(n)

  if (col ==None):
      col = gen_colors(n)

  if (subplot!=None):
    ##pos=subplot.pos()
    ax = subplot.next()
  else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=20., azim=0)
  
  ##ax.plot(x,y,z,c=(0.15,0.25,0.25),label="input",linewidth=3.0)
  #ax.set_prop_cycle(color=col)
  ##ax.axis("off")
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])
  ax.zaxis.set_ticks([])

  if gradient:
      for k in range(n):
          ax.scatter(x[k],y[k],z[k],color=col[k],linewidth=3.0)
  else:
      ax.plot(x,y,z,color=col[0],linewidth=3.0)
  ##plt.title(title)
  ##plt.axis("off")
  if (subplot==None):
    plt.show()
    
def pca_display(Z,col=None,gradient=True,title="PCA display",subplot=None):
  display(PCA(Z),col,gradient,title,subplot)
 

def ff_layer(Z,args):
  N = args.output_size
  iW = tdist.Normal(torch.tensor([0.0]), torch.tensor([args.SigmaW/np.sqrt(N)]))
  iB = tdist.Normal(torch.tensor([args.Mu]), torch.tensor([args.SigmaB]))
  W = iW.sample([N,N]).reshape([N,N]).cuda()
  B = iB.sample([N]).reshape([1,N]).cuda()
  L = Z.mm(W)+B
  return args.act(L)

def calcQ(L,args):
  torch.mean(L*L,0)
  return args.act(L)

def autoC(L,C,d):
  CMat=L.mm(L.transpose(0,-1))
  K = len(CMat)
  for k in np.roll(range(K),int(K/2)):
      C[d][k]= CMat.trace()/K
      CMat=CMat.roll(1,0)
  return C
      
def ff_sim(args,display_every=5,subplot=None,autoCorrelations=True,normalize=True):
  D = args.sim_time
  doshow = False
  if (subplot==None):
    doshow=True
    subplot= Subplot(1,int(D/display_every)+1)
  
  n = args.num_samples
  N = args.output_size
  iV= tdist.Normal(torch.tensor([0.0]), torch.tensor([1/np.sqrt(N)]))
  
  v1 = iV.sample([N]).reshape([N])
  v2 = iV.sample([N]).reshape([N])
  v2 = v2 - drnn.angle(v1,v2)*v1
  
  if (normalize):
    q= drnn.get_ff_qstar(args)
  else:
    q=1
  factor1 = torch.sqrt(torch.mean(v1*v1)/q)
  factor2 = torch.sqrt(torch.mean(v2*v2)/q)
  v1=v1/factor1
  v2=v2/factor2
  v1 = v1.reshape([1,N])
  v2 = v2.reshape([1,N])
  
  theta = torch.linspace(0,2*np.pi,n).reshape([n,1])
  Z = v1*torch.cos(theta)+v2*torch.sin(theta)
  L = Z.cuda()
  
  
  C=None
  if (autoCorrelations):
    C = torch.zeros([D,n])
  for d in range(D): ## range(args.sim_time):
    if (d % display_every==0):
      s = "Layer %d" % d
      pca_display(L,gradient=True,title=s,subplot=subplot)
    if (autoCorrelations):
      C=autoC(L,C,d)
    L = ff_layer(L,args)
    
  s = "Layer %d" % D
  pca_display(L,gradient=True,title=s,subplot=subplot)

  if (autoCorrelations):
    C=autoC(L,C,d)
    subplot.add(C)
  if (doshow):
    plt.show()
  return


def compare(args,display_every=5,autoCorrelations=True,name=None):
  D = args[0].sim_time
  subplot= Subplot(len(args),int(D/display_every)+1,autoCorrelations)

  for i in range(int(D/display_every)+1):
    if (i==0):
      s="Input"
    else:
      s="Layer %d" % int(i*display_every)
    subplot.set_line_title(i,s)

  for i in range(len(args)):
    subplot.set_col_title(i,args[i].label(),args[i].num_samples)

  for arg in args:
    arg.sim_time = D
    ff_sim(arg,display_every,subplot,autoCorrelations=autoCorrelations)
    subplot.new()
    ##t = torch.linspace(0,3,10)
    ##print("activation of %s:\n %s" % (t, arg.act(t)))

    
  if name is not None:
    subplot.fig2.tight_layout()
    subplot.fig2.savefig("%s.pdf" % name, bbox_inches='tight')
    
  plt.show()

  

def multiple_formatter(denominator=2, number=np.pi, latex='\pi', scale = 100):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        x = x*2*(np.pi)/scale-np.pi
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter
  
class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
  
  
if __name__ == "__main__":
  print("visualizaiton.py")
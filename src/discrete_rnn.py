import numpy as np
import argparse
import re
import math
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt

import activations as actv

from sympy.solvers import solve , nsolve
from sympy import Symbol
from sympy import sin, limit
from scipy import stats

class RnnArgs:
  def __init__(self, activation = (lambda x : torch.sign(x)) , num_samples=1024, input_size=1024, sim_time=20,tied_weights=False, SigmaW=1.0, SigmaU=1.0, \
               SigmaB=1.0, mu = 0.0, C0=0.5, Q0=1.0, LambdaIn= None, plot=False, desc=None, offsets=[]):
    self.input_size = input_size
    self.output_size = input_size
    self.num_samples = num_samples
    self.sim_time = sim_time
    self.tied_weights = tied_weights
    self.Mu = float(mu)
    self.SigmaW = float(SigmaW)
    self.SigmaU = float(SigmaU)
    self.SigmaB = float(SigmaB)
    self.Q0 = float(Q0)
    self.C0 = float(C0)
    if (LambdaIn==None):
      LambdaIn= (lambda x : seq_input_step( x, T=sim_time))
    self.In = LambdaIn(self)
    self.plot = plot
    self.act = activation
    self.offsets = offsets
    if (desc==None):
      self.desc = "Mu=%.2f, W=%.2f, U=%.2f, B=%.2f" % (self.Mu,self.SigmaW, self.SigmaU,self.SigmaB)
    else:
      self.desc = desc
    pass

  def label(self):
    return self.desc
  
  def present(self,rng=5.0):
    x = torch.linspace(-rng,rng,1000)
    y=self.act(x)
    plt.plot(x.numpy(),y.numpy())
    plt.plot(x.numpy(),x.numpy())
    plt.grid()
    plt.show()

def parse():
  parser = argparse.ArgumentParser(description='Calculate Covariance Matrix and match with Simulation')

  parser.add_argument('-n','--num', dest='num_samples', default=1024, \
                      type=int, help='Number of samples to simulate or use for calculation (default: 1024)')

  parser.add_argument('-M','--input_size', dest='input_size', default=1024, \
                      type=int, help='Length of inputs (default: 1024)')

  parser.add_argument('-N','--output_size', dest='output_size', default=1024, \
                      type=int, help='Length of output (default: 1024)')

  parser.add_argument('-T','--time', dest='sim_time', default=100, \
                      type=int, help='Length of time to simulate (default: 100)')

  parser.add_argument('-t','--tied_weights', dest='tied_weights', default=False,
                      action='store_true', help='Use tied weights (Simulation only, default: untied)')

  parser.add_argument('-s','--sim', dest='simulate', default=False,
                      action='store_true', help='Run calculation')

  parser.add_argument('-c','--calc', dest='calculate', default=False,
                      action='store_true', help='Run simulation')

  parser.add_argument('-p','--params', dest='params', default='0,1,1,1', \
                      type=str, help="mu,Sigma_w,Sigma_u,Sigma_b default: 0,1,1,1")

  parser.add_argument('-I','--initial', dest='initial', default='1,1', \
                      type=str, help="format: Q[0],C[0] default: 1,1")

  inputs= {
    'Step' : (lambda x : seq_input_step( x, T=10)),
    'Short Step' : (lambda x : seq_input_step( x, T=2 )),
    'Long Step' : (lambda x : seq_input_step( x, T=50)),
    'Same' : (lambda x : seq_input_step( x, T=0)),
    'Reverse' : (lambda x : seq_input_reverse_step( x, T=10)),

  }

  parser.add_argument('-x','--input_sequence', dest='input_seq', default='Step', \
                      type=str, help="Options: %s (default: 'Step')" % [key for (key,val) in inputs.items()])

  args = parser.parse_args()
  
  if (args):
    args.input_size = args.output_size
    if (args.input_size != args.output_size):
      print("Input dimnesion must be equal to ouput dimension")
      return None
    params = re.split(',',args.params)
    args.Mu = float(params[0])
    args.SigmaW = float(params[1])
    args.SigmaU = float(params[2])
    args.SigmaB = float(params[3])
    args.params=None
  
  
    initials = re.split(',',args.initial)
    args.Q0 = float(initials[0])
    args.C0 = float(initials[1])
    args.initial = None
    args.In = inputs[args.input_seq](args)
  
  return args



def normalize(v):
  return (v / v.norm())
  pass

def angle(a,b):
  return torch.sum(a*b)/(a.norm()*b.norm())
  pass

def Edot(a,b):
  return torch.sum(a*b)
  pass

class seq_input:
  def gen(self,t):
    pass

  def normGen(self,t):
    return normalize(self.gen(t))
##torch.Tensor.numpy()
  def cov(self,t, N=100):
    cov12 = 0
    cov11 = 0
    for n in range(N):
      cov12 += angle(self.gen(t),self.gen(t)) ##  self.normGen(t).mm(self.normGen(t))
      a = self.normGen(t)
      cov11 += np.dot(np.transpose(a),a)
    cov11 = cov11 / N
    cov12 = cov12 / N

    return np.matrix([[cov11, cov12],[cov12,cov11]])
    pass

  def cosine(self,t, n=100): ##off diagonal
    cov12 = 0
    for n in range(n):
      cov12 += angle(self.gen(t),self.gen(t)) ##  self.normGen(t).mm(self.normGen(t))
    cov12 = cov12 / n
    if (cov12 > 1.00):
      cov12 = 0.99
    return cov12
    pass

class seq_input_step(seq_input): 
  def __init__(self, args, R=1.0, T=10, sigma = 1.0):
    self.R = R
    self.start = T
    self.sigma = sigma
    size= args.sim_time - T
    self.input_size= args.input_size
    self.values = np.random.normal(0, sigma, (size, self.input_size))
    self.dist = tdist.Normal(torch.tensor([0.0]), torch.tensor([sigma]))
    self.values = self.dist.sample([size, self.input_size])
  
  def gen(self,t):
    if (t < self.start):
      return self.dist.sample([self.input_size])
    else:
      return self.values[t - self.start]

class seq_input_reverse_step(seq_input): 
  def __init__(self, args, R=1.0, T=10, sigma = 1.0):
    self.R = R
    self.start = T
    self.sigma = sigma
    size= T
    self.input_size= args.input_size
    self.values = np.random.normal(0, sigma, (size, self.input_size))
    self.dist = tdist.Normal(torch.tensor([0.0]), torch.tensor([sigma]))
    self.values = self.dist.sample([size, self.input_size])
  
  def gen(self,t):
    if (t < self.start):
      return self.values[t]
    else:
      return self.dist.sample([self.input_size])

def sim(args):
  
  samples = args.num_samples
  num_nets = int(np.sqrt(samples))
  tests_per_net = int(samples/num_nets)

  N = args.output_size
  iW = tdist.Normal(torch.tensor([0.0]), torch.tensor([args.SigmaW/np.sqrt(N)]))
  W = iW.sample([N,N])

  iU = tdist.Normal(torch.tensor([0.0]), torch.tensor([args.SigmaU/np.sqrt(N)]))
  iB = tdist.Normal(torch.tensor([args.Mu]), torch.tensor([args.SigmaB]))
 
  n = num_nets
  m = tests_per_net
  U = iU.sample([n,N,N]).repeat(m,1,1,1).reshape([n*m,N,N])
  W = iW.sample([n,N,N]).repeat(m,1,1,1).reshape([n*m,N,N])
  B = iB.sample([n,N]).repeat(m,1,1,1).reshape([n*m,N,1])
  
  s = torch.zeros([n*m,N,1])
  
  Z = torch.zeros(n*m,N,1)

  T = args.sim_time
  Q = np.ones(T)
  Qu = np.zeros(T)
  C = np.ones(T)
  Cu = np.zeros(T)
  Zdiag = np.zeros(T)
  Xi = np.zeros(T)

  for t in  range(T): ## range(args.sim_time):
    for k in range(n*m):
      Z[k] = args.In.gen(t).reshape(N,1)


    if (not args.tied_weights):
        W = iW.sample([n,N,N]).repeat(m,1,1,1).reshape([n*m,N,N])

    u = W.bmm(s) + U.bmm(Z) + B
    s = args.act(u)
    
    Es = torch.zeros([N,1])
    Eu = torch.zeros([N,1])

    for a_sample in range(samples):
      g = np.random.randint(0,num_nets)
      i = np.random.randint(0,tests_per_net)*num_nets + g
      j = np.random.randint(0,tests_per_net)*num_nets + g
      Q[t]+= Edot(s[i],s[i])
      C[t]+= Edot(s[i],s[j])
      Qu[t]+= Edot(u[i],u[i])
      Cu[t]+= Edot(u[i],u[j])
      Es= Es + s[i]
      Eu= Eu + u[i]
      Zdiag[t]+= angle(Z[i],Z[j])
    
    normalizor= samples*N
    Es = Es/(samples)
    Eu = Eu/(samples)
    Q[t] = Q[t]/(normalizor) - Edot(Es,Es)/N
    C[t] = C[t]/(normalizor)  - Edot(Es,Es)/N
    C[t] = C[t]/Q[t]
    Qu[t] = Qu[t]/(normalizor) - Edot(Eu,Eu)/N
    Cu[t] = Cu[t]/(normalizor) - Edot(Eu,Eu)/N
    Cu[t] = Cu[t]/Qu[t]
    Zdiag[t] =Zdiag[t]/samples
  
    if (t>1):
      Xi[t] = C[t]/C[t-1]
  if (args.plot):
    plt.plot(range(T),C,label="C")
    ##plt.plot(range(T),Q,label="Q")
    plt.plot(range(T),Zdiag,label="Z (diag)")
    ##plt.plot(range(T),Qu,label="Qu")
    plt.plot(range(T),Cu,label="Cu")
    #plt.plot(range(T),Xi,label="Xi")

    plt.legend()
    plt.show()
  return { 'Q': Q, 'C': C, 'Qu': Qu, 'Cu': Cu, 'Xi': Xi, 'Zdiag' : Zdiag}
  


def compute_Xi(mu, qu, cu, ww):
  muu = mu**2
  tmp = 1 + (muu)/qu+(2*mu/np.sqrt(qu))*cu
  tmp = tmp/(1-cu*cu)
  tmp = -(1/2) * tmp * (muu/qu)
  tmp = np.exp(tmp)
  tmp = (2*ww/(np.pi))*tmp
  return tmp

def calc(args):
  ##print("Calculation not supported yet")
  T = args.sim_time
  Q = np.ones(T)*args.Q0
  Qu = np.zeros(T)
  C = np.ones(T)*args.C0
  Cu = np.zeros(T)
  CholskeyC = np.zeros(T)
  Xi = np.zeros(T)
  
  Zdiag = np.zeros(T)
  ww = args.SigmaW**2
  uu = args.SigmaU**2
  bb = args.SigmaB**2
  mu = args.Mu

  n = args.num_samples ##num samples for convariance measurement
  
  stable_point = False
  
  for t in range(1,T):    
    Zdiag[t] = args.In.cosine(t,n)# sigma Z off diagonal
    Qu[t] = ww*Q[t-1]+uu+bb
    Cu[t]= (ww*Q[t-1]*C[t-1] + uu*Zdiag[t] + bb)/Qu[t]
    Xi[t] = compute_Xi(mu,Qu[t],Cu[t],ww)
    
    if (stable_point):
      Q[t] = 1-math.erf(mu/(np.sqrt(2)*Qu[t]))
      C[t] = C[t-1]*Xi
    else:
      
      if Cu[t] >= 1.0:
        Cu[t] = 0.99
      
      diag = Qu[t]
      off_diag = Qu[t]*Cu[t]

      Cov= [[diag,off_diag],[off_diag,diag]]
      U = tdist.Normal(torch.tensor([mu]), torch.tensor([Qu[t]]))
      u = U.sample([n]).sign()
      Eu = torch.mean(u)
      Q[t] = torch.mean(u*u) - Eu**2
      
      U1 = tdist.MultivariateNormal(torch.tensor([mu,mu]), torch.tensor(Cov))
      
      Us = U1.sample([n])
      u1 = Us[:,0]
      u2 = Us[:,1]
      ## no cholskey decomposition
      C[t] = torch.mean(u1.sign()*u2.sign())- Eu**2
      C[t] = C[t]/Q[t]

      Uc = tdist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.eye(2)).sample([n])
      u1 = Uc[:,0]
      u2 = Uc[:,1]
      ch1 = np.sqrt(Qu[t])*u1+mu
      ch2 = np.sqrt(Qu[t])*(Cu[t] * u1 + np.sqrt(1-Cu[t]*Cu[t])*u2)  + mu
      CholskeyC[t] = torch.mean(ch1.sign()*ch2.sign()) - Eu**2
      CholskeyC[t] = CholskeyC[t]/Q[t]
      
  if (args.plot):
    plt.plot(range(T),C,label="C")
    ##plt.plot(range(T),Q,label="Q")
    plt.plot(range(T),Zdiag,label="Z (diag)")
    ##plt.plot(range(T),Qu,label="Qu")
    plt.plot(range(T),Cu,label="Cu")
    #plt.plot(range(T),Xi,label="Xi")
    #plt.plot(range(T),CholskeyC,label="Cholskey C")
    plt.legend()
    plt.show()

  return { 'Q': Q, 'C': C, 'Qu': Qu, 'Cu': Cu, 'Xi': Xi, 'Zdiag' : Zdiag}


def get_ff_qstar(args):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  mu = args.Mu
  n = args.num_samples ##num samples for convariance measurement  
  N = args.output_size
  
  q = 0.5
  for i in range(N):

    Qu = ww*q+bb
    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)
    q = torch.mean(u*u) - Eu**2
    
  return q

def get_ff_star(args):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  mu = args.Mu
  n = args.num_samples ##num samples for convariance measurement  
  N = args.output_size
  
  Qstar = get_ff_qstar(args)
  c=0.5
  for i in range(N):
    Qu = ww*Qstar+bb
    Cu= (ww*Qstar*c + bb)/Qu
    ##print("Qu = %.02f, Cu = %.02f" % (Qu,Cu))
    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)
    
    Uc = tdist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.eye(2)).sample([n])
    u1 = Uc[:,0]
    u2 = Uc[:,1]
    ch1 = np.sqrt(Qu)*u1+mu
    ch2 = np.sqrt(Qu)*(Cu * u1 + np.sqrt(1.0-Cu*Cu)*u2) + mu
    tmp = torch.mean(args.act(ch1) * args.act(ch2) ) - Eu**2
    c = tmp/Qstar
    if (c>=1.0):
      return 1.0,Qstar
    elif (c<=0.0):
      return 0.0,Qstar
  return c,Qstar

def get_chi_star(args,Cstar,Qstar,eps =0.01):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  mu = args.Mu
  n = args.num_samples ##num samples for convariance measurement  
  N = args.output_size
  M = np.zeros(N)
  mx = min(Cstar+eps,1.0)
  mn = max(Cstar-eps,0.0)
  Cv= torch.linspace(mn,mx,N)
  for i in range(N):
    c= Cv[i]
    Qu = ww*Qstar+bb
    Cu= (ww*Qstar*c + bb)/Qu
    ##print("Qu = %.02f, Cu = %.02f" % (Qu,Cu))
    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)
    
    Uc = tdist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.eye(2)).sample([n])
    u1 = Uc[:,0]
    u2 = Uc[:,1]
    ch1 = np.sqrt(Qu)*u1+mu
    ch2 = np.sqrt(Qu)*(Cu * u1 + np.sqrt(1.0-Cu*Cu)*u2) + mu
    tmp = torch.mean(args.act(ch1) * args.act(ch2) ) - Eu**2
    mc = tmp/Qstar
    if (mc>=1.0):
      mc= 1.0
    M[i] = mc
  
  slope, intercept, r_value, p_value, std_err = stats.linregress(Cv,M)

  return slope

def get_qstar(args):
  ww = args.SigmaW**2
  uu = args.SigmaU**2
  bb = args.SigmaB**2
  mu = args.Mu

  n = args.num_samples ##num samples for convariance measurement  
  N = args.output_size

  Q = np.linspace(0,50,N)
  
  Mq = np.zeros(N)
  for i in range(N):
    q = Q[i]
    Qu = ww*q+uu+bb
    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)
    Mq[i] = torch.mean(u*u) - Eu**2


  
  argmin = np.argmin(np.abs(Mq - Q))
  Qstar1 = Q[argmin]
  
  
  if (args.plot):
    max_plot= N if (argmin*2 > N) else argmin*2
    plt.plot(Q[0:argmin*2],Mq[0:argmin*2],label="Mapping")
    plt.plot(Q[0:argmin*2],Q[0:argmin*2],label="Q=Q line")
    plt.title("Q Mapping")
    plt.legend()
    plt.show()
  
  delta=1.0
  if (delta < Qstar1):
    delta = Qstar1
  Q = np.linspace(Qstar1-delta,Qstar1+delta,N)
  
  Mq = np.zeros(N)
  for i in range(N):
    q = Q[i]
    Qu = ww*q+uu+bb
    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)
    Mq[i] = torch.mean(u*u) - Eu**2

  argmin = np.argmin(np.abs(Mq - Q))
  Qstar = Q[argmin]

  return Qstar

def draw_map(args, C = None):
  ww = args.SigmaW**2
  uu = args.SigmaU**2
  bb = args.SigmaB**2
  mu = args.Mu

  n = args.num_samples ##num samples for convariance measurement  

  
  if not type(C) is np.ndarray:
    C = np.linspace(0,1,args.output_size)
  
  N = len(C)
  M = np.zeros(N)

  Qstar = get_qstar(args)
  Z = 1


  for i in range(N):
    c = C[i]
    Qu = ww*Qstar+uu+bb
    Cu= (ww*Qstar*c + uu*Z + bb)/Qu

    U = tdist.Normal(torch.tensor([mu]), torch.tensor([np.sqrt(Qu)]))
    u = args.act(U.sample([n]))
    Eu = torch.mean(u)

    Uc = tdist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.eye(2)).sample([n])
    u1 = Uc[:,0]
    u2 = Uc[:,1]
    ch1 = np.sqrt(Qu)*u1+mu
    ch2 = np.sqrt(Qu)*(Cu * u1 + np.sqrt(1-Cu*Cu)*u2) + mu
    M[i] = torch.mean(args.act(ch1) * args.act(ch2) ) - Eu**2
    M[i] = M[i]/Qstar

  if (args.plot):
    plt.plot(C,M,label="Mapping")
    plt.plot(C,C,label="C=C line")
    plt.title("C Mapping")
    plt.legend()
    plt.show()
    
    
  argmin = np.argmin(np.abs(C[0:(len(C)-1)] - M[0:(len(C)-1)]))
  Cstar = M[argmin]
  return {'C': C, 'Qstar': Qstar, 'M': M, 'Cstar': Cstar}


def plot_convergence(simC, calC):
  T = len(simC)
  simC_star=  simC[T-1]
  calC_star= calC[T-1]

  simCdiff= np.abs(simC- simC_star)
  calCdiff= np.abs(calC- calC_star)
  
  plt.plot(range(T),simCdiff,label="simulation convergence")
  plt.plot(range(T),calCdiff,label="calculation convergence")
  plt.yscale = "log"
  plt.legend()
  plt.show()

def plot_graph(results):
  T = len(results['C'])
  plt.plot(range(T),results['C'],label="C")
  plt.plot(range(T),results['Q'],label="Q")
  plt.plot(range(T),results['Qu'],label="Qu")
  plt.plot(range(T),results['Zdiag'],label="Z (diag)")
  plt.plot(range(T),results['Cu'],label="Cu")
  plt.legend()
  plt.show()

def compare(results):
  keys=list(results.keys())
  T = len(results[keys[0]])
  for key in keys:
    plt.plot(range(T),results[key],label=key)
  plt.legend()
  plt.gcf().set_size_inches(10, 10)
  plt.show()

  
def noisy_sign(x):
  return actv.noisy_sign(x)

def sr_sign(x):
  return actv.sr_sign(x)

def cast_int2(x):
  return actv.cast_int2(x)

def cast_intn(x,n):
  return actv.cast_intn(x,n)

def cast_even_intn(x,n):
  return actv.cast_even_intn(x,n)


def report(arguments, samples= 1024*256):
    for args in arguments:
        tmp = args.num_samples
        args.num_samples = samples
        c,q = get_ff_star(args)
        chi = get_chi_star(args,c,q)
        args.num_samples = tmp
        print("%s:, C = %.02f, Q = %.02f, chi = %.02f" % (args.desc,c,q,chi))

def get_cu(C,args,q):
    ww = args.SigmaW**2
    bb = args.SigmaB**2
    uu = args.SigmaU**2
    return (q*C*(ww)+bb+uu)/(ww*q+bb+uu)

def get_qu(args):
    ww = args.SigmaW**2
    bb = args.SigmaB**2
    uu = args.SigmaU**2
    q= get_qstar(args)
    Qu = ww*q+uu+bb
    return Qu
  
def main():
  np.set_printoptions(precision=2)
  args = parse()
  if (args==None):
    return
  if args.simulate:
    sQ,sC,sQu,sCu,sXi = sim(args)
  if args.calculate:
    cQ,cC,cQu,cCu,cXi = calc(args)
    plot_convergence(args,sC,cC)

  pass

if __name__ == "__main__":
  main()

##runfile('C:/Users/Yaniv/Downloads/np_calc.py', "-T 50 -c -p 0.5,1.3,1,1 -I 1,1")
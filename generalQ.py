import numpy as np
import argparse
import re
import math
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt
import time

import activations as actv
import discrete_rnn as drnn

from sympy.solvers import solve , nsolve
from sympy import Symbol
from sympy import sin, limit
from scipy import stats



def normal_cdf(x,sigma2=1.0,offset=0.0):
  return tdist.Normal(torch.tensor([offset]), torch.tensor([sigma2])).cdf(x)

def normal_pdf(x,Q=1.0,offset=0.0):
  factor =1.0/(np.sqrt(2*np.pi*Q)) 
  return factor*np.exp((-0.5)*((x-offset)**2)/Q)


def meshgrid(offsets):
  N = offsets.numel()
  offsets_x=offsets.view([1,N])
  offsets_y=offsets.view([N,1])
  mu0 =  offsets_x.repeat(N,1)
  mu1 =  offsets_y.repeat(1,N)
  return mu0, mu1



def calc_Mq(args,offsets,const,q):
  N = offsets.numel()
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  Qu = ww*q+bb
  factor=(2*const/(N))**2
  cdfs=normal_cdf(offsets,np.sqrt(Qu)).repeat(N,1)
  cdfs= 1 - cdfs
  mcdf = torch.min(cdfs,cdfs.t())
  ccdf = cdfs*cdfs.t()
  res = torch.sum(mcdf-ccdf)
  return res*factor

def calc_qstar(args,offsets, const , steps=20 , init_q=1.0):
  q = init_q
  for i in range(steps):
    q = calc_Mq(args,offsets,const,q)
  return q

def calc_chi(args,offsets, const ,Q,C=0.0):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  Qu = ww*Q+bb
  Cu = (ww*Q*C+bb)/Qu
  N = offsets.numel()
  s=0.0
  factor=ww*(2*const/(N))**2
  factor= factor/(Qu*2*np.pi*np.sqrt(1-Cu**2))
  mu0,mu1 = meshgrid(offsets)
  s = torch.sum(torch.exp(-1/(2*Qu*(1-Cu**2))*(mu0**2+mu1**2+2*Cu*mu0*mu1)))
  return s*factor

def ff_partial_map_approx(args, mu0, mu1, Q=None, C=0.0):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  Qstar = Q
  Qu = ww*Qstar +bb 
  Cu = (ww*Qstar*C +bb)/Qu 
  g = np.arcsin(Cu)
  ro = np.sqrt(1.0-Cu**2)
  tmp1 = g/(2*np.pi)
  tmp1 = tmp1/Q
  mu0 = mu0/np.sqrt(Qu)
  mu1 = mu1/np.sqrt(Qu)
  tmp2 = mu0**2+mu1**2- mu0*mu1*(2*Cu)/(1.0+ro)
  tmp2 = Cu/(2*g*ro)*tmp2
  M = tmp1*(np.exp(-tmp2))
  return M

def calc_c0(args,offsets,const,Q):
  N = offsets.numel()
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  Qu = ww*Q+bb  
  Cu = bb/Qu 
  factor=(2*const/(N))**2
  factor= factor/Q
  s=0.0
  mu0,mu1 = meshgrid(offsets)
  g = np.arcsin(Cu)
  ro = np.sqrt(1.0-Cu**2)
  tmp1 = (g/(2*np.pi))*factor
  tmp2 = mu0**2+mu1**2- mu0*mu1*(2*Cu)/(1.0+ro)
  tmp2 = Cu/(2*g*ro*Qu)*tmp2
  tmp3 = tmp1*(np.exp(-tmp2))
  s= torch.sum(tmp3) 
  return s

def estimate_chi0(states,W):
  args = drnn.RnnArgs(SigmaW = W, SigmaU=0.0 ,SigmaB=0.001)
  offsets, scale, sigmaW = optimal_spacing(states)
  const = scale
  Qstar =  calc_qstar(args, offsets, const ,steps=200)
  chi0=calc_chi(args, offsets, const ,Qstar,0.0)
  if np.isnan(chi0):
    print("Error: chi is nan")
    print(Qstar)
    print(scale)
    print(offsets)
  return chi0

def estimate_chi(args, offsets, const, Qstar):
  chi0=calc_chi(args, offsets, const ,Qstar,0.0)
  c0=calc_c0(args, offsets, const ,Qstar)
  Cest = c0/(1-chi0)
  chi=calc_chi(args, offsets, const ,Qstar,Cest)
  return chi

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

  return c,Qstar

def heaviside(x):
  res = 1/2+torch.sign(x)/2
  return res

def ff_partial_map(args, mu0, mu1, Q=None, C = None, factor=1):
  ww = args.SigmaW**2
  bb = args.SigmaB**2
  mu = args.Mu
  n = args.num_samples ##num samples for convariance measurement  
  
  if not type(C) is np.ndarray:
    C = np.linspace(0,1,args.output_size)
    
  ##if (Q == None):
  ##  Qstar = drnn.get_ff_qstar(args)
  ##else:
  Qstar = Q

  N = len(C)
  M = np.zeros(N)

  act = (lambda x: heaviside(x))
  
  for i in range(N):
    c = C[i]
    Qu = ww*Qstar +bb 
    Cu= (ww*Qstar*c + bb)/Qu
    
    ##I still need to calculate the diagonal/off diagonal 
    if (0):
      U1 = tdist.Normal(torch.tensor([mu0]), torch.tensor([np.sqrt(Qu)]))
      U2 = tdist.Normal(torch.tensor([mu1]), torch.tensor([np.sqrt(Qu)]))
      u1 = act(U1.sample([n]))
      u2 = act(U2.sample([n]))
      Eu1 = torch.mean(u1)
      Eu2 = torch.mean(u2)
    else:
      Eu1= normal_cdf(mu0,Qu)
      Eu2= normal_cdf(mu1,Qu)

    Uc = tdist.MultivariateNormal(torch.tensor([0.0,0.0]), torch.eye(2)).sample([n])
    u1 = Uc[:,0]
    u2 = Uc[:,1]
    ch1 = np.sqrt(Qu)*u1+mu0
    ch2 = np.sqrt(Qu)*(Cu * u1 + np.sqrt(1-Cu*Cu)*u2) + mu1
    M[i] = torch.mean(act(ch1) * act(ch2))  - Eu1*Eu2
    M[i] = M[i]*factor/Qstar
  return {'C': C, 'Qstar': Qstar, 'M': M}



def summary(results):
  N = 0
  for res in results:
    N2 = len(res['C'])
    if (N>0 and N!=N2):
      print("Error- unmatching vectors in generalQ/summary")
      return
    else:
      N = N2
    
  tot_sum=np.zeros(N)
  for res in results:
    ##print(res['M'].shape)
    tot_sum+=res['M']
  C = results[0]['C']
  return {'M': tot_sum, 'C': C, 'label': "total"}

def gen_cast(x,offsets,const):
  N=offsets.numel()
  if x.is_cuda:
    res = torch.zeros(x.size()).cuda()
    factor=torch.Tensor([2*const/N]).cuda()
    if not const.is_cuda:
      res = res - const.cuda()
  else:
    res = torch.zeros(x.size())
    factor=torch.Tensor([2*const/N])
    res=res-const

  for g in offsets:
    res+= factor*heaviside(x - g)
  return res

def lincast(x,states,const=2.0):
  return gen_cast(x,torch.linspace(-const,const,states-1),const)


def Estimate(args,offsets,const,steps=50):
    Qstar = calc_qstar(args, offsets, const ,steps=steps)
    chi=estimate_chi(args, offsets,const ,Qstar)
    return chi

def Derivative(args,offsets,const,dx=0.1):
    N= offsets.numel()
    ders=torch.zeros(N)
    v1 = Estimate(args,offsets,const)
    aoffsets = offsets.repeat(N,1)+ torch.eye(N)*dx
    for k in range(N):
        offsets2 = aoffsets[k][:]
        v2 = Estimate(args,offsets2,const)
        ders[k] = ((v2-1.0)**2 - (v1-1.0)**2)/dx
    return ders
  
def ConstDerivative(args,offsets,const,dx=0.1):
    N= offsets.numel()
    ders=torch.zeros(N)
    v1 = Estimate(args,offsets,const)
    v2 = Estimate(args,offsets,const+dx)
    const_der = ((v2-1.0)**2 - (v1-1.0)**2)/dx
    return const_der


def report_offsets(args, offsets,const):
  act = (lambda x: gen_cast(x, offsets, const ))
  args.act = act
  args.present(const+3.0)
  print("Constant: %.02f" % const)
  print("Offsets: %s" % offsets)
  return

def save(offsets,const,states,name="opt"):
  torch.save(offsets,"%s_offset%d.pt" % (name,states))
  torch.save(const,"%s_const%d.pt" % (name,states))
  
def load(states,name="opt"):
  try:
    offsets = torch.load("%s_offset%d.pt" % (name,states))
    const = torch.load("%s_const%d.pt" % (name,states))
  except:
    try:
      offsets = torch.load("%s_offset%d.pt" % ("opt",states))
      const = torch.load("%s_const%d.pt" % ("opt",states))
    except:
      const = 2.0
      offsets = torch.linspace(-const,const,states-1)
  return offsets, const

def optimize_offsets(W,B,offsets,const,lr=5.0,lrr=0.00,verbose=2):
  states = offsets.numel()+1
  args= drnn.RnnArgs(SigmaW = W, SigmaU=0.0 ,SigmaB=B)
  if (verbose>=2):
    report_offsets(args,offsets,const)
  best_chi=Estimate(args,offsets,const)
  stable_chi=-1.0
  convergence = 0.000001
  opt_chi = 0.00
  for p in range(500):
      j=0
      while (j < 10):
          start=time.time()
          for i in range(100):
              ders = Derivative(args,offsets,const)
              offsets = offsets - ders*lr
              cd = ConstDerivative(args,offsets,const)
              const = const - cd*lr
          chi = Estimate(args,offsets,const)
          totime=time.time()-start
          if (verbose>=1):
              print("chi: %.06f, time: %.02f" % (chi ,totime))
          if torch.isnan(chi) or (chi < best_chi):

              offsets,const = load(states,name="opt2")
              best_chi = Estimate(args,offsets,const)
              stable_chi = best_chi
              chi = best_chi
              if (verbose>=1):
                  print("falling back to... %.04f" % best_chi)
              lr = lr/2
              lrr = lrr/2
          else:
              save(offsets,const,states,name="opt2")
              best_chi = chi
              j=j+1
              
      if (lr-lrr>0.0):
          lr=lr-lrr
      if (verbose>=2):
          report_offsets(args,offsets,const)
      if (chi >= best_chi):
          if (best_chi-stable_chi<convergence):
              if (opt_chi > best_chi):
                  print("Warning: Opt chi > chi: (%.04f > %.04f)"  % (opt_chi, best_chi) )
              return best_chi
          stable_chi = best_chi
          opt_chi = max(opt_chi, best_chi)
  print("Warning: No convergence.")
  return best_chi

##----------------------------------

def Estimate2(args,offsets,const):
    N = offsets.numel()
    const = (offsets[N-1] - offsets[0])/2
    Qstar = calc_qstar(args, offsets, const ,steps=50)
    chi=estimate_chi(args, offsets,const ,Qstar)
    return chi
  
def Derivative2(args,offsets,const,dx=0.1):
    N= offsets.numel()
    ders=torch.zeros(N)
    v1 = Estimate2(args,offsets,const)
    aoffsets = offsets.repeat(N,1)+ torch.eye(N)*dx
    for k in range(N):
        offsets2 = aoffsets[k][:]
        v2 = Estimate2(args,offsets2,const)
        ders[k] = ((v2-1.0)**2 - (v1-1.0)**2)/dx
    return ders

def optimize_offsets2(W,B,offsets,const,lr=1.6,lrr=0.01,verbose=2):
  states = offsets.numel()+1
  args= drnn.RnnArgs(SigmaW = W, SigmaU=0.0 ,SigmaB=B)
  if (verbose>=2):
    report_offsets(args,offsets,const)
  best_chi=Estimate(args,offsets,const)
  stable_chi=-1.0
  convergence = 0.000001
  opt_chi = 0.00
  for p in range(500):
      j=0
      while (j < 10):
          start=time.time()
          for i in range(100):
              ders = Derivative(args,offsets,const)
              offsets = offsets - ders*lr
              const = (offsets[states-2] - offsets[0])/2
          chi = Estimate(args,offsets,const)
          totime=time.time()-start
          if (verbose>=1):
              print("chi: %.06f, time: %.02f" % (chi ,totime))
          if torch.isnan(chi) or (chi < best_chi):
              offsets,const = load(states)
              best_chi = Estimate(args,offsets,const)
              stable_chi = best_chi
              chi = best_chi
              if (verbose>=1):
                  print("falling back to... %.04f" % best_chi)
              lr = lr/2
              lrr = lrr/2
          else:
              save(offsets,const,states)
              best_chi = chi
              j=j+1
              
      if (lr-lrr>0.0):
          lr=lr-lrr
      if (verbose>=2):
          report_offsets(args,offsets,const)
      if (chi >= best_chi):
          if (best_chi-stable_chi<convergence):
              if (opt_chi > best_chi):
                  print("Warning: Opt chi > chi: (%.04f > %.04f)"  % (opt_chi, best_chi) )
              return best_chi
          stable_chi = best_chi
          opt_chi = max(opt_chi, best_chi)
  print("Warning: No convergence.")
  return best_chi

def optimal_activation(states):
  spacing = optimal_normalized_spacing(states)
  D = states-1
  K = int((D-1)/2)
  offsets = spacing * torch.linspace(-K,K,D)
  thresh = spacing*D
  q = get_required_q(thresh)
  
  cdfs=normal_cdf(offsets,1.0).repeat(D,1)
  maxcdf = torch.max(cdfs,cdfs.t())
  mincdf = torch.min(cdfs,cdfs.t())
  tmp = torch.sum((1-maxcdf)*mincdf)
  scale = (D/2)*np.sqrt(q/tmp)
  sigmaW = 1.0/np.sqrt(q)
  return offsets, scale, sigmaW, thresh


def DepthScale(states):
  ##evens: slope: -1.8175, intercept: 0.7097, std_err: 0.0032 (s+1) v2
  ##evens: slope: -1.8172, intercept: 0.7100, std_err: 0.0030 (s+1) v3 (close enough to regard as identical)
  
  slope = -1.8172
  intercept = 0.71
  chi =  1 - ((states+1)**slope)*np.exp(intercept)
  return -1.0/np.log(chi)

def Mjj(Qu, sigmaW, threshold,slope):
  return ((sigmaW*slope)**2)*math.erf(threshold/np.sqrt(2*Qu))

def optimize_ste(Qu,sigmaW, threshold, samples=1000):
  a = threshold/np.sqrt(2*Qu)
  a = torch.Tensor([a])
  ww = sigmaW**2
  slope = sigmaW*np.sqrt(torch.erf(a))  
  return slope

def optimize_ste2(Qu,sigmaW, threshold, samples=1000):
  a = threshold/np.sqrt(2*Qu)
  a = torch.Tensor([a])
  ww = sigmaW**2
  slope = sigmaW*np.sqrt(torch.erf(a)/(Qu))
  return slope

def optimal_spacing2(states):
  spacing = optimal_normalized_spacing(states)
  D = states-1
  if (states % 2 == 0):
    K = int((D-1)/2)
    offsets = spacing * torch.linspace(-K,K,D)
    scale = spacing*K
  else:
    K = int(D/2)
    offsets = spacing * torch.linspace(-K+0.5,K-0.5,D)
    scale = spacing*(K-0.5)
  sigmaW = get_optimizing_W(offsets, scale)
  return offsets, scale, sigmaW

def optimal_spacing(states):
  spacing = optimal_normalized_spacing(states)
  D = states-1
  if (states % 2 == 0):
    K = int((D-1)/2)
    offsets = spacing * torch.linspace(-K,K,D)
    scale = spacing*K
  else:
    K = int(D/2)
    offsets = spacing * torch.linspace(-K+0.5,K-0.5,D)
    scale = spacing*(K-0.5)
  scale=1.0
  factor=(2*scale/D)**2
  normalized_offsets = offsets
  cdfs=normal_cdf(normalized_offsets,1.0).repeat(D,1)
  maxcdf = torch.max(cdfs,cdfs.t())
  mincdf = torch.min(cdfs,cdfs.t())
  Q = factor*torch.sum((1-maxcdf)*mincdf)
  real_spacing= 2.0/(D)
  ##real_spacing= 3.0/(D)
  
  sigmaW = np.sqrt(1.0/Q)*(real_spacing/spacing)
  offsets = offsets*(real_spacing/spacing)
  Qu = Q*(sigmaW**2)
  return offsets, scale, sigmaW, Qu
  
  
  
  
  
def optimal_normalized_spacing(states):
  ##evens: slope: -0.8809, intercept: 1.4080, std_err: 0.0009 (s+1) all-accounted v2
  ##odds: slope: -0.8916, intercept: 1.4598, std_err: 0.0024 (s) all-accounted v2
  ##evens: slope: -0.8787, intercept: 1.4015, std_err: 0.0009 (s+1) v3
  ##odds: slope: -0.8793, intercept: 1.4034, std_err: 0.0010 (s+1) v3
  ##both : slope: -0.8790, intercept: 1.4025, std_err: 0.0007 (s+1) v3
  slope=-0.879
  intercept=1.4025
  spacing= ((states+1)**slope)*np.exp(intercept)
  return spacing

def get_required_q(thresh):
  x = thresh/np.sqrt(2)
  q = math.erf(x)/(2*x**2)
  return q

def get_optimizing_W(normalized_offsets,scale=1.0):
  D = normalized_offsets.numel() ## states-1
  factor=(2*scale/D)**2
  cdfs=normal_cdf(normalized_offsets,1.0).repeat(D,1)
  maxcdf = torch.max(cdfs,cdfs.t())
  mincdf = torch.min(cdfs,cdfs.t())
  Q = factor*torch.sum((1-maxcdf)*mincdf)
  ##Qu = (W**2)*Q -> (for sqrt(Qu) = rescale):
  W = np.sqrt(1.0/Q)
  return  W

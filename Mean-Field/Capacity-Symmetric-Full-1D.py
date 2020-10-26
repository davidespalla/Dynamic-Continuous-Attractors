#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:01:59 2019

@author: davide
"""
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import os




def main():
    foldername="Capacity-Symmetric-Full-1D"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    g=np.linspace(0.6,3,20)
    L=np.logspace(1,3,10)
    numw=20
    nsummands=10
    Smat=np.zeros((len(L),len(g),numw))
    X=np.zeros((len(g),numw))
    Y=np.zeros((len(g),numw))
    for l in range(len(L)):
        for i in range(len(g)):
            ws=np.linspace(findwstar(g[i])-4,findwstar(g[i])-0.001,numw)
            for j in range(len(ws)):
                resolution=int(5*L[l])
                Smat[l][i][j]=(S(g[i],ws[j],L[l],resolution,nsummands))
                X[i][j]=g[i]
                Y[i][j]=ws[j]
                print("Done L="+str(L[l])+", g="+str(i+1)+"/20")
    np.save(foldername+"/Smat",Smat)
    np.save(foldername+"/X",Smat)
    np.save(foldername+"/Y",Smat)


    return


#FUNCTIONS

def sigma(x):
    return norm.pdf(x)
def PHI(x):
    return norm.cdf(x)

def N(x):
    return x*PHI(x)+sigma(x)

def M(x):
    return (1+pow(x,2))*PHI(x)+x*sigma(x)

def U1(x,g,w):
    return 2*g*N(x)-x+w

def U2(x,g):
    return 2*g*PHI(x)-1

def U(x,g,w):
    I=quad(U1,-5,x,args=(g,w))[0]
    return I


def K(r,r1,L):
    d=abs(r-r1)
    if d>float(L/2.0):
        d=L-d
    return np.exp(-d)

def findutilde(g):
    x=1.0/(2.0*g)
    utilde=norm.ppf(x)
    return utilde

def samesign(a, b):
        return a * b > 0

def findumax(g,w,precision):
    low=-100
    high=findutilde(g)

    while samesign(U1(low,g,w), U1(high,g,w)):
        low=2*low

    converged=False

    while not converged:
        midpoint = (low + high) / 2.0
        if samesign(U1(low,g,w),U1(midpoint,g,w)):
            low = midpoint
        else:
            high = midpoint
        converged= (abs(U1(midpoint,g,w))<precision)

    return midpoint


def findmin(g,w,precision):
    low=findutilde(g)
    high=100

    while samesign(U1(low,g,w), U1(high,g,w)):
        low=2*low

    converged=False

    while not converged:
        midpoint = (low + high) / 2.0
        if samesign(U1(low,g,w),U1(midpoint,g,w)):
            low = midpoint
        else:
            high = midpoint
        converged= (abs(U1(midpoint,g,w))<precision)

    return midpoint


def findwstar(g):
    return -2*g*N(findutilde(g))+findutilde(g)

def findustar(g,w,precision):
    low=findutilde(g)
    high=100
    umax=findumax(g,w,precision)
    y=U(umax,g,w)

    while samesign(U(low,g,w)-y, U(high,g,w)-y):
        high=2*high

    converged=False

    while not converged:
        midpoint = (low + high) / 2.0
        if samesign(U(low,g,w)-y, U(midpoint,g,w)-y):
            low = midpoint
        else:
            high = midpoint
        converged= (abs(U(midpoint,g,w)-y)<precision)

    return midpoint

def F(x,g,w):
    return -2*g*N(x)+x-w

def Integrate(u0,g,w,rmax,resolution,precision):
    umax=findumax(g,w,precision)
    xold=u0
    vold=0
    r=np.linspace(0,rmax,resolution)
    Dr=rmax/float(resolution)
    solution=np.zeros(len(r))
    for i in range(len(r)):
        solution[i]=xold
        a=F(xold,g,w)
        vnew=vold+(a)*Dr
        xnew=xold+(vnew)*Dr
        if xnew>xold: #prevents the function from oscillating
            xnew=xold
        if xnew<umax: #prevents the function from exploding
            xnew=umax
        xold=xnew
        vold=vnew
    return r,solution

def vfunc(u,g):
    return g*N(u)

def Findvprofile(g,w,rmax,resolution):
    if g<=0.5:
        print("Value of g <0.5: no soultion ammitted")
        return
    wstar=findwstar(g)
    if w>=wstar:
        print("Value of w>w*: no solution ammitted")
        return
    precision=0.00001
    ustar=findustar(g,w,precision)
    r,u=Integrate(ustar,g,w,rmax,resolution,precision)

    v=np.asarray(list(map(lambda x: vfunc(x,g),u)))
    return r,v

def h(r,r1,v,w,L):
    k0=1.0/float(L)
    k=np.asarray(list(map(lambda x: K(x,r,L),r1)))
    Dr=float(L)/float(len(r1))
    h=sum((k-k0)*v)*Dr+w
    return h

def PSI(r,r1,v,w,L):
    Dr=float(L)/float(len(r1))
    I=sum(list(map(lambda x: PHI(h(x,r,v,w,L)),r1)))*Dr
    return (1.0/L)*I

def T0summand(p,Psi):
    num=(2.0/(1.0+pow(p,2)))
    den=(1.0-Psi*num)
    return pow(num,2)/pow(den,2)

def calculateT0(Psi,nsummands):
    out=0
    for p in range(1,nsummands):
        out=out+T0summand(p,Psi)
    return out

def S(g,w,L,resolution,nsummands):
    r0,v0=Findvprofile(g,w,L/2,resolution)
    r=r0
    Dr=float(L)/float(len(r0))
    I=sum(list(map(lambda x: M(h(x,r0,v0,w,L)),r)))*Dr
    Psi=PSI(r,r0,v0,w,L)
    print("PSI="+str(Psi))
    T0=calculateT0(Psi,nsummands)
    print("T0="+str(T0))
    return pow(g,2)*I*T0

if __name__ == "__main__":
    main()

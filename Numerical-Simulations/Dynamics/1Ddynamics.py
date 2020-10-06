#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import random
import os



def main():

    SimulationName="1D"
    N=1000
    m=1
    gamma=0.5
    L=10.0
    f=0.3

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

    print("Starting dynamics")

    grid=RegularPfc(N,L,m) # defines environment
    np.save(SimulationName+"/pfc",grid)
    J=BuildJ(N,grid,L,gamma) # Builds connectivity
    #V=np.random.uniform(0,1,N)
    V=correlate_activity(grid[0],L)
    V=V/np.mean(V)

    Vvec=dynamics(f,V,N,J)

    np.save(SimulationName+"/Vdynamics",Vvec)


    print("Dynamics terminated, result saved")
    return

# FUNCTIONS

def K(x1,x2,L,gamma):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return 0*np.exp(-abs(d))+gamma*np.sign(d)*np.exp(-abs(d))

def KS(x1,x2,L):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return np.exp(-abs(d))

def transfer(h):
        if h>0:
            return h
        else:
            return 0

def RegularPfc(N,L,m):
        grid=np.zeros((m,N))
        tempgrid=np.zeros(N)
        for i in range(N):
            tempgrid[i]=i*float(L)/float(N)
        for j in range(m):
            random.shuffle(tempgrid)
            grid[j][:]=tempgrid
        return grid

def BuildJ(N,grid,L,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L,gamma)
    return J

def Sparsify(V,f):
        vout=V
        th=np.percentile(V,(1.0-f)*100)
        for i in range(len(V)):
            if vout[i]<th:
                vout[i]=0
            else:
                vout[i]=vout[i]-th
        return vout

def dynamics(f,V,N,J):
        maxsteps=200
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec

def correlate_activity(pos,L):
    V=np.zeros(len(pos))
    center=L/2
    for i in range(len(V)):
        V[i]=KS(pos[i],center,L)
    return V

if __name__ == "__main__":
    main()

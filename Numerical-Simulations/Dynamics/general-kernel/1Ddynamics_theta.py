#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import os


def main():

    SimulationName="1D_THETA"
    N=1000
    gamma=1
    L=10.0
    f=0.2

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

     #SAVE PARAMETERS:
    outfile=open(SimulationName+"/parameters.txt","w")
    outfile.write("N="+str(N)+"\n")
    outfile.write("gamma="+str(gamma)+"\n")
    outfile.write("L="+str(L)+"\n")
    outfile.write("f="+str(f)+"\n")
    outfile.close()

    print("Starting dynamics")

    grid=RegularPfc(N,L) # defines environment
    np.save(SimulationName+"/pfc",grid)
    J=BuildJ(N,grid,L,gamma) # Builds connectivity
    V=np.random.uniform(0,1,N)
    V=V/np.mean(V)

    Vvec=dynamics(f,V,N,J)

    np.save(SimulationName+"/Vdynamics",Vvec)



    print("Dynamics terminated, result saved")
    return

# FUNCTIONS

def K(x1,x2,L,gamma,KL=1):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        if d >0 and d< KL:
            asy=gamma
        elif d <0 and d> -KL:
            asy=-gamma
        else:
            asy=0
        return np.exp(-pow(d,2))+asy

def transfer(h):
        if h>0:
            return h
        else:
            return 0

def RegularPfc(N,L):
        grid=np.zeros(N)
        for i in range(N):
            grid[i]=i*float(L)/float(N)
        return grid

def BuildJ(N,grid,L,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            x1=grid[i]
            x2=grid[j]
            if i!=j:
                J[i][j]=K(x1,x2,L,gamma)
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
        maxsteps=100
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec

if __name__ == "__main__":
    main()

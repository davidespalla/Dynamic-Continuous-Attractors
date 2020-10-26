
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import os
import numpy as np
from scipy.optimize import newton


def main():

    L=60
    gammas=np.linspace(0,3,15)
    gs=[0.7]
    numw=1
    resolution=10000
    resampling_factor=100


    foldername="Capacity-asymmetric-L60"
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    x0=-L/3.0 #size of the environment=30
    raw_u=np.zeros((len(gammas),len(gs),numw,int(L*resolution/resampling_factor)))
    corrected_u=np.zeros((len(gammas),len(gs),numw,int(L*resolution/resampling_factor)))
    Deltas=np.zeros((len(gammas),len(gs),numw))
    wsMat=np.zeros((len(gs),numw))
    Smat=np.zeros((len(gammas),len(gs),numw))

    for i in range(len(gammas)):
        for j in range(len(gs)):
            ws=[wstar(gs[j])-0.001]  #wstar(gs[j])+np.linspace(-2,-0.001,numw)
            wsMat[j][:]=ws
            for k in range(len(ws)):
                print("Calculating profile for: gamma="+str(gammas[i])+" g="+str(gs[j])+" w="+str(ws[k]))
                w=ws[k]
                g=gs[j]
                gamma=gammas[i]
                u0=find_u0(g,w)+0.01
                Deltax=find_Deltax(g,w,gamma,u0,x0)
                Deltas[i][j][k]=Deltax
                Deltax_corrected=Deltax+0.01*Deltax
                corrected_u[i][j][k][:]=u_profile(g,w,gamma,Deltax_corrected,u0,x0,resolution)[::resampling_factor]
                #raw_u[i][j][k][:],raw_du=Integrate_u(g,w,gamma,Deltax,u0,x0,resolution)[::resampling_factor]
                print("profile calculated")

                v0=g*N(corrected_u[i][j][k][::resampling_factor])
                r0=np.linspace(-L/2.0,L/2.0,len(v0))
                r=r0
                Dr=float(L)/float(len(r0))
                I=sum(list(map(lambda x: M(h(x,r0,v0,w,gamma,L)),r)))*Dr
                Smat[i][j][k]=pow(g,2)*I
                print("Done. "+str(k+j*len(gs)+i*len(gs)*len(gammas))+"/"+str(len(gammas)*len(gs)*len(ws))+" completed")
    np.save(foldername+"/gammas",gammas)
    np.save(foldername+"/gs",gs)
    np.save(foldername+"/wsMat",wsMat)
    np.save(foldername+"/Deltas",Deltas)
    np.save(foldername+"/raw_u",raw_u)
    np.save(foldername+"/corrected_u",corrected_u)
    np.save(foldername+"/Smat",Smat)
    print("Calculation completed. Results saved.")


    return


def sigma(x):
    return norm.pdf(x)

def PHI(x):
    return norm.cdf(x)

def N(x):
    return x*PHI(x)+sigma(x)

def M(x):
    return (1+pow(x,2))*PHI(x)+x*sigma(x)

def Nprime(x):
    return PHI(x)

def F(u,uprime,utilde,gamma,g,w):
    return -2*g*gamma*PHI(u)*uprime-2*g*N(u)+utilde-w

def wstar(g):
    x=1.0/(2.0*g)
    return -2*g*N(norm.ppf(x))+norm.ppf(x)

def Integrate_u(g,w,gamma,Deltax,u0,x0,resolution_density=50):
    xmax=2*abs(x0)
    resolution=int(3*abs(x0)*resolution_density)
    u_old=u0
    up_old=0
    x=np.linspace(x0,xmax,resolution)
    Dr=abs(xmax-x0)/resolution
    solution=np.zeros(len(x))
    derivative=np.zeros(len(x))
    solution[:int(resolution*Deltax/(xmax-x0))]=u0
    derivative[:int(resolution*Deltax/(xmax-x0))]=0
    for i in range(int((Deltax)*resolution/(xmax-x0)),2*int((Deltax)*resolution/(xmax-x0))):
        solution[i]=u_old
        derivative[i]=up_old
        a=F(u0,0,u_old,gamma,g,w)
        up_new=up_old+(a)*Dr
        u_new=u_old+(up_new)*Dr
        u_old=u_new
        up_old=up_new

    Di=int(Deltax*resolution/(xmax-x0))
    for i in range(2*int((Deltax)*resolution/(xmax-x0)),len(x)):
        solution[i]=u_old
        derivative[i]=up_old
        a=F(solution[i-Di],derivative[i-Di],u_old,gamma,g,w)
        up_new=up_old+(a)*Dr
        u_new=u_old+(up_new)*Dr
        u_old=u_new
        up_old=up_new

    #solution[solution<u0]=u0
    return solution,derivative


def u_profile(g,w,gamma,Deltax,u0,x0,resolution_density=50):
    xmax=2*abs(x0)
    resolution=int(3*abs(x0)*resolution_density)
    u_old=u0
    up_old=0
    x=np.linspace(x0,xmax,resolution)
    Dr=abs(xmax-x0)/resolution
    solution=np.zeros(len(x))
    derivative=np.zeros(len(x))
    solution[:int(resolution*Deltax/(xmax-x0))]=u0
    derivative[:int(resolution*Deltax/(xmax-x0))]=0
    for i in range(int((Deltax)*resolution/(xmax-x0)),2*int((Deltax)*resolution/(xmax-x0))):
        solution[i]=u_old
        derivative[i]=up_old
        a=F(u0,0,u_old,gamma,g,w)
        up_new=up_old+(a)*Dr
        u_new=u_old+(up_new)*Dr
        u_old=u_new
        up_old=up_new

    Di=int(Deltax*resolution/(xmax-x0))
    for i in range(2*int((Deltax)*resolution/(xmax-x0)),len(x)):
        solution[i]=u_old
        derivative[i]=up_old
        a=F(solution[i-Di],derivative[i-Di],u_old,gamma,g,w)
        up_new=up_old+(a)*Dr
        u_new=u_old+(up_new)*Dr
        u_old=u_new
        up_old=up_new

    mins_and_maxs=[i for i in range(len(derivative[1:])) if derivative[i-1]*derivative[i]<0]
    if len(mins_and_maxs)>1:
        solution[mins_and_maxs[1]:]=u0


    solution[solution<u0]=u0

    solution=np.roll(solution,int(len(solution)/2)-np.argmax(solution))

    return solution

def find_u0(g,w):
    u0=newton(lambda u,g,w: u-2*g*N(u)-w,w,args=(g,w))
    return u0

def find_Deltax(g,w,gamma,u0,x0,precision=0.001):
    low=0
    high=gamma+0.1

    converged=False
    count=0
    Deltax_old=0
    while not converged:
        midpoint=(high+low)/2.0
        Deltax=midpoint
        u,du=Integrate_u(g,w,gamma,Deltax,u0,x0)
        if u[-1]<u0 and np.asarray([du[int(len(du)/2):]<0]).all():
            high=midpoint
        else:
            low=midpoint

        converged=(abs(Deltax-Deltax_old)<precision) or count>200

        Deltax_old=Deltax
        count=count+1
    if count>200:
        print("Not converged!")

    return Deltax


def KS(x,x1,L):
    d=abs(x-x1)
    if d>float(L/2.0):
        d=L-d
    return np.exp(-abs(d))

def K(x,x1,gamma,L):
    d=(x-x1)
    if d>float(L/2.0):
        d=d-L
    if d<float(-L/2.0):
        d=d+L
    return np.exp(-abs(d))+gamma*np.sign(d)*np.exp(-abs(d))

def h(r,r1,v,w,gamma,L):
    k0=2.0/float(L)
    k=np.asarray(list(map(lambda x: K(x,r,gamma,L),r1)))
    Dr=float(L)/float(len(r1))
    h=sum((k-k0)*v)*Dr+w
    return h

def S(g,w,gamma,L):
    x0=-L/3.0
    #finds shape of function v
    u0=find_u0(g,w,x0)
    print("u0 found")
    Deltax=find_Deltax(g,w,gamma,u0,x0)
    print("Delta x found")
    Deltax_corrected=Deltax+0.1*Deltax
    corrected_u=u_profile(g,w,gamma,Deltax_corrected,u0,x0)
    v0=g*N(corrected_u)
    r0=np.linspace(-L/2.0,L/2.0,len(v0))
    print("profile calculated")
    r=r0
    Dr=float(L)/float(len(r0))
    hvec=list(map(lambda x: h(x,r0,v0,w,gamma,L),r))
    plot(r,hvec)
    I=sum(list(map(lambda x: M(h(x,r0,v0,w,gamma,L)),r)))*Dr
    return pow(g,2)*L*meanK2*I,r0,v0

if __name__ == "__main__":
    main()

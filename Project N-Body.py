#!/usr/bin/env python
# coding: utf-8

# In[124]:


#the project is incomplete
#lack of periodic boundary conditions
#unable to create animated 3D plots
#the force only seems to act in the x-direction... and is very small in other directions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
get_ipython().run_line_magic('matplotlib', 'inline')
#
import tensorflow as tf
#
from flowpm import linear_field, lpt_init, nbody, cic_paint
import flowpm

class Particles:
    def __init__(self,pos,v,m):
        self.pos = pos.copy()
        self.v = v.copy()
        self.m = m.copy()


dim = 3 #do not change
n = 100000
nbins = 100
dt = 10**8

#initialize positions of n particles in a cube of [-1,1],[-1,1],[-1,1]
#uniformly distributed
#each particle has a mass of 1 (don't change the mass)

pos = np.zeros([n,dim])
for i in range(n):
    pos[i] = 2*np.random.random(3)-1
v = np.random.randn(n,dim)*0
m = np.ones(n)
parts = Particles(pos,v,m)


# In[2]:


def get_rho(parts):
    H,edges = np.histogramdd(parts.pos, bins = (nbins,nbins,nbins), range = ([-1,1],[-1,1],[-1,1]))
    rho = H*((2/nbins)**3)
    points = np.zeros([dim,nbins])
    for i in range(dim):
        for j in range(len(edges[i])-1):
            if (j!=(len(edges[i])-1)):
                points[i][j] = (edges[i][j] + edges[i][j+1])/2
            else:
                points[i][j] = (edges[i][j-1]+edges[i][j])/2
    return rho, edges, points           


# In[40]:


def get_greens(nbins,points):
    #get the potential from a point mass at (0,0)
    xmat,ymat,zmat=np.meshgrid(points[0],points[1],points[2])
    dr=np.sqrt(xmat**2+ymat**2+zmat**2)
    dr[nbins/2,nbins/2,nbins/2] = 1
    pot = 1/(4*np.pi*dr)
    pot[dr<=0.001] = 1/(4*np.pi*0.001)
    #pot=pot-pot[nbins//2,nbins//2,nbins//2]  #set it so the potential at the edge goes to zero
    return pot


# In[4]:


def rho2pot(rho,greens):
    temp = rho.copy()
    temp = np.pad(temp,(nbins,nbins))
    tempft = np.fft.rfftn(temp)
    greenspad = np.pad(greens,(nbins,nbins))
    temp = np.fft.irfftn(tempft*np.fft.rfftn(greenspad))
    temp = temp[rho.shape[0]:2*rho.shape[0],rho.shape[1]:2*rho.shape[1],rho.shape[2]:2*rho.shape[2]]
    return temp


# In[130]:


def get_forces(rho):
    pot = rho2pot(rho,greens)
    force = np.gradient(pot)
    return force, pot


# In[169]:


def get_indices(parts):
    indices = np.zeros([len(parts.pos),dim])
    for i in range(len(parts.pos)):
        for j in range(dim):
            for k in range(nbins):
                if parts.pos[i][j]<=edges[j][k]:
                    indices[i][j] = k-1
                    break
    return indices.astype(int)


# In[171]:


def take_step(rho,parts):
    parts.pos = parts.pos + parts.v*dt
    f,pot = get_forces(rho)
    ind = get_indices(parts)
    for i in range(len(parts.pos)):
        for j in range(dim):
            parts.v[i][j] = parts.v[i][j] + 0.5*dt*f[j][ind[i][0]][ind[i][1]][ind[i][2]]
    parts.pos = parts.pos + dt*parts.v
    for i in range(len(parts.pos)):
        for j in range(dim):
            parts.v[i][j] = parts.v[i][j] + dt*f[j][ind[i][0]][ind[i][1]][ind[i][2]]
    


# In[320]:


#delimiter
#delimiter
#delimiter


# In[1]:


rho,edges,points = get_rho(parts)
greens = get_greens(nbins,points)
for i in range(10):
    print(i)
    fig=plt.figure(figsize=(10,10))#Create 3D axes
    try: ax=fig.add_subplot(111,projection="3d")
    except : ax=Axes3D(fig) 
    ax.scatter(parts.pos[:,0],parts.pos[:,1], parts.pos[:,2], color = 'royalblue', marker = '.',s = 0.02)
    take_step(rho,parts)
    rho, edges, points = get_rho(parts)
    


# In[192]:


fig=plt.figure(figsize=(10,10))#Create 3D axes
try: ax=fig.add_subplot(111,projection="3d")
except : ax=Axes3D(fig) 
ax.scatter(parts.pos[:,0],parts.pos[:,1], parts.pos[:,2], color = 'royalblue', marker = '.',s = 0.02)


# In[146]:


#a single particle remains at rest
pos_single = np.zeros([1,dim])
v_single = pos_single
parts_single = Particles(pos_single, v_single, [1])


# In[166]:


rho_single,edges,points = get_rho(parts_single)
greens = get_greens(nbins,points)
for i in range(10):
    print(i)
    take_step(rho_single,parts_single)
    rho_single, edges, points = get_rho(parts_single)
    


# In[149]:


fig=plt.figure(figsize=(10,10))#Create 3D axes
try: ax=fig.add_subplot(111,projection="3d")
except : ax=Axes3D(fig) 
ax.scatter(parts_single.pos[:,0],parts_single.pos[:,1], parts_single.pos[:,2], color = 'royalblue', marker = '.',s = 10)


# In[187]:


#two particles orbit
pos_orb = np.zeros([2,dim])
v_orb = pos_orb.copy()
pos_orb[0][0] = 0.01
pos_orb[1][0] = -0.01
v_orb[0][1] = 0.1*10**-9
v_orb[1][1] = -0.1*10**-9

parts_orb = Particles(pos_orb,v_orb,[1,1])


# In[188]:


rho_orb,edges,points = get_rho(parts_orb)
greens = get_greens(nbins,points)
for i in range(10):
    print(i)
    take_step(rho_orb,parts_orb)
    rho_orb, edges, points = get_rho(parts_orb)
    print(parts_orb.pos[0])
    print(parts_orb.pos[1])
    


# In[155]:


fig=plt.figure(figsize=(10,10))#Create 3D axes
try: ax=fig.add_subplot(111,projection="3d")
except : ax=Axes3D(fig) 
ax.scatter(parts_orb.pos[:,0],parts_orb.pos[:,1], parts_orb.pos[:,2], color = 'royalblue', marker = '.',s = 10)


# In[184]:


parts_orb.pos


# In[62]:


test = greens2(100)
test


# In[18]:


rho, edges, points = get_rho()
greens = get_greens([0,0,0])


# In[50]:


t = [0,0,2,2,0,0,0]


# In[47]:


x = np.linspace(-3,3,10000)

def func(x):
    if (x<=1 and x>=-1):
        return 1/np.abs(4*np.pi)
    if (x>1 or x<-1):
        return 1/np.abs(4*np.pi*x)
xx = np.zeros(10000)
for i in range(len(xx)):
    xx[i] = func(x[i])
plt.scatter(x,xx)


# In[37]:


test2 = np.zeros([3,3,3])
test2[1,1]


# In[115]:


dx


# In[28]:


dx=np.arange(nbins)
dx[nbins//2:]=dx[nbins//2:]-nbins
pot=np.zeros([nbins,nbins,nbins])
xmat,ymat,zmat=np.meshgrid(dx,dx,dx)
dr=np.sqrt(xmat**2+ymat**2+zmat**2)
#dr[0,0,0]=1 #dial something in so we don't get errors


# In[38]:


dr


# In[ ]:





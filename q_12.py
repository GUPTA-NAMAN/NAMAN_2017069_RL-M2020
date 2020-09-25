#!/usr/bin/env python
# coding: utf-8

# In[89]:


import random
import numpy
import matplotlib


# In[82]:


class Arm_set :
    count_arms = -1
    epsilon = -1
    Arms = -1
    actions = -1 
    arm = -1
    rewards = -1
    
    def __init__(self,k,esp,desc) :
        self.count_arm = k 
        self.epsilon = esp
        self.Arms = []
        self.actions = 0 
        self.arm = []
        self.rewards = []
        for i in range(k) :
            mean = desc[i][0]
            var  = desc[i][1]
            initial = desc[i][2]
            self.Arms.append( Arm(mean,var,initial) )
            self.arm.append( i )
            
    def Take_Action(self) :
        option = numpy.random.binomial(1,self.epsilon,1)[0]
        if option==1 :
            sel_arm = random.sample(self.arm,1)[0]  
        else :
            sel_arm = greedy() 
        rew = self.Arms[sel_arm].update
        self.actions = self.actions + 1
        self.rewards.append(rew)
        self.est_error() 
            
    def est_error(self) : 
        for i in range(self.count_arm) :
            self.Arms[i].cal_error() 
        
    def greedy(self) :
        index=0
        maxi =self.Arms[0].qk
        for i in range(1,count_arm) :
            if Arms[i].qk > maxi :
                maxi = Arms[i].qk 
                index = i 
        return index 
        


# In[83]:


class Arm :
    qk=-1
    var=-1
    mean=-1
    k=-1
    error= -1
    def __init__(self,a,b,c) :
        self.qk = c 
        self.var = b 
        self.mean = a 
        self.k = 0
        self.error = []
        
    def update(self) :
        self.k=self.k+1.0 
        award = Award(mean,var)
        self.qk = self.qk + (award-self.qk)/(self.k) 
        return award
    
    def Award(mean,var) :
        return numpy.random.normal(mean,var)
    
    def cal_error(self) :
        diff = abs(self.qk-self.mean) 
        self.error.append(diff)


# In[84]:


print("enter number of arm and epsilon\n")


# In[85]:


k=int(input())
ep=float(input())


# In[ ]:


desc=[]


# In[ ]:


for i in range(k) :
    temp=[]
    for j in range(3) :
        inp  = int(input())
        temp.append(inp)
    desc.append(temp)


# In[86]:


arms_set = Arm_set(k,ep,desc)


# In[87]:


tot_acts = int(input("enter number of total actions\n"))


# In[88]:


for i in range(tot_acts) :
    arms_set.Take_Action()


# In[ ]:





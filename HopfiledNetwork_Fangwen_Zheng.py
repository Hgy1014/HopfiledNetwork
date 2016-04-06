# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:57:29 2016

@author: milly
"""

import numpy as np
from random import randint
from matplotlib import pyplot as plt

#num_neuron=25; # number of neuron
#num_tpatterns= 5 # number of training patterns
tpattern_matrix= np.zeros((4,25)) # flatten training patterns matrix 4*5
# set sum as float
# single_w value of single weight 
#row_w=np.zeros(num_neuron)
#weight_matrix= np.zeros((num_neuron,num_neuron)) # weight_matrix

########### HopfiledNetwork

class HopfiledNetwork(object):
    def __init__(self, num_neuron):
        self.num_neuron=num_neuron
        self.weights=np.random.uniform(-1.0,1.0,(self.num_neuron,self.num_neuron))
    
    def run(self,noise_pattern):
        iteration_count=0
        max_iteration=10
        try_pattern=noise_pattern.copy()
        while True:
            list=range(self.num_neuron)
            #test_pattern=noise_pattern.copy()
            num=len(try_pattern)
            changed=False
            new_pattern=np.zeros(self.num_neuron) #output
            for index in list:
                #new_pattern=np.zeros(self.num_neuron) #output
                
                for j in range(num):
                
                    new_pattern[index]+=self.weights[index][j]*try_pattern[j]
                    
                if new_pattern[index]<0:
                    new_pattern[index]=-1.0
                else:
                    new_pattern[index]=1.0
                
                if new_pattern[index]!=try_pattern[index]:
                    try_pattern[index]=new_pattern[index]
                    changed=True
            #try_pattern=try_pattern
            iteration_count+=1
            if changed==False or iteration_count==max_iteration:
               
                return try_pattern
    def set_weight(self,weight):
        self.weights=weight
        
                    
            


########### calculate weight matrix

        
def calculate_single_weight(i,j,tpattern_matrix):
    num_tpatterns=len(tpattern_matrix)
    sum=0.0
    for r in range(num_tpatterns):
        sum+=tpattern_matrix[r][i]*tpattern_matrix[r][j]
    single_w=sum/float(num_tpatterns)
    return single_w
        
def calculate_row_weight(i,tpattern_matrix,num_neuron):
    row_w=np.zeros(num_neuron)
    for j in range(num_neuron):
        if i==j: continue
        row_w[j]=calculate_single_weight(i,j,tpattern_matrix)
    return row_w
    
def calculate_weight_matrix(tpattern_matrix,num_neuron):
    weight_matrix= np.zeros((num_neuron,num_neuron))
    for i in range(num_neuron):
        weight_matrix[i]=calculate_row_weight(i,tpattern_matrix,num_neuron)
    return weight_matrix
########### Traning the weights

d_pattern=np.array([[1,1,1,1,0],
                    [1,0,0,0,1],
                    [1,0,0,0,1],
                    [1,0,0,0,1],
                    [1,1,1,1,0]])

                    
j_pattern=np.array([[1,1,1,1,1],
                    [0,0,1,0,0],
                    [0,0,1,0,0],
                    [1,0,1,0,0],
                    [1,1,1,0,0]])
                    
c_pattern=np.array([[0,1,1,1,1],
                    [1,0,0,0,0],
                    [1,0,0,0,0],
                    [1,0,0,0,0],
                    [0,1,1,1,1]])
                    
m_pattern=np.array([[1,0,0,0,1],
                    [1,1,0,1,1],
                    [1,0,1,0,1],
                    [1,0,0,0,1],
                    [1,0,0,0,1]])
                    
                    
#t_pattern=np.array([[1,1,1,1,1],
                    #[0,0,1,0,0],
                    #[0,0,1,0,0],
                    #[0,0,1,0,0]])
d_pattern *=2 
d_pattern -=1

j_pattern *=2
j_pattern -=1

c_pattern *=2
c_pattern -=1

m_pattern *=2
m_pattern -=1

#t_pattern *=2
#t_pattern -=1
                    
tpattern_matrix=np.array([d_pattern.flatten(),
                          j_pattern.flatten(),
                          c_pattern.flatten(),
                          m_pattern.flatten()])
                          #t_pattern.flatten()])

#print tpattern_matrix                         
weight=calculate_weight_matrix(tpattern_matrix,25) # flatten training patterns matrix 4*5

#print weight
  #row_w[j]=single_w

######################## test add noise

network = HopfiledNetwork(25)

network.set_weight(weight)



def shuffle(test_pattern):
    for i in range(2):
        p = randint(0, 24)
        test_pattern[p]*=-1
    return test_pattern

d_shuffle=shuffle(d_pattern.flatten())  
d_result=network.run(d_shuffle)
d_result.shape=(5,5)
d_shuffle.shape=(5,5)
plt.subplot(4,4,1)
plt.imshow(d_shuffle,interpolation="nearest")
plt.subplot(4,4,2)
plt.imshow(d_result,interpolation="nearest")

j_shuffle=shuffle(j_pattern.flatten())  
j_result=network.run(j_shuffle)
j_result.shape=(5,5)
j_shuffle.shape=(5,5)
plt.subplot(4,4,3)
plt.imshow(j_shuffle,interpolation="nearest")
plt.subplot(4,4,4)
plt.imshow(j_result,interpolation="nearest")

c_shuffle=shuffle(c_pattern.flatten())  
c_result=network.run(c_shuffle)
c_result.shape=(5,5)
c_shuffle.shape=(5,5)
plt.subplot(4,4,5)
plt.imshow(c_shuffle,interpolation="nearest")
plt.subplot(4,4,6)
plt.imshow(c_result,interpolation="nearest")

m_shuffle=shuffle(m_pattern.flatten())  
m_result=network.run(m_shuffle)
m_result.shape=(5,5)
m_shuffle.shape=(5,5)
plt.subplot(4,4,7)
plt.imshow(m_shuffle,interpolation="nearest")
plt.subplot(4,4,8)
plt.imshow(m_result,interpolation="nearest")


d_shuffle=(d_shuffle+1)/2
print 'pattern D with noise'
print d_shuffle
d_result=(d_result+1)/2
print 'pattern D after running the program'
print d_result

j_shuffle=(j_shuffle+1)/2
print 'pattern J with noise'
print j_shuffle
j_result=(j_result+1)/2
print 'pattern J after running the program'
print j_result

c_shuffle=(c_shuffle+1)/2
print 'pattern C with noise'
print c_shuffle
c_result=(c_result+1)/2
print 'pattern C after running the program'
print c_result

m_shuffle=(m_shuffle+1)/2
print 'pattern M with noise'
print m_shuffle
m_result=(m_result+1)/2
print 'pattern m after running the program'
print m_result




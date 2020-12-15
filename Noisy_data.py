#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:/Users/mun/Desktop/Research/Ny mappe/GPS_track.csv') 


# In[3]:


df.head(1000)


# In[4]:


lat = np.array([df.latitude])
print(lat)


# In[5]:


long = np.array([df.longitude])
print(long)
print(len(long[0]))


# In[6]:


# calculate covariance matrix for our data which is latitude and longitude


# In[7]:


for i in range(len(long)):
    print(long[i][0])


# In[8]:


for i in range(len(lat[0])):
    print(lat[0][i])


# In[9]:


print(len(lat[0]))
print(len(long[0]))

#length of the arrays. the arrays should always have the same length
lng=len(lat[0])
print(lng)


# In[10]:


for index in range(lng):
    print(lat[0][index])
    print(long[0][index])
    


# In[11]:


for index in range (lng):
    np.array((lat[0][index], long[0][index]))


# In[12]:


coord1 = [list(i) for i in zip (lat[0],long[0])]
print(coord1)


# In[16]:


coord = list(zip(lat[0],long[0]))


# In[18]:


from pylab import *
from numpy import *
import matplotlib.pyplot as plt

class Kalman:
    def __init__(self, ndim):
        self.ndim    = ndim
        self.Sigma_x = eye(ndim)*1e-6     # Process noise (Q)
        self.A       = eye(ndim)          # Transition matrix which predict state for next time step (A)
        self.H       = eye(ndim)           # Observation matrix (H)
        self.mu_hat  = 0                    # State vector (X)
        self.cov     = eye(ndim)          # Process Covariance (P)
        self.R       = (1e-4)   # Sensor noise covariance matrix / measurement error (R)

    def update(self, obs):

        # Make prediction
        self.mu_hat_est = dot(self.A,self.mu_hat)
        self.cov_est = dot(self.A,dot(self.cov,transpose(self.A))) + self.Sigma_x

        # Update estimate
        self.error_mu = obs - dot(self.H,self.mu_hat_est)
        self.error_cov = dot(self.H,dot(self.cov,transpose(self.H))) + self.R
        self.K = dot(dot(self.cov_est,transpose(self.H)),linalg.inv(self.error_cov))
        self.mu_hat = self.mu_hat_est + dot(self.K,self.error_mu)
        if ndim>1:
            self.cov = dot((eye(self.ndim) - dot(self.K,self.H)),self.cov_est)
        else:
            self.cov = (1-self.K)*self.cov_est 

if __name__ == "__main__":		
    #print "***** 1d ***********"
    ndim = 1
    nsteps = 3
    k = Kalman(ndim)    
    mu_init=array([54.907134])
    cov_init=0.001*ones((ndim))
    obs = random.normal(mu_init,cov_init,(ndim, nsteps))
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
        print ("Actual: ", obs[:, t], "Prediction: ", k.mu_hat_est)


# In[19]:


coord_output=[]

for coordinate in coord1:
    temp_list=[]
    ndim = 2
    nsteps = 100
    k = Kalman(ndim)    
    mu_init=np.array(coordinate)
    cov_init=0.0001*ones((ndim))
    obs = zeros((ndim, nsteps))
    for t in range(nsteps):
        obs[:, t] = random.normal(mu_init,cov_init)
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
        print ("Actual: ", obs[:, t], "Prediction: ", k.mu_hat_est[0])
    temp_list.append(obs[:, t])
    temp_list.append(k.mu_hat_est[0])
        
    print("temp list")
    print(temp_list)
    coord_output.append(temp_list)


# In[20]:


for coord_pair in coord_output:
    print(coord_pair[0])
    print(coord_pair[1])
    print("--------")
print(line_actual)


# In[21]:


print(coord_output)


# In[22]:


df2= pd.DataFrame(coord_output)
print(df2)


# In[23]:


Actual = df2[0] 
Prediction = df2[1]
print (Actual)
print(Prediction)


# In[24]:


Actual_df = pd.DataFrame(Actual)
Prediction_df = pd.DataFrame(Prediction)
print(Actual_df)
print(Prediction_df)


# In[25]:


Actual_coord = pd.DataFrame(Actual_df[0].to_list(), columns = ['latitude', 'longitude'])
Actual_coord.to_csv('C:/Users/mun/Desktop/Research/Ny mappe/Actual_noise.csv')


# In[26]:


Prediction_coord = pd.DataFrame(Prediction_df[1].to_list(), columns = ['latitude', 'longitude'])
Prediction_coord.to_csv('C:/Users/mun/Desktop/Research/Ny mappe/Prediction_noise.csv')


# In[27]:


print (Actual_coord)
print (Prediction_coord)


# In[28]:


Actual_coord.plot(kind='scatter',x='longitude',y='latitude',color='red')
plt.show()


# In[29]:


Prediction_coord.plot(kind='line',x='longitude',y='latitude',color='green')
plt.show()


# In[107]:


from rdp import rdp


# In[124]:


simple_Prediction_coord_rdp = rdp(Prediction_coord[['latitude','longitude']].values, epsilon = 0.00005)
print ("{} points reduced to {}!". format(df.shape[0], simple_Prediction_coord_rdp.shape[0]))


# In[125]:


print(simple_Prediction_coord_rdp)


# In[131]:


latitude = np.array(simple_Prediction_coord_rdp[:, 0])
longitude = np.array(simple_Prediction_coord_rdp[:, 1])
simple_Prediction_coord_rdp = pd.DataFrame({'latitude': latitude, 'longitude': list(longitude)}, columns=['latitude', 'longitude'])


# In[132]:


simple_Prediction_coord_rdp.to_csv('C:/Users/mun/Desktop/Research/Ny mappe/Prediction_noise_rdp.csv')


# In[127]:


plt.plot (Prediction_coord['longitude'].values, Prediction_coord['latitude'].values)
plt.plot (Prediction_coord['longitude'].values, Prediction_coord['latitude'].values, 'ro');


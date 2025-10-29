#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling,Dense,Dropout,Conv2D,Conv2DTranspose,MaxPool2D,Flatten,Input,Reshape
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential,Model
from sklearn.manifold import TSNE


#Model

class vae(Model):
    
    def __init__(self,img_shape,latdim = 64):
        super().__init__()

        self.encoder = Sequential([
            Input(shape = img_shape),
            Conv2D(kernel_size = 16,
                   filters = 16,
                   kernel_initializer = VarianceScaling(),
                   strides = 4,
                   activation = relu,
                  ),                    #103x103x16
            MaxPool2D(pool_size = (5,5),
                         strides = 2,
                
            ),                          #50x50x16
            Conv2D(kernel_size = 4,
                   filters = 16,
                   kernel_initializer = VarianceScaling(),
                   strides = 2,
                   activation = relu,
            ),                          #24x24x16
            MaxPool2D(pool_size = (2,2),
                         strides = 2,
                        ),               #12x12x16
            Flatten(),                  # flatten to 12x12x16 = 2034 pixels
            Dense(512,activation = 'relu'),
            Dense(256,activation = 'relu'),
            Dense(128,activation = 'relu'), 
            
        ])
        
        self.mu = Dense(latdim,activation = 'linear')

        self.logvar = Dense(latdim,activation = 'linear')
        
        self.decoder = Sequential([
            Dense(2304,activation = 'relu'), # 64 --> 2034
            Reshape((12,12,16)),             # 2304 --> 16 fms of 12x12 size.
            Conv2DTranspose(filters = 16,
                            kernel_size = 2,
                            strides = 2,
                            activation = 'relu', 
                            ),                    #24x24x16
            Conv2DTranspose(filters = 16,
                            kernel_size = 4,
                            strides = 2,
                            activation = 'relu', 
                            ),                   #50x50x16
            Conv2DTranspose(filters = 16,
                            kernel_size = 5,
                            strides = 2,
                            activation = 'relu', 
                            ),                   #103x103x16
            Conv2DTranspose(filters = 3,
                            kernel_size = 16,
                            strides = 4,
                            activation = 'relu', 
                            ),                   #424x424x3 (RGB)
        ])


    def rep_trick(self,mu,logvar):
        
        eps = tf.random.normal(tf.shape(mu),0.0,1.0)
        std = tf.math.exp(0.5*logvar)
        z = mu + eps*std
        return z
        
    def encoder_out(self,X):
        
        enc = self.encoder(X)
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        z = self.rep_trick(mu,logvar)
            
        return mu,logvar,z

    def decoder_out(self,z):
            
        back = self.decoder(z)

        return back
            
 
    @tf.function
    def loss(self,X,Xhat,mu,logvar,klw,recw):

        kl_loss = logvar - tf.math.exp(logvar) - tf.square(mu) + 1
        kl_loss = -0.5*tf.reduce_sum(kl_loss,axis = 1)               #scaling after computation
        batch_kl_loss = tf.reduce_mean(kl_loss)

        
        #MSE
        
        rec_loss = tf.square(X - Xhat)
        batch_rec_loss = tf.reduce_sum(rec_loss,axis = [1,2,3])
        per_img_rec_loss = tf.reduce_mean(batch_rec_loss)

        
        total_loss = klw * batch_kl_loss + recw *per_img_rec_loss

        return total_loss

    @tf.function
    def validate(self,val,klw,recw):

        val_mu,val_logvar,val_z = self.encoder_out(val)
        val_predicted = self.decoder_out(val_z)
        val_loss = self.loss(val,val_predicted,val_mu,val_logvar,klw,recw)

        return val_loss

    def get_latent_space(self,X):
        _,_,z = self.encoder_out(X)
        return z
        
        
    
    @tf.function
    def backprop(self,X,klw,recw,optimizer):

        with tf.GradientTape() as tape:
        
            mu,logvar,z = self.encoder_out(X)
            Xhat = self.decoder_out(z)
            loss = self.loss(X,Xhat,mu,logvar,klw,recw)

        grad = tape.gradient(loss,self.trainable_variables)
        optimizer.apply_gradients(zip(grad,self.trainable_variables))
    
        return loss








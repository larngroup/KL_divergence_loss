import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import time
import sys


class Kl_annealing_loss(tf.keras.layers.Layer):
    def __init__(self,beta,**kwargs):
        super().__init__(**kwargs)
        self.beta=beta

    def call(self,inputs):

        mean=inputs[0]
        var=inputs[1]
        kl_loss= var - tf.square(mean) - tf.exp(var)
        kl_loss_batch=-0.5*tf.reduce_mean(kl_loss)
        out=tf.math.multiply(self.beta,kl_loss_batch)
        return out
      
    def get_config(self):
      config=super().get_config()
      return config

class My_cross_entropy(tf.keras.layers.Layer):
    def __init__(self,name="Loss layer",**kwargs):
        super().__init__(name=name)

    def call(self,predict,ground):
        loss_cross= tf.keras.losses.CategoricalCrossentropy(name="My_CategoricalCrossentropy")
        m=loss_cross(ground,predict)

        return m





class AnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self,beta,name,total_epochs):
        self.beta=beta
        self.name=name
        self.total_epochs=total_epochs
    
    def on_epoch_end(self,epoch,logs={}):
      
        R=1.5
        if self.name=="normal":
            pass
        elif self.name=="monotonic":
            
            new_value=epoch/float(self.total_epochs)*R
            if new_value>1:
                new_value=1
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))
        elif self.name=="cyclical":
            T=self.total_epochs
            M=5
            frac=int(self.total_epochs/M)
            tt=((epoch)%frac)/float(frac)
            
            new_value=tt
            if new_value>1:
                new_value=1
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))



class Annealing_model(Model):

    def __init__(self,type_annealing,latent_space,inputs,outputs,name="Variational Autoencoder"):
        Model.__init__(self,inputs=inputs,outputs=outputs,name=name)
        self.type_annealing=type_annealing
        if self.type_annealing=="normal":
            self.beta_weight=1.0
        else:
            self.beta_weight=0.0
        self.beta_var=tf.Variable(self.beta_weight,trainable=False,name="Beta_annealing",validate_shape=False)

        #Generate Losses
        kl_loss=Kl_annealing_loss(self.beta_var)(latent_space)

        self.add_loss(kl_loss)
        self.add_metric(kl_loss,name="KL Diverg. Loss",aggregation="mean")


    def fit(self,*args,**kwargs):
        total_epochs=kwargs["epochs"]

        my_callback=AnnealingCallback(self.beta_var,self.type_annealing,total_epochs)

        callbacks_list=kwargs["callbacks"]
        callbacks_list.append(my_callback)
        kwargs["callbacks"]=callbacks_list

        return super(Annealing_model, self).fit(*args,**kwargs)

    def compile(self,*args,**kwargs):
        
        loss_calculator=tf.keras.losses.CategoricalCrossentropy(name="My_CategoricalCrossentropy")
        
        my_metric = tf.keras.metrics.CategoricalCrossentropy()
        return super(Annealing_model, self).compile(*args,**kwargs,loss=loss_calculator,metrics=[my_metric])
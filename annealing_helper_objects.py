import tensorflow as tf
from tensorflow import keras


class Kl_annealing_loss(tf.keras.layers.Layer):
    def __init__(self,beta,name="loss_layer",**kwargs):
        super().__init__(name=name)
        self.beta=beta

    def call(self,inputs):

        mean=inputs[0]
        var=inputs[1]
        kl_loss= var - tf.square(mean) - tf.exp(var)
        kl_loss_batch=-0.5*tf.reduce_mean(kl_loss)
        out=tf.math.multiply(self.beta,kl_loss_batch)
        return out


class My_MSE(keras.metrics.Metric):
    def __init__(self, name="custom_mse", **kwargs):
        super(My_MSE, self).__init__(name=name, **kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        reconstruction_loss=tf.losses.mse(y_true,y_pred)
        self.reconstruction_loss_batch=tf.reduce_mean(reconstruction_loss)
           
        
    def result(self):
        return self.reconstruction_loss_batch



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
            M=4
            frac=int(self.total_epochs/M)
            tt=((epoch+1)%frac)/float(frac)
            
            new_value=tt
            if new_value>1:
                new_value=1
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))
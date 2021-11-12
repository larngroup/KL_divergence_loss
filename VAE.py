
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate,LSTM, Bidirectional, Dense, Input, GaussianNoise, BatchNormalization, RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
tf.compat.v1.disable_eager_execution()
import math

from annealing_helper_objects import Kl_annealing_loss,My_MSE,AnnealingCallback



class VAE_v1:

    def __init__(self,input_size,latent_dim,dense_units1,dense_units2,annealing_mode):

        self.input_size=input_size
        self.latent_dim=latent_dim
        self.dense_units1=dense_units1
        self.dense_units2=dense_units2
        self.annealing_mode=annealing_mode

        if self.annealing_mode=="normal":
            self.beta_weight=1.0
        else:
            self.beta_weight=0.0
        self.beta_var=tf.Variable(self.beta_weight,trainable=False,name="Beta_annealing",validate_shape=False)

        #Build models
        self.build_mean_encoder()
        self.build_std_encoder()
        self.build_decoder()
        self.build_model()



    def build_mean_encoder(self):

        input_data=Input(shape=(self.input_size,),name="Input_data_mean")
        some_output=Dense(self.dense_units2,activation="relu",name="dense_1_mean")(input_data)
        some_output=Dense(self.dense_units1,activation="relu",name="dense_2_mean")(some_output)
        some_output=Dense(self.latent_dim,activation="relu",name="dense_3_mean")(some_output)
        self.mean_encoder_model=Model(inputs=input_data,outputs=some_output,name="mean_encoder_model")

    def build_std_encoder(self):

        input_data=Input(shape=(self.input_size,),name="Input_data_std")
        some_output=Dense(self.dense_units2,activation="relu",name="dense_1_std")(input_data)
        some_output=Dense(self.dense_units1,activation="relu",name="dense_2_std")(some_output)
        some_output=Dense(self.latent_dim,activation="relu",name="dense_3_std")(some_output)
        self.std_encoder_model=Model(inputs=input_data,outputs=some_output,name="std_encoder_model")

    def build_decoder(self):

        latent_input=Input(shape=(self.latent_dim,),name="latent_input")
        some_output=Dense(self.dense_units1,activation="relu",name="dense_1_all")(latent_input)
        some_output=Dense(self.dense_units2,activation="relu",name="dense_2_all")(some_output)
        some_output=Dense(self.input_size,activation="relu",name="dense_3_all")(some_output)
        self.decoder_model=Model(inputs=latent_input,outputs=some_output,name="decoder_model")

    def _sample_latent_features(self,distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tf.shape(distribution_variance)[0]
        random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
        return distribution_mean + tf.exp(0.5 * distribution_variance) * random



    def build_model(self):

        input_data=Input(shape=(self.input_size,),name="Input_data_whole")

        self.mean_distribution=self.mean_encoder_model(input_data)
        self.std_distribution=self.std_encoder_model(input_data)
        self.distribution =  [self.mean_distribution, self.std_distribution]
        
        kl_loss=Kl_annealing_loss(self.beta_var)(self.distribution)
        
        latent_encoding = tf.keras.layers.Lambda(self._sample_latent_features)(self.distribution)
        some_output=self.decoder_model(latent_encoding)
        
        self.my_model=Model(inputs=input_data,outputs=some_output,name="my_model")

        self.my_model.add_loss(kl_loss)
        self.my_model.add_metric(kl_loss,name="Kl_divergence",aggregation="mean")


    def _get_loss(self):

        def my_loss(datax,datay):

            reconstruction_loss=tf.losses.mse(datax,datay)
            reconstruction_loss_batch=tf.reduce_mean(reconstruction_loss)

            return reconstruction_loss_batch

        return my_loss



    def fit_model(self,x,y,epochs,batch_size,optimizer):
       
        self.epochs=epochs
        self.batch_size=batch_size
        

        #Choosing optimizer
        if optimizer=="adam":
            self.optimizer=Adam(learning_rate=0.001)
        elif optimizer=="adam_clip":
            self.optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,amsgrad=False,clipvalue=3)


        #Callbacks and Metrics
        my_callback=AnnealingCallback(self.beta_var,self.annealing_mode,self.epochs)
        loss_tracker=keras.metrics.Mean(name="Reconstruction_loss")
        callbacks_list=[my_callback]
        my_metric=[loss_tracker]

        self.my_model.compile(optimizer=self.optimizer,loss=self._get_loss(),metrics=["mse"])
        
        results=self.my_model.fit(x,y,epochs=self.epochs,batch_size=batch_size,validation_split=0.1,shuffle=True,verbose=1,callbacks=callbacks_list)
        self.my_model.save_weights("./wb/VAE_weigths")

        #Plotting

        fig=plt.figure(1)
        ax1=fig.add_subplot(111)
        
        ax1.plot(results.history["loss"],label="Total loss")
        ax1.plot(results.history["mse"],label="Reconstruction error")
        ax1.plot(results.history["Kl_divergence"],label="Kl divergence loss")
        ax1.legend()
        ax1.set(xlabel="epochs",ylabel="loss")
        plt.title("Training")
        figure_path="plots/"+"loss_plot_train_"+self.annealing_mode+str(x.shape[0])+".png"
        fig.savefig(figure_path)
        plt.grid()

        fig=plt.figure(2)
        ax1=fig.add_subplot(111)
        
        ax1.plot(results.history["val_loss"],label="Total loss")
        ax1.plot(results.history["val_mse"],label="Reconstruction error")
        ax1.plot(results.history["val_Kl_divergence"],label="Kl divergence loss")
        ax1.legend()
        ax1.set(xlabel="epochs",ylabel="loss")
        plt.title("Validation")
        figure_path="plots/"+"loss_plot_val_"+self.annealing_mode+str(x.shape[0])+".png"
        fig.savefig(figure_path)
        plt.grid()
        plt.show()
    








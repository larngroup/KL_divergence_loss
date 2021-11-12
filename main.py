
from VAE import VAE_v1
import pandas as pd
import tensorflow as tf
import mnist
import numpy as np
import time 

if __name__ == '__main__':

    #help(mnist)
    images=mnist.train_images()

    images=images[0:3000]
    img=np.asarray(images)

    img=img.reshape(100,-1)/256.0
    img=img.astype("float32")

    
    #input_size,latent_dim,dense_units1,dense_unit2
    input_size=img.shape[1]
    latent_dim=100
    dense_dim1=256
    dense_dim2=512
    epochs=200
    batch_size=64
    
    #Name can be: normal, monotonic, cyclical
    name="cyclical"

    my_model=VAE_v1(input_size,latent_dim,dense_dim1,dense_dim2,name)
    my_model.fit_model(img,img,epochs,batch_size,"adam")
    #y_train=my_data["Survived"]
    #x_train=my_data.loc[:,my_data.columns !="Survived"]
    #print(x_train)

    



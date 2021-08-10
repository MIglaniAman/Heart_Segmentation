"""# defining model"""


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D,Conv2DTranspose, Concatenate,Input
from tensorflow.keras.models import  Model

#def funtion for Creating convolution block 
def conv_block(input, num_filters):
    x=Conv2D(num_filters,3,padding='same')(input)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    x=Conv2D(num_filters,3,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    return x

#def fun for encoder_block
def encoder_block(input,num_filters):
    x=conv_block(input,num_filters)
    p=MaxPool2D((2,2))(x)
    return x,p  #x act as an skipconnection and p is a output feature for next block

def decoder_block(input,skip_features,num_filters):
    #applying the transpose conv2d on b1
    x=Conv2DTranspose(num_filters,(2,2),strides=2,padding="same")(input)
    x=Concatenate()([x,skip_features]) #concatenation skip_connections from previous output and after convolution transpose 
    x=conv_block(x,num_filters)
    return x
        
    

#def fun for building model
def unet_model(input_shape):
    inputs=Input(input_shape)
    
    #4 encoder blocks of Unet_model
    s1,p1=encoder_block(inputs,64)  #s1,s2,s3,s4 are skip_connection
    s2,p2=encoder_block(p1,128)
    s3,p3=encoder_block(p2,256)
    s4,p4=encoder_block(p3,512)
    
    #bridge and the bottlneck part of the structure
    b1=conv_block(p4,1024)
    
    #print(s1.shape,s2.shape,s3.shape,s4.shape) chechking for proper skip_connection
    d1= decoder_block(b1,s4,512)
    d2=decoder_block(d1,s3,256)
    d3=decoder_block(d2,s2,128)
    d4=decoder_block(d3,s1,64)
    
    #output layers of 1X1 conv layer with sigmoid fuction
    outputs= Conv2D(1,1,padding="same",activation="sigmoid")(d4)
    
    model=Model(inputs,outputs,name="U-net")
    return model
    
        
    
    
    
if __name__=="__main__":
    input_shape=(512,512,3)
    model=unet_model(input_shape)
    model.summary()
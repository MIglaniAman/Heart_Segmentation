#importing important library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau,EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.metrics import Recall, Precision
from model import unet_model
from metrics import dice_loss, dice_coef, iou

H=512
W=512
#creating a dir function for checking the path
def create_dir(path):
    #"create a directory"
    if not os.path.exists(path):
        os.makedirs(path)
        
def shuffling(x,y):
    x,y=shuffle(x,y,random_state=42)
    return x,y

def load_data(path):
    x=sorted(glob(os.path.join(path,"image","*.jpg")))
    y=sorted(glob(os.path.join(path,"mask","*.jpg")))
    return x,y

#def funtion reading the images
def read_image(path):
    path= path.decode()
    x=cv2.imread(path,cv2.IMREAD_COLOR)
    x=x/255.0
    x=x.astype(np.float32)
    return x

#def function reading the mask
def read_mask(path):
    path= path.decode()
    x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x=x/255.0
    x=x>0.5   #threshold for creating image to 0 and 1
    x = x.astype(np.float32)
    x= np.expand_dims(x,axis=-1)
    return x

#def tf_parse for building data pipeline
def tf_parse(x,y):
    def _parse(x,y):
        x=read_image(x)
        y=read_mask(y)
        return x,y
    x,y=tf.numpy_function(_parse,[x,y],[tf.float32,tf.float32])  #using tf.numpy_function since we have used cv2.funtion outside the tf 
    x.set_shape([H,W,3])
    y.set_shape([H,W,1])
    return x,y


#last main function for buuilding data pipeline
def tf_dataset(x,y,batch=8):
    dataset= tf.data.Dataset.from_tensor_slices((x,y)) #this from_tensor_slices will give individuall image and mask path 
    dataset= dataset.map(tf_parse)                     #to tf_prarse funtion
    dataset= dataset.batch(batch)                      #not it willcreate a batch
    dataset= dataset.prefetch(10)                      #it will fetch some of the batch in the memory
    return dataset




if __name__=="__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """creating dir for storing files"""
    create_dir("files")
    
    
    """defining Hyperparameter"""
    batch_size=2
    lr=1e-4
    num_epochs=5
    model_path=os.path.join("files","model.h5")
    csv_path=os.path.join("files","data.csv")
    
    """Dataset_path"""
    dataset_path=os.path.join("new_data")
    train_path=os.path.join(dataset_path,"train")
    valid_path=os.path.join(dataset_path,"valid")
    
    
    train_x,train_y=load_data(train_path) #loading the training data
    train_x,train_y=shuffling(train_x,train_y) #shufflnig the traininng data
    valid_x,valid_y=load_data(valid_path)
    
    print("train:",len(train_x),"-",len(train_y))
    print("train:",len(valid_x),"-",len(valid_y))
    
    train_dataset= tf_dataset(train_x,train_y,batch=batch_size)
    valid_dataset= tf_dataset(valid_x,valid_y,batch=batch_size)
    
    #building the model
    model=unet_model((H,W,3))
    metrics=[dice_coef,iou,Recall(),Precision()]
    model.compile(loss=dice_loss,optimizer=Adam(lr), metrics=metrics )
    
    
    callbacks=[
        ModelCheckpoint(model_path,verbose=1,save_best_only=True),  #for saving model weight file
        ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,min_lr=1e-7,verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss',patience=50,restore_best_weights=False)
        
    ]
    
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_path,
        callbacks=callbacks,
        shuffle=False
    )
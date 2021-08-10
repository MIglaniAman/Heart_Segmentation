#Prediction
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"
import numpy as np
import cv2
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import  accuracy_score, f1_score, jaccard_score, precision_score,recall_score
from metrics import dice_loss, dice_coef, iou

#creating directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__=="__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """Directory for storing files"""
    create_dir("tests_result")
    
    """Loading model"""
    with CustomObjectScope({'iou': iou, 'dice_coef':dice_coef, "dice_loss": dice_loss}):
        model=tf.keras.models.load_model("files/model.h5")
        #model.summary()

    """Load the test dataset"""
    test_x=glob("data/test/*/*/*.dcm")
    print("testdata:",len(test_x))
    
    
    """Loop over the data"""
    for x in tqdm(test_x):
        """Extract the names"""
        dir_names=x.split("/")[-3]
        name=dir_names+"_"+x.split("/")[-1].split(".")[0]
        
        """Read the .dcm images"""
        images=dicom.dcmread(x).pixel_array
        #print(np.max(image))  #max pixel value is 2000 
        
        """Convertion the image  pixel b/t 0-255"""
        image=np.expand_dims(images,axis=-1)
        image=image/np.max(image)*255.0
        x=image/255.0   #since model the image btw 0 and 1
        x=np.concatenate([x,x,x],axis=-1)
        x=np.expand_dims(x,axis=0)
        
        
        """Doing prediction on test data """
        mask=model.predict(x)[0]
        mask=mask>0.5
        mask=mask.astype(np.int32)
        mask=mask*255
        """conacting image mask and pred_mask"""
        concatenate_images=np.concatenate([image,line,mask],axis=-1)
        cv2.imwrite(f"tests_result/{name}.png",concatenate_images)
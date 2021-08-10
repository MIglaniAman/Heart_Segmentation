#evaluation
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import  accuracy_score, f1_score, jaccard_score, precision_score,recall_score
from metrics import dice_loss, dice_coef, iou
from train import  load_data

H=512
W=512

#creating directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_results(image,mask,y_pred,save_image_path):
    line= np.ones((H,10,3))*128
    
    """expanding dimension for Mask"""
    mask= np.expand_dims(mask,axis=-1)  #now the size will be (512,512,1)
    mask=np.concatenate([mask,mask,mask],axis=-1) #concating 3 times bcz our image is in 3-dim sonow maskshape is (512,512,3)
                                                # now the image and mask and pred_mask are in same size and we can concatenate the easily
    
    """expanding dimension for PredMask"""
    y_pred= np.expand_dims(y_pred,axis=-1)  #now the size will be (512,512,1)
    y_pred=np.concatenate([y_pred,y_pred,y_pred],axis=-1) #concating 3 times bcz our image is in 3-dim sonow maskshape is (512,512,3)
    y_pred=y_pred*255
    
    """conacting image mask and pred_mask"""
    concatenate_images=np.concatenate([image,line,mask,line,y_pred],axis=-1)
    cv2.imwrite(save_image_path,concatenate_images)
    

        
if __name__=="__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """Directory for storing files"""
    create_dir("results")
    
    """Loading model"""
    with CustomObjectScope({'iou': iou, 'dice_coef':dice_coef, "dice_loss": dice_loss}):
        model=tf.keras.models.load_model("files/model.h5")
        #model.summary()
    
    """Load the dataset"""
    test_x=sorted(glob(os.path.join("new data","valid","image","*")))
    test_y=sorted(glob(os.path.join("new data","valid","mask","*")))
    print(f"Test: {len(test_x)}-{len(test_y)}")
    
    """Evaluation and prediction """
    score=[]
    for x,y in tqdm(zip(test_x,test_y),total=len(test_x)):
        """Extract the name"""
        name=x.split("/")[-1].split(".")[0]
        
        "reading the image"
        image=cv2.imread(x,cv2.IMREAD_COLOR)
        x=image/255.0
        x=np.expand_dims(x,axis=0)
        
        """Reading the mask"""
        mask=cv2.imread(y,cv2.IMREAD_GRAYSCALE)
        y= mask/255.0
        y=y>0.5
        y=y.astype(np.int32)
        
        
        """Know doingPredicting"""
        y_pred=model.predict(x)[0]   #model will take "x" bcz it is the batch size of 1  
        y_pred=np.squeeze(y_pred,axis=-1) #it will sequeeze on the last axis and it will ocnverted into H,W of 512 ,512
        y_pred= y_pred>0.5
        y_pred=y_pred.astype(np.int32)
        
        
        """ savning the Prediction"""
        save_image_path=f"results/{name}.png"
        save_results(image,mask,y_pred,save_image_path) #saveresuts take 4 thing: "origna_image","original_maks", "predict_mask", "save_path"
        
            
        #Now working on metrics
        """Flatten the array"""
        y=y.flatten()
        y_pred=y_pred.flatten()

        """Calculating the metrics values"""
        acc_value=accuracy_score(y,y_pred)
        f1_value=f1_score(y,y_pred,labels=[0,1],average,='binary',zero_division=1)
        jac_vaue=jaccard_score(y,y_pred,labels=[0,1],average,='binary',zero_division=1)
        recall_value= recall_score(y,y_pred,labels=[0,1],average,='binary',zero_division=1)
        precision_value=precision_score(y,y_pred,labels=[0,1],average,='binary',zero_division=1)
        score.append([name,acc_value,f1_value,jac_vaue,recall_value,precision_value])
    
    """Meterics value"""
    score=[s[1:]for s in score]
    score= np.mean(score,axis=0)
    print(f"Accuracy:{score[0]:0.5f}")
    print(f"F1:{score[1]:0.5f}")
    print(f"Jaccard:{score[2]:0.5f}")
    print(f"Recall:{score[3]:0.5f}")
    print(f"Precision:{score[4]:0.5f}")

    df=pd.DataFrame(score,columns=["Accuracy","F1", "Jaccard","Recall","Precision"])
    df.to_csv("files/score.csv")
        
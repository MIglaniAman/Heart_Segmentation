import os #using this library to join this path 
import numpy as np
import cv2
from glob import glob 
from tqdm import  tqdm
from sklearn.model_selection import  train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate


#creating a dir function for checking the path
def create_dir(path):
    #"create a directory"
    if not os.path.exists(path):
        os.makedirs(path)
        
#function for loading data
def load_data(path,split=0.2):
    #laod the images and mask
    images = sorted(glob(f"{path}/*/image/*.png"))
    masks  = sorted(glob(f"{path}/*/mask/*.png"))
    #print(len(images),len(masks))
    """Splitting the data"""
    split_size=int(len(images)*split)  # how many we want the images we want in validation set
    
    #random state in train_x and train_y should be same
    train_x,valid_x=train_test_split(images,test_size=split_size,random_state=42)
    train_y,valid_y=train_test_split(masks ,test_size=split_size,random_state=42)
    return (train_x,train_y),(valid_x,valid_y)


#Creating augmented funtion
def augment_data(images, masks, save_path, augment=True):
    #""" Performing data augmentation. """
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        dir_name = x.split("/")[-3]  #Extracting Directory name
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]  #now the name of the image is "folder_name+image_name"
        #"""Read the image and mask"""
        x= cv2.imread(x,cv2.IMREAD_COLOR)
        y= cv2.imread(y,cv2.IMREAD_COLOR)

        if augment == True:
            aug=HorizontalFlip(p=1.0)
            augmented=aug(image=x,mask=y)
            x1=augmented["image"]
            y1=augmented['mask']
            
            aug=VerticalFlip(p=1)
            augmented=aug(image=x,mask=y)
            x2=augmented["image"]
            y2=augmented['mask']
            
            aug=Rotate(limit=45,p=1.0)
            augmented=aug(image=x,mask=y)
            x3=augmented["image"]
            y3=augmented['mask']
            
            #appending the dataset after augmentation
            X = [x,x1,x2,x3]
            Y = [y,y1,y2,y3]
            
        else:
            X = [x]
            Y = [y]
            
        idx=0
        for i, m in zip(X, Y): #i and m are image and mask respectively
            #"""Now resiziing the image and mask"""
            i=cv2.resize(i, (W,H))
            m=cv2.resize(m, (W,H))
            m=m/255.0
            m=(m>0.5)*255  #value in the mask is b/w 0-255
            
            #saving images and mask  
            if len(X)==1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name  = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{idx}.jpg"
                tmp_mask_name  = f"{name}_{idx}.jpg"
                
            image_path=os.path.join(save_path,"image/",tmp_image_name)
            mask_path=os.path.join(save_path,"mask/",tmp_mask_name)
            cv2.imwrite(image_path,i)
            cv2.imwrite(mask_path,m)
            idx+=1



if __name__ == "__main__":
    """ Load the dataset """
    dataset_path = os.path.join("data", "train")
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.2)

    print("Train: ", len(train_x))
    print("Valid: ", len(valid_x))

    #creating dir for  saving data augumentation data    
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/valid/image/")
    create_dir("new_data/valid/mask/")

    #applying data augmentation(HorizontalFlip, VerticalFlip, Rotate) only in train data 
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(valid_x, valid_y, "new_data/valid/", augment=False)
    
    
import os 
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json



# funtion to create directory if the path doesnot exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#writing a funtion to creat image and mask
def process_data(image_path,json_path,save_dir):
    #first opening json file 
    f=open(json_path,"r")  #just reading the file
    data= json.load(f)  #we have loaded the and its would be in dictionary 
    #print(data)

    #looping over json file
    for key, value in tqdm(data.items()):
        #print(key,value)

        #Getting the filename
        filename=value['filename']

        # Extracting the name of the image, by removing its extension
        name=filename.split(".")[0]

        #Reading the imag
        image=cv2.imread(f"{image_path}/{filename}",cv2.IMREAD_GRAYSCALE)
        H,W=image.shape

        #Extracting information about the annotated regions
        regions=value["regions"]
        #print(region)

        #in some region it would be blank as some the image don't have heart
        # for that we give empty mask 
        if len(regions)==0:  #there is no region so mask should be blank
            mask= np.zeros((H,W))
        else:
            mask= np.zeros((H,W))
            for region in regions: #extracting the bounding elipse
                cx= int(region["shape_attributes"]["cx"])
                cy= int(region["shape_attributes"]["cy"])
                rx= int(region["shape_attributes"]["rx"])
                ry= int(region["shape_attributes"]["ry"])
                center_coordinates=(cx,cy)
                axes_length=(rx,ry)
                angle=0
                start_angle=0
                end_angle=360
                color=(255,255,255)
                thickness=-1  #-1 indicates the entire elipse with specific color
                mask=cv2.ellipse(
                    mask,center_coordinates,axes_length,angle,
                    start_angle,end_angle,color,thickness
                    )
        """ Saving the image and mask """
        cv2.imwrite(f"{save_dir}/image/{name}.png",image)
        cv2.imwrite(f"{save_dir}/mask/{name}.png",mask)   



            #print(regions) # we will get a list here for  images and it will be on since
            # there is is only one annotation

        

if __name__=="__main__":
    """Dataset part"""
    # since the structure inside this folder  is same,
    #so we areusing *  and loop again and agian and use glob funtion to load dataset
    dataset_path="dataset"
    dataset=glob(os.path.join(dataset_path,"*"))
    #print(dataset) #to see the folder inside our dataset folder
    
    # NOw we will loop inside the folder 
    for data in dataset:
        #now we need image path
        #using glob funtion which easy to use as  we only want to give extension in jpg
        #if you see in all the folder we have the jpg image & the folder name is starting with "   jpg"
        image_path=glob(os.path.join(data,"*","jpg*"))[0]  
        #"[0]" bcz image path  will return a list  and
        # # we want only 1st element you can check before adding[0]
        #print(image_path)

        json_path=glob(os.path.join(data,"*","*.json"))[0]
        #print(json_path)


        # now we will save image for each patient in seperate subfolder with same name
        dir_name=data.split("\\")[1]
        #print(dir_name)
        #now we will create the folder name "data: which will contian "mask folder and image folder"
        #and in the folder we will save our dataset
        save_dir=f"data/{dir_name}/"
        create_dir(f"{save_dir}/image")
        create_dir(f"{save_dir}/mask")

        process_data(image_path,json_path,save_dir)
        





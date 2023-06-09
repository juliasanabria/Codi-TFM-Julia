import cv2
import numpy as np
from skimage.segmentation import clear_border
import pandas as pd
import os



#____________________________________________PRE-PROCESS_____________________________________________________________
input_images_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Imatges'
files_names = os.listdir(input_images_path)
files_names = sorted(files_names, key=lambda x: int("".join([i for i in x if i.isdigit()]))) 
rows = []


for file_name in files_names: 
    image_path = input_images_path + '/' + file_name


    ################# READ IAMGE + CONVERT  TO GRAY SCALE ################
    img = cv2.imread(image_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.subtract(255,gray)
    ######################################################################

    ##################### REMOVE BORDER AROUND IMAGE #####################
    thresh = 30
    maxval = 255
    im_bin = (gray > thresh) * maxval
    im_bin = np.array(im_bin)
    cleared_border = clear_border(im_bin)
    cleared_border = cleared_border.astype(np.uint8)
    ######################################################################

    ######################################################################
    ret,thresh = cv2.threshold(cleared_border,5,255,cv2.THRESH_TOZERO)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,11)) ## (11,11)
    kernel2 = np.ones((2,2),np.uint8)
    erosion = cv2.erode(thresh,kernel2,iterations = 1)
    dilation = cv2.dilate(erosion,kernel1,iterations = 1)
    ######################################################################

    ########################### WITH HOLES ###############################
    im_floodfill = dilation.copy()
    h, w = dilation.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = dilation | im_floodfill_inv
    ######################################################################

    ######################################################################
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 1500000
    mask1 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask1[output == i + 1] = 255

    mask1 = mask1.astype(np.uint8)

    res = cv2.bitwise_and(img,img,mask = mask1)
    mask2 = cv2.bitwise_not(mask1)
    img_WON1 = cv2.bitwise_not(res,res, mask = mask2) # img_WON = img without noise 
    ######################################################################
#______________________________________________________________________________________________________________________________________


#________________________________________________PROCESS______________________________________________________________________
    ######################################################################
    gray_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    retval, tresh = cv2.threshold(gray_img, 70, 255, 0)
    img_contours, _ = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_WON1, img_contours, -1, (0, 255, 0))
    ######################################################################

    r1=[]

    for i in range(1,len(img_contours)):
        ar1 = cv2.contourArea(img_contours[i])
        r1.append(ar1)

    maskcontours = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(res, maskcontours, -1, (0, 0, 255),4)
    area_punch = cv2.contourArea(maskcontours[0])
    area_marro = sum(r1)
    percentatge_marro = area_marro/area_punch*100
#______________________________________________________________________________________________________________________________________


#_____________________________________________________________RESULTS_________________________________________________________________________
    rows.append([file_name,area_marro,area_punch, percentatge_marro])

    cv2.imwrite('/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Imatges_Resultants/'+file_name, res)

    df_info = pd.DataFrame(rows)
    print("fi imatge", file_name)
    
titols = ['Imatge', 'Àrea Marró', 'Àrea Punch', '% Marró/Punch']

print("#######################################################################")

ex = df_info.to_excel(r'/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats/Resultats.xlsx', index = False, header=titols)
#______________________________________________________________________________________________________________________________________

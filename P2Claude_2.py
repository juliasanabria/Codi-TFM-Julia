#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Màster en Bioinformàtica i Bioestadística
@author: juliasanabriafranquesa
2023

"""



import numpy as np
import os 
from PIL import Image, ImageDraw
import time

start_time = time.time()

input_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves'  
output_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2Claude'

for filename in os.listdir(input_dir):    
    if filename.endswith(".tif"):   
        img = Image.open(os.path.join(input_dir,filename)).convert("RGB")
        
        # Extreure nuclis marrons
        img_np = np.array(img)
        binary = img_np[:,:,0] < 150 
       
        # Resseguir en verd     
        draw = ImageDraw.Draw(img) 
        for y in range(img.size[1]):
          for x in range(img.size[0]):
             if binary[y,x] == True:
                 draw.point((x,y),'green')  
                
        # Exportar       
        img.save(os.path.join(output_dir,filename))
        



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")

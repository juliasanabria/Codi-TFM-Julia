#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Màster en Bioinformàtica i Bioestadística
@author: juliasanabriafranquesa
2023
"""

import os
from PIL import Image
import numpy as np
import time

start_time = time.time()
input_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves'  
output_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2Claude'

for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):  
        img = Image.open(os.path.join(input_dir,filename))        
        img_np = np.array(img) # convertir imatge a array de numpy
        binary = img_np[:,:,0] < 150  # extreure nuclis marrons (threshold <150)
        area_total = binary.size  
        area_nucleus = np.sum(binary) # comptar nuclis marrons                
        percentatge = area_nucleus/area_total * 100                 
        
        # exportar imatge resultat
        binary = binary*255  
        out_img = Image.fromarray(binary.astype(np.uint8))
        out_img.save(os.path.join(output_dir, filename))
                
        print(f'{filename}: {percentatge:.2f}%')
        



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")

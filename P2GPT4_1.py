#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Màster en Bioinformàtica i Bioestadística
@author: Júlia Sanabria Franquesa
2023
"""

import os

import numpy as np

from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
import time

start_time = time.time()

input_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves/A (1).tif'
output_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2GPT4'

image = io.imread(input_dir)
image_gray = color.rgb2gray(image)
thresh = threshold_otsu(image_gray)
binary = image_gray < thresh

binary_clear = clear_border(binary)
labels = measure.label(binary_clear)
regions = measure.regionprops(labels)
        
area_total = 0
area_nuclis = 0

for region in regions:
    area_total += region.area
    if np.mean(image[labels == region.label]) < 100:
        area_nuclis += region.area
        for coord in region.coords:
            image[coord[0], coord[1]] = (255, 0, 0)



output_file = os.path.join(output_dir, os.path.basename(output_dir))
tiff_file = "A (1)"

ratio = area_nuclis / area_total * 100
print(f"Imatge: {tiff_file}, Àrea total: {area_total}, Àrea nuclis: {area_nuclis}, Percentatge: {ratio:.2f}%")

output_path = os.path.join(output_dir, "output_image.tif")
io.imsave(output_path, image)

end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")

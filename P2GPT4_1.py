#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Màster en Bioinformàtica i Bioestadística
@author: Júlia Sanabria Franquesa
2023
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
import time

start_time = time.time()
def analisi_nuclis(input_dir, output_dir):
    tiff_files = glob.glob(os.path.join(input_dir, '*.tif'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tiff_file in tiff_files:
        image = io.imread(tiff_file)
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



        output_file = os.path.join(output_dir, os.path.basename(tiff_file))
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        ratio = area_nuclis / area_total * 100
        print(f"Imatge: {tiff_file}, Àrea total: {area_total}, Àrea nuclis: {area_nuclis}, Percentatge: {ratio:.2f}%")

if __name__ == '__main__':
    input_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves'
    output_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2GPT4'
    analisi_nuclis(input_dir, output_dir)
    



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")
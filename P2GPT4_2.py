#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:22:53 2023

@author: juliasanabriafranquesa
"""

import cv2
from skimage.color import label2rgb
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

        # Afegit: Eliminar les regions petites que no són rellevants
        min_nuclei_area = 50
        regions = [region for region in regions if region.area > min_nuclei_area]

        area_total = 0
        area_nuclis = 0

        # Afegit: Crear una màscara buida per a l'àrea de biòpsia
        biopsy_mask = np.zeros_like(binary_clear)

        for region in regions:
            area_total += region.area
            mean_intensity = np.mean(image[labels == region.label])

            # Afegit: Detectar l'àrea de biòpsia (àrea més gran amb intensitat alta)
            if mean_intensity > 150:
                biopsy_mask[labels == region.label] = True

            # Modificat: Detectar nuclis marro fosc
            elif mean_intensity < 100:
                area_nuclis += region.area

        # Afegit: Trobar i resseguir el contorn de l'àrea de biòpsia
        contours, _ = cv2.findContours(biopsy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(image, [contour], 0, (255, 0, 0), 2)

        # Afegit: Ressaltar els nuclis marro fosc amb una linia de color verd
        overlay = label2rgb(labels, image=image, bg_label=0, colors=[(0, 255, 0)])
        image = np.where(overlay != 0, overlay, image).astype(np.uint8)

        plt.figure()
        plt.imshow(image)
        plt.axis('off')

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Màster en Bioinformàtica i Bioestadística
@author: Júlia Sanabria Franquesa
2023
"""

import cv2
import numpy as np
import time

start_time = time.time()

def quantify_nuclei(image_path):
    # Llegir la imatge
    image = cv2.imread(image_path)

    # Convertir la imatge a l'espai de colors HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir els rangs de color per al marró fosc (DAB) i el marro clar (àrea biòpsia)
    lower_brown = np.array([0, 100, 50])
    upper_brown = np.array([30, 255, 255])
    lower_light_brown = np.array([0, 50, 100])
    upper_light_brown = np.array([30, 255, 200])

    # Crear una màscara per seleccionar els píxels de color marró fosc
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Crear una màscara per seleccionar els píxels de color marro clar (àrea biòpsia)
    mask_light_brown = cv2.inRange(hsv_image, lower_light_brown, upper_light_brown)

    # Combinar les màscares per obtenir una única màscara que inclogui els nuclis marrons foscos i l'àrea de la biòpsia
    mask_combined = cv2.bitwise_or(mask_brown, mask_light_brown)

    # Aplicar l'operació de morfologia per millorar la màscara combinada
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    # Quantificar el nombre de píxels marrons i el nombre de píxels de l'àrea de la biòpsia
    num_brown_pixels = cv2.countNonZero(mask_brown)
    num_light_brown_pixels = cv2.countNonZero(mask_light_brown)

    # Calcular l'àrea total de la imatge
    total_area = image.shape[0] * image.shape[1]

    # Calcular la fracció d'àrea ocupada pels nuclis marrons i l'àrea de la biòpsia
    fraction_brown_area = num_brown_pixels / total_area
    fraction_light_brown_area = num_light_brown_pixels / total_area

    return fraction_brown_area, fraction_light_brown_area, mask_combined

# Ruta de la imatge
image_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves/A (1).tif'

# Quantificar els nuclis marrons i l'àrea de la biòpsia, i obtenir la màscara combinada
fraction_brown_area, fraction_light_brown_area, mask_combined = quantify_nuclei(image_path)

# Reseguir les àrees en la imatge original
image = cv2.imread(image_path)
image_with_areas = image.copy()
image_with_areas[mask_combined > 0] = [0, 0, 255]  # Reseguir en color vermell

# Exportar la imatge resultant
output_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2GPT3/A (1)_GPT3_2.tif'
cv2.imwrite(output_path, image_with_areas)



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")
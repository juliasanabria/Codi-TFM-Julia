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

    # Crear una màscara per seleccionar els píxels de color marro clar (àrea punch)
    mask_light_brown = cv2.inRange(hsv_image, lower_light_brown, upper_light_brown)

    # Combinar les màscares per obtenir una única màscara que inclogui els nuclis marrons foscos i l'àrea del punch
    mask_combined = cv2.bitwise_or(mask_brown, mask_light_brown)

    # Aplicar l'operació de morfologia per millorar la màscara combinada
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    # Trobar els contorns dels nuclis marrons i de l'àrea del punch
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Resseguir els contorns dels nuclis marrons en verd
    image_with_areas = image.copy()
    cv2.drawContours(image_with_areas, contours, -1, (0, 255, 0), 2)

    # Resseguir l'àrea del punch en vermell
    image_with_areas[mask_light_brown > 0] = [0, 0, 255]

    # Calcular el nombre de nuclis marrons i l'àrea total de la imatge

    total_area = image.shape[0] * image.shape[1]

    # Calcular la fracció d'àrea ocupada pels nuclis marrons i l'àrea del punch
    fraction_brown_area = np.sum(mask_brown) / total_area
    fraction_punch_area = np.sum(mask_light_brown) / total_area

    return fraction_brown_area, fraction_punch_area, image_with_areas

# Ruta de la imatge
image_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves/A (1).tif'

# Quantificar els nuclis marrons i l'àrea del punch
fraction_brown_area, fraction_punch_area, image_with_areas = quantify_nuclei(image_path)

# Exportar la imatge resultant amb els nuclis i l'àrea del punch resseguits
output_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2GPT3/A (1)_result8.tif'
cv2.imwrite(output_path, image_with_areas)

# Mostrar els resultats
print("Fracció d'àrea ocupada pels nuclis marrons:", fraction_brown_area)
print("Fracció d'àrea ocupada per l'àrea del punch:", fraction_punch_area)



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")
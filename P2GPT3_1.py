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

    # Definir els rangs de color per al marró fosc (DAB)
    lower_brown = np.array([0, 100, 50])
    upper_brown = np.array([30, 255, 255])

    # Crear una màscara per seleccionar els píxels de color marró fosc
    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Aplicar l'operació de morfologia per millorar la màscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Quantificar el nombre de píxels marrons
    num_brown_pixels = cv2.countNonZero(mask)

    # Calcular l'àrea total de la imatge
    total_area = image.shape[0] * image.shape[1]

    # Calcular la fracció d'àrea ocupada pels nuclis marrons
    fraction_brown_area = num_brown_pixels / total_area

    return fraction_brown_area, mask

# Ruta de la imatge
image_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves/A (1).tif'

# Quantificar els nuclis marrons i obtenir la màscara
fraction_brown_area, mask = quantify_nuclei(image_path)

# Reseguir les àrees en la imatge original
image = cv2.imread(image_path)
image_with_areas = image.copy()
image_with_areas[mask > 0] = [0, 0, 255]  # Reseguir en vermell (àrea punch)
image_with_areas[mask == 0] = [0, 255, 0]  # Reseguir en verd (àrea nuclis)

# Ruta de sortida per a la imatge resultant amb les àrees resseguides
output_path = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2GPT3/A (1)_GPT3_1.tif'

# Guardar la imatge resultant amb les àrees resseguides
cv2.imwrite(output_path, image_with_areas)

# Imprimir el resultat
print(f'Fracció d\'àrea marró = {fraction_brown_area}')

print('Procesament finalitzat.')



end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")
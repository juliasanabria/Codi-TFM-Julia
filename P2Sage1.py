# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Màster en Bioinformàtica i Bioestadística
# @author: Júlia Sanabria Franquesa
# 2023
# """

import cv2
import numpy as np
import os
import time

start_time = time.time()

# Llegir la imatge
img = cv2.imread('/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/ImatgesOriginals_Proves/A (1).tif')


# Convertir a escala de grisos
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold adaptatiu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Buscar contorns dels nuclis
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibuixar àrees resseguides dels nuclis i calcular la seva àrea
brown_area_sum = 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > 10:  # Descartar nuclis massa petits
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  # Dibuixar àrea del nucli en verd
        brown_area_sum += area

# Dibuixar àrea de la biòpsia en vermell
img_shape = img.shape
pts = np.array([[0, 0], [img_shape[1], 0], [img_shape[1], img_shape[0]], [0, img_shape[0]]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0,0,255), thickness=2)

# Calcular densitat de nuclis marrons
total_area = cv2.contourArea(pts)
brown_density = brown_area_sum / total_area

print("Densitat de nuclis marrons: ", brown_density)




# Exportar la imatge resultant al directori desitjat
output_dir = '/Users/juliasanabriafranquesa/Desktop/UOC/Q4/Q4TFM/Resultats Julia/P2Sage'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'A1result1sage.tif')
cv2.imwrite(output_path, img)
print("Imatge exportada correctament a:", output_path)





end_time = time.time()
execution_time = end_time - start_time

print("El temps d'execució del codi és:", execution_time, "segons")
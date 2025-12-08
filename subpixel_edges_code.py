# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:32:05 2025

@author: gokhanpoyraz
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from subpixel_edges import subpixel_edges

klasor = r"C:\Users\gokhanpoyraz\Desktop\Çalışmalar\Circle_Fitting_Comparison\Concentricity_Measurement-main\Images_Outer"
dosya_adi = "resim1.png"
tam_yol = os.path.join(klasor, dosya_adi)

if not os.path.exists(tam_yol):
    raise FileNotFoundError(f"Görüntü dosyası bulunamadı:\n{tam_yol}")

# Türkçe karakterli yollarda imdecode yöntemi
with open(tam_yol, 'rb') as f:
    img_bytes = f.read()
img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise ValueError("cv2.imdecode başarısız: Dosya okunamadı veya bozuk olabilir")

edges = subpixel_edges(img_gray, 25, 0, 2)

plt.imshow(img_gray, cmap='gray')
plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
plt.title("Subpixel Edge Detection")
plt.show()

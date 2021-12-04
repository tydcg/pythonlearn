import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math

from scanpy import read_10x_h5

import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings

import matplotlib.colors as clr
import matplotlib.pyplot as plt
# In order to read in image data, we need to install some package. Here we recommend package "opencv"
# inatll opencv in python
# !pip3 install opencv-python
import cv2

path = "/Users/wangtong/Desktop/zj"

adata = read_10x_h5(path + "/151673/expression_matrix.h5")
spatial = pd.read_csv(path + "/151673/positions.txt", sep=",", header=None, na_filter=False, index_col=0)
adata.obs["x1"] = spatial[1]
adata.obs["x2"] = spatial[2]
adata.obs["x3"] = spatial[3]
adata.obs["x4"] = spatial[4]
adata.obs["x5"] = spatial[5]
adata.obs["x_array"] = adata.obs["x2"]
adata.obs["y_array"] = adata.obs["x3"]
adata.obs["x_pixel"] = adata.obs["x4"]
adata.obs["y_pixel"] = adata.obs["x5"]

adata = adata[adata.obs["x1"] == 1]
adata.var_names = [i.upper() for i in list(adata.var_names)]
adata.var["genename"] = adata.var.index.astype("str")
adata.write_h5ad("sample_data.h5ad")

img = cv2.imread(path + "/151673/histology.tif")
img_new = img.copy()

x_array = adata.obs["x_array"].tolist()
y_array = adata.obs["y_array"].tolist()
x_pixel = adata.obs["x_pixel"].tolist()
y_pixel = adata.obs["y_pixel"].tolist()

for i in range(len(x_pixel)):
    x = x_pixel[i]
    y = y_pixel[i]
    img_new[int(x - 20):int(x + 20), int(y - 20):int(y + 20), :] = 0

# cv2.imwrite('./sample_results/151673_map.jpg', img_new)
cv2.imwrite('151673_map.jpg', img_new)

s = 1
b = 49

adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s,
                               histology=True)
xp, yp = np.meshgrid(np.arange(0, adj.shape[0], 1), np.arange(0, adj.shape[1], 1))
z = adj
N = 100
z = z[0:N, 0:N]
xp = xp[0:N, 0:N]
yp = yp[0:N, 0:N]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

# ax.set_xlim([0, 100])
# ax.set_ylim([0, 100])
# ax.set_zlim([])

# pc = ax.plot_surface(xp, yp, z, cmap='rainbow')
ax.contour(xp, yp, z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))

plt.show()
# self.ax.contour(self.xp, self.yp, self.z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# self.ax.add_collection(self.pc)
#
# print(adj)

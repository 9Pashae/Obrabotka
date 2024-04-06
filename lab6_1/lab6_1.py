import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils

image = cv.imread('D:\laby python\obrabotka\lab6\chery.jpg')
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

seeds = [(703,569),(1366,1179),(693,648),(734,665),(728,407),(839,509),(837,546),(395,447),(1410,1092),(1195,1187),(1359,1078),(553,1167),(503,1070),(425,1176),(358,544),(353,359),(661,109),(501,136),(619,32),(400,516),(690,647),(690,466),(741,629),(1253,1184),(498,91),(666,61),(589,120),(253,493),(716,665),(747,639),(1282,1098),(1223,1050),(554,547)]

x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))

threshold = 85

segmented_region = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)

result = cv.bitwise_and(image, image, mask=segmented_region)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()
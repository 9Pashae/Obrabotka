import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('D:\laby python\obrabotka\lab5\chery.jpg')


bright_img = cv.convertScaleAbs(img, alpha=1.2, beta=50)


hsv = cv.cvtColor(bright_img, cv.COLOR_BGR2HSV)

# Определение диапазона красного цвета в HSV
lower_red = np.array([0,100,100])
upper_red = np.array([1,255,255])

#  маски к изображению
mask = cv.inRange(hsv, lower_red, upper_red)

# Бинарное изображение, где красные области белые, а все остальные черные
result = cv.bitwise_and(bright_img, bright_img, mask=mask)


plt.figure(figsize=(15, 15))


plt.subplot(1, 4, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(cv.cvtColor(bright_img, cv.COLOR_BGR2RGB))
plt.title('Brightened Image')

plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title('Mask')

plt.subplot(1, 4, 4)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title('Result Image')

plt.show()
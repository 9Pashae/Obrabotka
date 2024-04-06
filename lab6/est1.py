import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Загрузите изображение
image = cv.imread('D:\laby python\obrabotka\lab6\chery.jpg')

# Преобразуйте изображение в пространство HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Установите границы красного цвета в пространстве HSV
red_lower = (0, 160, 160)
red_upper = (10, 255, 255)

# Создайте маску для красного цвета
mask = cv.inRange(hsv_image, red_lower, red_upper)

# Примените маску к исходному изображению
result = cv.bitwise_and(image, image, mask=mask)

# Отобразите изображения
plt.figure(figsize=(15, 20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(hsv_image)
plt.title('HSV Image')
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title('Result Image')
plt.show()
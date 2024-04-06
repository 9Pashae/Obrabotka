import numpy as np
import cv2  as cv
import matplotlib.pyplot as plt
input_image = cv.imread('D:\laby python\obrabotka\lab4\etrans.jpg')
plt.imshow(input_image)

image = input_image

gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)


channels = [0]
histSize = [256]
range = [0, 256]

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.hist(gray_image.reshape(-1), 256, range)
plt.show()

threshold = 115
image = gray_image

ret, binary_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
plt.title("Binary")
plt.imshow(binary_image,"gray")
plt.show()

cv.imwrite("binary_image.jpg", binary_image)

image = binary_image

kernel = np.ones((4, 4), np.uint8)
dilation = cv.dilate(image, kernel, iterations=1)
erosion = cv.erode(image, kernel, iterations=1)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
closeAndOpen = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

# вывод
plt.figure(figsize=(15, 8))
gs = plt.GridSpec(2, 3)

titles = ['Зашумленное изображение', 'Дилатация', 'Эрозия', 'Открытие', 'Закрытие',
          'Последовательное закрытие и открытие']
outImages = [image, dilation, erosion, opening, closing, closeAndOpen]

for i in np.arange(len(outImages)):
    plt.subplot(gs[i])
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
    plt.imshow(outImages[i], cmap='gray')

plt.show()

cv.imwrite("erosion.jpg", erosion)

image = erosion

kernel = np.ones((3, 3), np.uint8)
dilation = cv.dilate(image, kernel, iterations=1)
erosion = cv.erode(image, kernel, iterations=1)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
closeAndOpen = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

# вывод
plt.figure(figsize=(15, 8))
gs = plt.GridSpec(2, 3)

titles = ['Зашумленное изображение', 'Дилатация', 'Эрозия', 'Открытие', 'Закрытие',
          'Последовательное закрытие и открытие']
outImages = [image, dilation, erosion, opening, closing, closeAndOpen]

for i in np.arange(len(outImages)):
    plt.subplot(gs[i])
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
    plt.imshow(outImages[i], cmap='gray')

plt.show()
plt.imshow(closeAndOpen,"gray")
cv.imwrite("closeAndOpen.jpg", closeAndOpen)
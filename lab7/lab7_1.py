import cv2
import numpy as np

big_image_path = 'D:\laby python\obrabotka\lab7\ig.jpg'
small_image_path = 'D:\laby python\obrabotka\lab7\mal2.jpg'

big_image = cv2.imread(big_image_path)
small_image = cv2.imread(small_image_path)

# матрицы
result_ccoeff = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
result_sqdiff = cv2.matchTemplate(big_image, small_image, cv2.TM_SQDIFF_NORMED)
result_ccorr = cv2.matchTemplate(big_image, small_image, cv2.TM_CCORR_NORMED)

# Нахождение максимального совпадения для каждого метода
_, max_val_ccoeff, _, max_loc_ccoeff = cv2.minMaxLoc(result_ccoeff)
_, max_val_sqdiff, _, max_loc_sqdiff = cv2.minMaxLoc(result_sqdiff)
_, max_val_ccorr, _, max_loc_ccorr = cv2.minMaxLoc(result_ccorr)


max_val_ccoeff > 0.9 or max_val_sqdiff > 0.9 or max_val_ccorr > 0.9
    # прямоугольник
cv2.rectangle(big_image, (max_loc_ccoeff[0], max_loc_ccoeff[1]), (max_loc_ccoeff[0] + small_image.shape[1], max_loc_ccoeff[1] + small_image.shape[0]), (0, 0, 255), 2)
cv2.imwrite('result.jpg', big_image)

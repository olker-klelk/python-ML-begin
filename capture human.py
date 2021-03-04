
import cv2
import numpy as np


np.sum(12)
img = cv2.imread(r'./image/image01.jpg',cv2.IMREAD_GRAYSCALE)

print(type(img))
print(img.shape)
cv2.imshow('imgg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


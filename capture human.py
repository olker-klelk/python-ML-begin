'''
import cv2
import numpy as np


np.sum(12)
img = cv2.imread(r'./image/image01.jpg',cv2.IMREAD_GRAYSCALE)

print(type(img))
print(img.shape)
cv2.imshow('imgg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import numpy as np
from scipy.optimize import fsolve


def founc():

    x1 = np.array([2, 4, 2, 12, -8])
    y = np.array([2, -2, 1, -1, 2])
    z = np.array([1, -2, 3, -2, -2])
    a = np.array([-1, 2, -3, -3, 2])
    b = np.array([-3, 1, 2, -5, 2])
    oup = np.array([20, -1, 35, -9, -8])

    D = np.linalg.det(np.vstack((x1, y, z, a, b)))
    Dx = np.linalg.det(np.vstack((oup, y, z, a, b)))
    Dy = np.linalg.det(np.vstack((x1, oup, z, a, b)))
    Dz = np.linalg.det(np.vstack((x1, y, oup, a, b)))
    Da = np.linalg.det(np.vstack((x1, y, z, oup, b)))
    Db = np.linalg.det(np.vstack((x1, y, z, a, oup)))

    inxd = np.array([Dx, Dy, Dz, Da, Db])
    inx = inxd / D

    print(inx)
    print("unknow!")

def founction_5(unknow):

    x, y, z, a, b = unknow[0], unknow[1], unknow[2], unknow[3], unknow[4]

    f1 = x * 2 + 2 * y + z * 1 + a * -1 + b * -3 - 20
    f2 = x * 4 + y * -2 + -2 * z + 2 * a + b * 1 + 1
    f3 = x * 2 + y * 1 + z * 3 + a * -3 + 2 * b - 35
    f4 = x * 12 + y * -1 + z * -2 + -3 * a + b * -5 + 9
    f5 = x * -8 + 2 * y + z * -2 + a * 2 + 2 * b + 8

    return [f1, f2, f3, f4, f5]

founc()

x = fsolve(founction_5, [0, 0, 0, 0, 0])
print(x)

print("Program done!")

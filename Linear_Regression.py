import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r"./803_LRdata.csv")
x1 = np.array(df.x1.iloc[0:1000])
x2 = np.array(df.x2.iloc[0:1000])
x3 = np.array(df.x3.iloc[0:1000])
x4 = np.array(df.x4.iloc[0:1000])
x5 = np.array(df.x5.iloc[0:1000])
y1 = np.array(df.y1.iloc[0:900])
y2 = np.array(df.y2.iloc[0:900])
oy1 = np.array(df.y1.iloc[900:1000])
oy2 = np.array(df.y2.iloc[900:100])
x1_5=np.array(df.iloc[0:1000,0:5])

def martix_f1(A,B):

    A_inv = np.linalg.inv(A)
    ans = A_inv.dot(B)

    return ans

ans1 = np.full((180, 5), 0,dtype=float)
ans2 = np.full((180, 5), 0,dtype=float)
for i in range(0, 900, 5):
    ans1[int(i/5)] = martix_f1(x1_5[i:i+5], y1[i:i+5])
for i in range(0, 900, 5):
    ans2[int(i/5)] = martix_f1(x1_5[i:i+5], y1[i:i+5])
for i in range(0, 180):
    for j in range(i*5+0, i*5+5):
        mse = (np.sum(ans1[i]*x1_5[j])-y1[j])**2
print(mse)
print(np.dot(x1_5[0:5],ans1[0]))
lr = LinearRegression()
lr.fit(x1_5[0:900], y1[0:900])
print('Intercept:')
print(lr.intercept_)
print('\n')
print('Coefficient:')
print(lr.coef_)

y_pred = lr.predict(x1_5[0:900])
mse_validation = mean_squared_error(y1[0:900], y_pred)
print('MSE:')
print(mse_validation)
np.set_printoptions(precision = 2)
Y_o=lr.predict(x1_5[900:1000])



'''

plt.scatter(y_pred, y1[0:720], label='YYyy')

plt.plot(x, 5 * x + x * 10 + x * 1 + 2 * x + 4 * x, 'r')
plt.legend()
plt.show()
def martix_function(x1,y,z,a,b,oup):

    D = np.linalg.det(np.vstack((x1, y, z, a, b)))
    Dx = np.linalg.det(np.vstack((oup, y, z, a, b)))
    Dy = np.linalg.det(np.vstack((x1, oup, z, a, b)))
    Dz = np.linalg.det(np.vstack((x1, y, oup, a, b)))
    Da = np.linalg.det(np.vstack((x1, y, z, oup, b)))
    Db = np.linalg.det(np.vstack((x1, y, z, a, oup)))
    inxd = np.array([Dx, Dy, Dz, Da, Db])
    inx = inxd / D
    return inx

ans1 = {}
ans2 = {}
for i in range(0,900,5):
    ans1[i/5] = martix_function(x1[i:i+5],x2[i:i+5],x3[i:i+5],x4[i:i+5],x5[i:i+5],y1[i:i+5])
for i in range(0,900,5):
    ans2[i/5] = martix_function(x1[i:i+5],x2[i:i+5],x3[i:i+5],x4[i:i+5],x5[i:i+5],y2[i:i+5])
'''






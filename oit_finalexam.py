import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import math
import random

pi = math.pi
ts = 1/1000
f = 20
x=[0]*1000
x_noise=[0]*1000
for i in range(1000):
    x[i]=math.sin(2*pi*f*i*ts)
    x_noise[i]=x[i]+random.random()

X=fft(x)
X_noise=fft(x_noise)

h1=[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
h2=[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
h3=[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
H1=fft(h1,256)
H2=fft(h2,256)
H3=fft(h3,256)
x1=np.convolve(x_noise,h1)
x2=np.convolve(x_noise,h2)
x3=np.convolve(x_noise,h3)

plt.subplot(221)
plt.plot(x)
plt.subplot(222)
plt.plot(x_noise)
plt.subplot(223)
plt.plot(abs(X[1:int(len(X)/2)]))
plt.subplot(224)
plt.plot(abs(X_noise[1:int(len(X_noise)/2)]))
plt.figure()
plt.subplot(311)
plt.plot(x1)
plt.subplot(312)
plt.plot(x2)
plt.subplot(313)
plt.plot(x3)
plt.figure()
plt.subplot(231)
plt.plot(h1)
plt.subplot(232)
plt.plot(h2)
plt.subplot(233)
plt.plot(h3)
plt.subplot(234)
plt.plot(abs(H1[:int(len(H1)/2)]))
plt.subplot(235)
plt.plot(abs(H2[:int(len(H2)/2)]))
plt.subplot(236)
plt.plot(abs(H3[:int(len(H3)/2)]))
plt.show()

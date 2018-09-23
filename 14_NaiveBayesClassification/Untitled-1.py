import random
import matplotlib.pyplot as plt


elem = 10000
x = []
y = []
colors = []

def iteration(x):
    n = 0
    result = 0
    while n < 1000: # max iterations
        result = pow(result, 2) + c
        if abs(result) > 2:
            return n           
        n += 1
    return n

for i in range(elem):
    c = complex(random.uniform(-2.5,1),random.uniform(-1.5,1.5))
    x.append(c.real)
    y.append(c.imag)
    colors.append(iteration(c))


plt.scatter(x, y, marker=',', c = colors, cmap= 'magma', vmin = 0, vmax = 1000)
plt.axis('equal')
plt.axis([-2,1,-1.5,1.5])
plt.title('Mandelbrot Set', y = 1.05)
plt.ylabel('Im')
plt.xlabel('Re')
plt.colorbar()
plt.show()
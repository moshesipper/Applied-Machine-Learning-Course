import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation



def ackley(x1,x2):
    a = 20
    b = 0.2
    c = 2*np.pi
    
    sum1 = x1**2 + x2**2 
    sum2 = np.cos(c*x1) + np.cos(c*x2)
    
    term1 = - a * np.exp(-b * ((1/2.) * sum1**(0.5)))
    term2 = - np.exp((1/2.)*sum2)

    return term1 + term2 + a + np.exp(1)

plotN = 100
x1 = np.linspace(-1, 1, plotN)
x2 = np.linspace(-1, 1, plotN)

x1, x2 = np.meshgrid(x1,x2)

z = ackley(x1,x2)

fig = plt.figure()

#It's smooth
ax = Axes3D(fig)
ax.plot_wireframe(x1,x2,z)

#It's heavy but it has color
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)

ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.xaxis.set_ticks([-1,0,1])
ax.yaxis.set_ticks([-1,0,1])
ax.zaxis.set_ticks([0,2,4])
plt.show()

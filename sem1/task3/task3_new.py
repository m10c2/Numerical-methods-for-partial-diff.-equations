import matplotlib.pyplot as plt
import numpy as np

def func_u(x,t):
    return x**4 - x + t*x + t**2 - t*np.exp(x)

def func_f(x,t):
    return x + 2*t - np.exp(x) - 0.036*(12*x**2 - t*np.exp(x))

def print_matrix(A):
    print()
    for i in range(len(A)):
        for j in range(len(A[i])):
            print(A[i][j], end = ' ')
        print()


#Nx = 10,40,100,1000 => h = 0.1 , 0,025, 0,01, 0,001
a = 0.036
Nx = 10
Nt = 10
h = 1/Nx
tau = 1/Nt

u = np.zeros((Nx+1,Nt+1))

#t=0
for k in range(Nx+1):
    u[k][0] = func_u(h*k,0)

#x=0 & x=1
for j in range(Nt+1):
    u[0][j] = func_u(0,tau*j)
    u[Nx][j] = func_u(1,tau*j)

for j in range(1,Nt+1):
    for k in range(1,Nx):
        u[k][j] = ((a*tau)/h**2)*(u[k+1][j-1] - 2*u[k][j-1] + u[k-1][j-1]) + tau*func_f(h*k,tau*(j-1)) + u[k][j-1]

real_u = np.zeros(u.shape)
for k in range (Nx+1):
    for j in range(Nt+1):
        real_u[k][j] = func_u(h*k,tau*j)
print(np.max(np.abs(real_u - u)))

if ((a*tau)/h**2 <= 1/2):
    print('stable')
else:
    print('false')


#график приближенного и точного решения
t = np.arange(0, 1+tau, tau)
x = 0.5
y = x**4-x+t*x+t**2-t*np.exp(x)
fig = plt.figure()
plt.plot(t, u[5], color='red', label = 'численное решение')
plt.plot(t, y, color='blue', label = 'точное решение')
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

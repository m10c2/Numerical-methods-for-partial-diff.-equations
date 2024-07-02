import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as pi

def func_u(x,t):
    return (-(t-0.3)**3)+np.cos(pi*x)+2*pi*x

def func_phi0(x):
    return np.cos(pi*x)+0.027+2*pi*x

def func_u0(t):
    return (-(t-0.3)**3)+1

def func_c(x,t):
    return (3*(t-0.3)**2)/(2*pi-pi*np.sin(pi*x))

def thomas_alg(u,c,Nx,Nt,j):
    h = 1/Nx
    tau = 1/Nt
    a = (4*h)/tau

    u[Nx][j+1] = (1 - (tau/h)*func_c(h*Nx,tau*j))*u[Nx][j] + (tau/h)*func_c(h*Nx,tau*j)*u[Nx-1][j]

    A = np.zeros((Nx-1,Nx-1))
    A[0][0] = a
    A[0][1] = c[1][j]
    A[Nx-2][Nx-3] = -c[Nx-1][j]
    A[Nx-2][Nx-2] = a
    for k in range (1,Nx-2):
        A[k][k-1] = -c[k+1][j]
        A[k][k] = a
        A[k][k+1] = c[k+1][j]

    b = np.zeros(Nx-1)
    b[0] = c[1][j]*u[0][j+1]+a*u[1][j]-c[1][j]*u[2][j]+c[1][j]*u[0][j]
    b[Nx-2] = -c[Nx-1][j]*u[Nx][j+1]+a*u[Nx-1][j]-c[Nx-1][j]*u[Nx][j]+c[Nx-1][j]*u[Nx-2][j]
    for k in range(1,Nx-2):
        b[k] = a*u[k+1][j] - c[k+1][j]*u[k+2][j] + c[k+1][j]*u[k][j]

    solution = np.linalg.solve(A,b)
    return(solution)

def print_matrix(A):
    print()
    for i in range(len(A)):
        for j in range(len(A[i])):
            print(A[i][j], end = ' ')
        print()

Nx = 10
Nt = 10
h = 1/Nx
tau = 1/Nt
a = (4*h)/tau

c = np.zeros((Nx+1,Nt+1))
for k in range(Nx+1):
    for j in range(Nt+1):
        c[k][j] = func_c(h*k,tau*(j+1/2))

u = np.zeros((Nx+1,Nt+1))

#t=0
for k in range(Nx+1):
    u[k][0] = func_phi0(h*k)

#x=0 & x=1
for j in range(Nt+1):
    u[0][j] = func_u0(tau*j)

#нахождение численного решения
for j in range(Nt):
    solution = thomas_alg(u,c,Nx,Nt,j)
    for k in range(Nx-1):
        u[k+1][j+1] = solution[k]

#точность
real_u = np.zeros(u.shape)
for k in range (Nx+1):
    for j in range(Nt+1):
        real_u[k][j] = func_u(h*k,tau*j)
print(np.max(np.abs(real_u - u)))

#Графики
t = np.arange(0, 1+tau, tau)
x = 0.5
y = (-(t-0.3)**3)+np.cos(pi*x)+2*pi*x
fig = plt.figure()
plt.plot(t, u[5], color='red', label = 'численное решение')
plt.plot(t, y, color='blue', label = 'точное решение')
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

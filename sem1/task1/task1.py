import matplotlib.pyplot as plt 
import numpy as np

def func_u(x,t):
    return x**4-x+t*x+t**2-t*np.exp(x)
def func_mu(x):
    return x**4 - x

def func_mu1(t):
    return t**2 - t

def func_mu2(t):
    return t + t**2 - t*np.exp(1)

def func_f(x, t):
    a = 0.036
    return x + 2*t - np.exp(x) - a*(12*x**2 - t*np.exp(x))
#тут будет прогонка вместо линалга
def lin_eq_sol(A,b):
    result = np.linalg.solve(A,b)
    return result

a = 0.036

h = 0.1
tau = 0.1
c = a/(h**2)

N_x = int(1/h)
N_t = int(1/tau)

u = np.zeros((N_x+1, N_t+1))
accuracy = 0

# t = 0
for i in range(N_x+1):
    u[i][0] = func_mu(i*h)
    
# x = 0
for i in range(N_t+1):
    u[0][i] = func_mu1(i*tau)

# x = 1
for i in range(N_x+1):
    u[N_x][i] = func_mu2(i*tau)
    
A = np.zeros((N_x-1,N_x-1))

for i in range(N_x - 2):
    A[i][i] = 1 + 2*c*tau
    A[i+1][i] = -c*tau
    A[i][i+1] = -c*tau
    if i == (N_x - 3):
        A[i+1][i+1] = 1 + 2*c*tau
        
# ищем решение послойно
for j in range (N_t): #<---- вычисляем j+1 слой
    #правая часть
    b = np.zeros(N_x - 1)
    b[0] = u[1][j] + tau*func_f(h,(j+1)*tau) + c*tau*u[0][j+1]
    b[N_x-2] = u[N_x-1][j] + tau*func_f((N_x-1)*h,(j+1)*tau) + c*tau*u[N_x][j+1]
    for k in range(1,N_x - 2):
        b[k] = u[k+1][j] + tau*func_f((k+1)*h,(j+1)*tau)
    
    x = lin_eq_sol(A,b)
    for k in range(N_x-1):
        u[k+1][j+1] = x[k]

#точность
for j in range (N_t+1):
    tmp = np.zeros(N_x + 1)
    for k in range(N_x + 1):
        tmp[k] = abs(u[k][j] - func_u(k*h,j*tau))
    
    max = np.max(tmp)
    if (max >= accuracy):
        accuracy = max
        
print(accuracy)   

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



    



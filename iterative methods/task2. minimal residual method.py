import numpy as np
import matplotlib.pyplot as plt
a = 1.1
b = 0.8

def norm(x,n):
    h = 1/n
    norm = 0
    for i in x:
        norm += i**2
    return np.sqrt(norm)*h

def sc(x, y):
    sc = 0
    for i in range(len(x)):
        sc += x[i]*y[i]
    return sc

def func_f(x,y):
    return 1.1*np.sin(x) + (3.2*(x**2) + 4.4*(y**2))*np.cos(2*x*y)

def func_phi(x,y):
    return np.sin(x) + np.cos(2*x*y)

def start_matrix(n):
    Y = np.zeros((n+1,n+1))
    h = 1/n
    #заполняю матрицу(зависит от области)
    for i in range (1,n):
        for j in range(1,int(n/2)):
            Y[i][j] = 1

    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            Y[i][j] = 1

    for j in range(1,int(n/2)):
        Y[0][j] = func_phi(0,h*j)

    for j in range(1,n+1):
        Y[n][j] = func_phi(1,h*j)

    for i in range(n+1):
        Y[i][0] = func_phi(h*i,0)

    for i in range(int(n/2)):
        Y[i][int(n/2)] = func_phi(h*i,h*int(n/2))

    for i in range(int(n/2),n+1):
        Y[i][i] = func_phi(h*i,h*i)

    return Y

def mult_by_A(Y,n):
    h = 1/n
    Res = np.zeros((n+1,n+1))
    for i in range (1,n):
        for j in range(1,int(n/2)):
            Res[i][j] = -1*(a/h**2)*(Y[i-1][j]-2*Y[i][j]+Y[i+1][j]) - (b/h**2)*(Y[i][j-1]-2*Y[i][j]+Y[i][j+1])
    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            Res[i][j] = -1*(a/h**2)*(Y[i-1][j]-2*Y[i][j]+Y[i+1][j]) - (b/h**2)*(Y[i][j-1]-2*Y[i][j]+Y[i][j+1])
    return Res

def init_F(n):
    h = 1/n
    F = np.zeros((n+1, n+1))

    for i in range (1,n):
        for j in range(1,int(n/2)):
            F[i][j] = func_f(h*i,h*j)

    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            F[i][j] = func_f(h*i,h*j)
    return F

def init_phi(n):
    h = 1/n
    phi = np.zeros((n+1, n+1))

    for i in range (1,n):
        for j in range(1,int(n/2)):
            phi[i][j] = func_phi(h*i,h*j)

    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            phi[i][j] = func_phi(h*i,h*j)

    for j in range(1,int(n/2)):
        phi[0][j] = func_phi(0,h*j)

    for j in range(1,n+1):
        phi[n][j] = func_phi(1,h*j)

    for i in range(n+1):
        phi[i][0] = func_phi(h*i,0)

    for i in range(int(n/2)):
        phi[i][int(n/2)] = func_phi(h*i,h*int(n/2))

    for i in range(int(n/2),n+1):
        phi[i][i] = func_phi(h*i,h*i)

    return phi

def solve(n, delta):
    Y_n = np.zeros((n+1,n+1))
    Y = start_matrix(n)
    F = init_F(n)
    phi = init_phi(n)

    R = mult_by_A(Y,n) - F
    print(R)
    print('\n',norm(R.reshape(-1),n))
    tau_k = sc(mult_by_A(R,n).reshape(-1), R.reshape(-1)) / sc(mult_by_A(R,n).reshape(-1),mult_by_A(R,n).reshape(-1) )

    print(tau_k)

    Y_n = Y - tau_k*R #Y1
    iter = 1

    while norm(((mult_by_A(Y_n,n)-F)-R).reshape(-1),n) > delta :

        Y = Y_n.copy()

        iter += 1

        R = mult_by_A(Y,n) - F #невязка на данном шаге
        tau_k = sc(mult_by_A(R,n).reshape(-1), R.reshape(-1)) / sc(mult_by_A(R,n).reshape(-1),mult_by_A(R,n).reshape(-1) )

        Y_n = Y - tau_k*(mult_by_A(Y,n)-F)

    epsilon = norm((Y_n - phi).reshape(-1),n)
    return [epsilon, iter]

def average_iterations(n,delta):
    h = 1/n
    res = (np.log(1/delta))/((np.pi*h)**2)
    return round(res)

accuracy = [10**(-6),10**(-7),10**(-8)]
grid = [10,20,40]


for k in grid:
    for delta in accuracy:
        print(average_iterations(k,delta))
    print('\n')

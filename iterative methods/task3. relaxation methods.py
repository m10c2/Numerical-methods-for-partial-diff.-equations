import numpy as np
import matplotlib.pyplot as plt
a = 1.1
b = 0.8
lambda_min = 38

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

def next(Y,n,w):
    h = 1/n
    Res = Y.copy()

    #заполняю матрицу(зависит от области)
    for i in range (1,n):
        for j in range(1,int(n/2)):
            Res[i][j] = (1-w)*Y[i][j] + (w/(2*(a+b)))*(a*Res[i-1][j] + b*Res[i][j-1] + a*Y[i+1][j] + b*Y[i][j+1] + (h**2)*func_f(i*h,j*h))

    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            Res[i][j] = (1-w)*Y[i][j] + (w/(2*(a+b)))*(a*Res[i-1][j] + b*Res[i][j-1] + a*Y[i+1][j] + b*Y[i][j+1] + (h**2)*func_f(i*h,j*h))
    return Res

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

def solve_opt(n, delta):
    Y_n = np.zeros((n+1,n+1))
    Y = start_matrix(n)
    F = init_F(n)
    phi = init_phi(n)

    h = 1/n

    betta = ((h**2)/(2*(a+b)))*lambda_min
    w = 2/(1 + np.sqrt(betta*(2-betta)))

    R = mult_by_A(Y,n) - F

    Y_n = next(Y,n,w)
    iter = 1

    while norm(((mult_by_A(Y_n,n)-F)-R).reshape(-1),n) > delta :

        Y = Y_n.copy()
        R = mult_by_A(Y,n) - F #невязка на данном шаге
        iter += 1

        Y_n = next(Y,n,w)
        print(norm(Y_n.reshape(-1),n))

    epsilon = norm((Y_n - phi).reshape(-1),n)
    return [epsilon, iter]

def solve(n, delta, w):
    Y_n = np.zeros((n+1,n+1))
    Y = start_matrix(n)
    F = init_F(n)
    phi = init_phi(n)
    h = 1/n

    R = mult_by_A(Y,n) - F
    Y_n = next(Y,n,w)
    iter = 1

    while norm(((mult_by_A(Y_n,n)-F)-R).reshape(-1),n) > delta :

        Y = Y_n.copy()
        iter += 1

        R = mult_by_A(Y,n) - F #невязка на данном шаге
        Y_n = next(Y,n,w)

    epsilon = norm((Y_n - phi).reshape(-1),n)
    return [epsilon, iter]

def average_iterations(n,delta):
    h = 1/n
    res = (np.log(1/delta))/(2*np.pi*h)
    return round(res)

def non_opt_iterations(n,delta):
    h = 1/n
    #res = np.log(1/delta)/(np.pi*h)**2
    res = 0.1*(n**2)*np.log(1/delta)
    return round(res)


accuracy = [10**(-6),10**(-7),10**(-8)]
grid = [10,20,40]


w = 1
for k in grid:
    for delta in accuracy:
        print(solve(k,delta,w))
    print('\n')

'''
for k in grid:
    for delta in accuracy:
        print(solve_opt(k,delta))
    print('\n')
'''

for k in grid:
    for delta in accuracy:
        print(non_opt_iterations(k,delta))
    print('\n')

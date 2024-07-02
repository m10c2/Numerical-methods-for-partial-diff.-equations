import numpy as np
import matplotlib.pyplot as plt
a = 1.1
b = 0.8

def norm(x):
    norm = 0
    for i in x:
        norm += i**2
    return np.sqrt(norm)

def scalar(x, y):
    sc = 0
    for i in range(len(x)):
        sc += x[i]*y[i]
    return sc

#выплевывает A*Y
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

def mult_by_B(Y,n,number):
    h = 1/n
    Res = np.zeros((n+1,n+1))
    for i in range (1,n):
        for j in range(1,int(n/2)):
            Res[i][j] =number*Y[i][j] + (a/h**2)*(Y[i-1][j]-2*Y[i][j]+Y[i+1][j]) + (b/h**2)*(Y[i][j-1]-2*Y[i][j]+Y[i][j+1])

    for i in range(int(n/2)+1,n):
        for j in range(int(n/2),i):
            Res[i][j] =number*Y[i][j] + (a/h**2)*(Y[i-1][j]-2*Y[i][j]+Y[i+1][j]) + (b/h**2)*(Y[i][j-1]-2*Y[i][j]+Y[i][j+1])

    return Res

def max_eigenv(Y,n,delta):
    #Y1
    Y_n = mult_by_A(Y,n)

    #первая итерация - вручную
    iter = 1
    L_max_n = scalar( Y_n.reshape(-1),Y.reshape(-1))/scalar(Y.reshape(-1),Y.reshape(-1))

    while True:
        Y = Y_n.copy()/norm(Y.reshape(-1))
        L_max_prev = L_max_n
        iter += 1

        Y_n = mult_by_A(Y,n)
        L_max_n = scalar(Y_n.reshape(-1),Y.reshape(-1))/scalar(Y.reshape(-1),Y.reshape(-1))

        if np.abs(L_max_n - L_max_prev)/np.abs(L_max_prev) < delta:
            break
    return [L_max_n, iter]

def min_eigenv(Y,n,delta):
    X = Y.copy()
    L_max_n = max_eigenv(Y,n,accuracy[2])[0]
    #B = L_max*E - A
    #Ищем максимальное собственное число матрицы B
    iter_B = 1
    X_n = mult_by_B(Y,n,L_max_n)

    L_max_B_n = scalar(X_n.reshape(-1),X.reshape(-1))/scalar(X.reshape(-1),X.reshape(-1))

    while True:
        X = X_n.copy()/norm(X.reshape(-1))
        L_max_prev = L_max_B_n
        iter_B += 1

        X_n = mult_by_B(X,n,L_max_n)
        L_max_B_n = scalar(X_n.reshape(-1),X.reshape(-1))/scalar(X.reshape(-1),X.reshape(-1))

        if np.abs(L_max_B_n - L_max_prev)/np.abs(L_max_prev) < delta:
            break
    # миниальное собственное значение А
    return [L_max_n - L_max_B_n,iter_B]


accuracy = [0.000001,0.0000001,0.00000001]
grid = [10,20,40]

n = grid[2]
delta = accuracy[2]

Y = np.zeros((n+1,n+1))
Y_n = np.zeros((n+1, n+1))

#заполняю матрицу(зависит от области)
for i in range (1,n):
    for j in range(1,int(n/2)):
        Y[i][j] = 1

for i in range(int(n/2)+1,n):
    for j in range(int(n/2),i):
        Y[i][j] = 1


'''
for k in grid:
    print(k)
    for delta in accuracy:
        print(max_eigenv(Y,k,delta))
    print('\n')

'''

print(min_eigenv(Y,n,delta))

import sys
import numpy as np
import matplotlib.pyplot as plt

def Point_charges(i, j, n):
    spacing = R / n
    if [i, j] == [n // 2 + int(a/(2*spacing)), n // 2]:
        return Q
    elif [i, j] == [n // 2 - int(a/(2*spacing)), n // 2]:
        return -Q
    else:
        return 0

def Jacobi(tolerance, n):
    i = 0
    spacing = R / n
    V = np.zeros((n, n))
    while True:
        V_new = V.copy()
        for i in range(1, n-1):
            for j in range(1, n-1):
                V_new[i, j] = 1 / 4 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] - spacing ** 2 * Point_charges(i, j, n))
        delta_V = np.max(np.abs(V - V_new))
        i += 1
        V = V_new.copy()
        if delta_V < tolerance:
            break
    return V, i

def SOR(tolerance, n):
    i = 0
    spacing = R / n
    factor = 2/ (1 + np.pi / n)
    V = np.zeros((n, n))
    while True:
        V_old = V.copy()
        for i in range(1, n-1):
            for j in range(1, n-1):
                V[i, j] = (1 - factor) * V[i, j] + factor * 1 / 6 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] + spacing ** 2 * Point_charges(i, j, n))
        delta_V = np.max(np.abs(V - V_old))
        i += 1
        if delta_V < tolerance:
            break
    return V, i

def Part_1(n):
    tolerance = 1e-6
    V, i = Jacobi(tolerance, n)
    plt.figure()
    plt.contour(V, levels=15)
    plt.colorbar()
    plt.title('Equipotential line')
    plt.savefig('Figure 10.jpeg', format='jpeg')
    plt.show()
    plt.figure()
    r = []
    spacing = R / n
    for i in range(n):
        r.append(i * n)
    plt.plot(r, V[:, n // 2])
    plt.title('V(r)')
    plt.ylabel('Potential V')
    plt.xlabel('Distance r')
    plt.savefig('Figure 11.jpeg', format='jpeg')
    plt.show()

def Part_2(n):
    tolerance = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    i = []
    for j in tolerance:
        V, interation = Jacobi(j, n)
        i.append(interation)
    plt.figure()
    plt.plot(tolerance, i, 'o-')
    plt.xlabel('Tolerances')
    plt.xscale('log')
    plt.ylabel('Iteration times')
    plt.title('Iteration times vs. Tolerance')
    plt.savefig('Figure 12.jpeg', format='jpeg')
    plt.show()

def Part_3():
    n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    Jacobi_iteration = []
    SOR_iteration = []
    alpha = 2 / (1 + np.pi / 2)
    tolerance = 1e-6
    for N in n:
        result, iteration = Jacobi(tolerance, N)
        Jacobi_iteration.append(iteration)
        result, iteration = SOR(tolerance, N)
        SOR_iteration.append(iteration)
    Num = np.zeros(len(n))
    for i in range(len(n)):
        Num[i] = n[i] ** 2
    plt.plot(Num, Jacobi_iteration, label='Jacobi metod')
    plt.plot(Num, SOR_iteration, label='SOR method')
    plt.xlabel('Number of grid points')
    plt.ylabel('Iteration times')
    plt.title('Iteration times of Jacobi and SOR vs. number of grid points')
    plt.legend()
    plt.savefig('Figure 13.jpeg', format='jpeg')
    plt.show()
        
if __name__ == '__main__':
    a = 0.6
    Q = 1
    R = 10
    n = 100    
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in number:
        if i == '1':
            Part_1(n)
        if i == '2':
            Part_2(n)
        if i == '3':
            Part_3()

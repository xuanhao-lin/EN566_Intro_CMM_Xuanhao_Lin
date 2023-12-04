import numpy as np
import matplotlib.pyplot as plt
import sys

def Energy(lattice, n):
    E = 0
    for i in range(n):
        for j in range(n):
            spin = lattice[i, j]
            neighbors = lattice[(i + 1) % n, j] + lattice[i, (j + 1) % n] + lattice[(i - 1) % n, j] + lattice[i, (j - 1) % n]
            E += -1.5 * spin * neighbors
    return E / 2

def Metropolis(lattice, n, MC_sweep, T):
    M = 0
    E = 0
    E_2 = 0
    E_current = Energy(lattice, n)
    print(MC_sweep)
    for step in range(MC_sweep):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        spin = lattice[i, j]
        neighbors = lattice[(i + 1) % n, j] + lattice[i, (j + 1) % n] + lattice[(i - 1) % n, j] + lattice[i, (j - 1) % n]
        dE = 2 * 1.5 * spin * neighbors
        if dE <= 0:
            lattice[i, j] = -lattice[i, j]
            E_current += dE
        elif np.random.rand() <= np.exp(-dE / T):
            lattice[i, j] = -lattice[i, j]
            E_current += dE
        M += np.abs(np.sum(lattice)) / (n ** 2)
        E += E_current
        E_2 += E_current ** 2
    return M / MC_sweep, E / MC_sweep, E_2 / MC_sweep

def Part_1():
    T = np.arange(0.1, 10.1, 0.1)
    n = 50
    MC_sweep = 10000 * (n ** 2)
    M = np.zeros(len(T))
    np.random.seed(0)
    for i in range(len(T)):
        lattice = np.random.choice([-1, 1], size=(n, n))
        M[i], E, E_2 = Metropolis(lattice, n, MC_sweep, T[i])
    plt.figure()
    plt.plot(T, M)
    plt.xlabel('T')
    plt.ylabel('M')
    plt.title('M vs. Temperature')
    plt.savefig('Part1.jpeg', format='jpeg')
    plt.show()

def Part_2():
    T = np.arange(1, 6.1, 0.1)
    n = np.array([5, 10, 20, 30, 40, 50, 75, 100])
    Tc = 3.404
    np.random.seed(0)
    for size in [5, 10]:
        MC_sweep = 10000 * (size ** 2)
        C = np.zeros(len(T))
        for i in range(len(T)):
            lattice = np.random.choice([-1, 1], size=(size, size))
            M, E, E_2 = Metropolis(lattice, size, MC_sweep, T[i])
            C[i] = ((E_2 - (E ** 2)) / (T[i] ** 2)) / (size ** 2)
        plt.figure()
        plt.plot(T, C)
        plt.xlabel('T')
        plt.ylabel('C/N')
        plt.title(f'C/N vs. Temperature at n={size}')
        plt.savefig(f'Part2_n={size}.jpeg', format='jpeg')
        plt.show()
    C_max = []
    for size in n:
        MC_sweep = 10000 * (size ** 2)
        lattice = np.random.choice([-1, 1], size=(size, size))
        M, E, E_2 = Metropolis(lattice, size, MC_sweep, Tc)
        C_max.append(((E_2 - (E ** 2)) / (Tc ** 2)) / (size ** 2))
    plt.figure()
    plt.plot(n, C_max)
    plt.xlabel('n')
    plt.ylabel(r'$C_{max}$/N')
    plt.title(r'$C_{max}$ vs. n')
    plt.savefig('Part2.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    Part_1()
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in string:
        if i == '1':
            Part_1()
        if i == '2':
            Part_2()

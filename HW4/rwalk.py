import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def rwalk_2D(N):
    max_walk = 10000
    xav = np.zeros(max_walk)
    x2av = np.zeros(max_walk)
    r2 = np.zeros(max_walk)
    timestep = np.arange(1, max_walk + 1, 1)
    for j in range(N):
        x = 0
        y = 0
        for i in range(max_walk):
            r = 2 * random.random()
            if r < 0.5:
                x += 1
            elif (r >= 0.5) & (r < 1):
                y += 1
            elif (r >= 1) & (r < 1.5):
                x -= 1
            else:
                y -= 1
            xav[i] += x + y
            x2av[i] += (x + y) ** 2
            r2[i] += (x ** 2) + (y ** 2)
    xav /= N
    x2av /= N
    r2 /= N
    return timestep, xav, x2av, r2

def Part_1():
    N = np.array([3, 10, 30, 50, 100])
    for i in range(len(N)):
        timestep, xav, x2av, r2 = rwalk_2D(N[i])
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(timestep, xav)
        plt.xlabel('Timestep')
        plt.ylabel('<$x_n$>')
        plt.title('<$x_n$> vs. t')
        plt.subplot(1, 2, 2)
        plt.plot(timestep, x2av)
        plt.xlabel('Timestep')
        plt.ylabel('<$(x_n)^2$>')
        plt.title('<$(x_n)^2$> vs. t')
        plt.suptitle(f'2D random walk with n={N[i]}')
        plt.tight_layout()
        plt.savefig(f'Part2_1_n={N[i]}.jpeg', format='jpeg')
        plt.show()

def Part_2():
    N = 100
    timestep, xav, x2av, r2 = rwalk_2D(N)
    plt.figure()
    plt.plot(timestep, r2)
    plt.xlabel('Timestep')
    plt.ylabel('<$r^2$>')
    plt.title('<$r^2$> vs. t')
    plt.savefig('Part2_2.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in number:
        if i == '1':
            Part_1()
        if i == '2':
            Part_2()

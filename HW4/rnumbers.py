import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def uniform_random(N):
    r = np.zeros(N)
    for i in range(len(r)):
        r[i] = random.uniform(0, 1)
    return r

def gaussian(x):
    return (1 / (1.0 * np.sqrt(2 * np.pi))) * np.exp(-x ** 2 / (2 * 1.0 ** 2))

def gaussian_random(N):
    x = np.linspace(-5, 5, N)
    y = gaussian(x)
    pmin = y.min()
    pmax = y.max()
    accept = []
    num = 0
    while num < N:
        rx = 10 * random.uniform(0, 1) - 5
        ry = random.uniform(pmin, pmax)
        if ry < gaussian(rx):
            accept.append(rx)
            num += 1
    result = np.array(accept)
    return result

def plot_random(r, N, p):
    weights = np.ones_like(r)/ len(r)
    subdivisions = [10, 20, 50, 100]
    plt.figure(figsize=(16, 12))
    for i, j in enumerate(subdivisions, 1):
        plt.subplot(2, 2, i)
        hist, bin_edges = np.histogram(r, j, density=True)
        edge_middles = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        if p == 1:
            plt.bar(edge_middles, hist * bin_width, width=bin_width)
        elif p == 2:
            plt.bar(edge_middles, hist, width=bin_width)
        if p == 2:
            x = np.linspace(-5, 5, N)
            y = gaussian(x)
            plt.plot(x, y, 'k', label='Gaussian')
        if p == 1:
            plt.title(f'Probability distribution with {j} subdivisions')
        elif p == 2:
            plt.title(f'Probability density distribution with {j} subdivisions')
        plt.xlabel('Number range')
        plt.ylabel('Probability')
    if p == 1:
        plt.suptitle(f'Uniform random number distribution with N={N}')
    elif p == 2:
        plt.suptitle(f'Non-uniform random number distribution with N={N}')
    plt.tight_layout()
    if p == 1:
        plt.savefig(f"Part1_1_N={N}.jpeg", format='jpeg')
    if p == 2:
        plt.savefig(f"Part1_2_N={N}.jpeg", format='jpeg')
    plt.show()

def Part_1():
    N = 1000
    r = uniform_random(N)
    plot_random(r, N, 1)
    N = 1000000
    r = uniform_random(N)
    plot_random(r, N, 1)

def Part_2():
    N = 1000
    r = gaussian_random(N)
    plot_random(r, N, 2)
    N = 1000000
    r = gaussian_random(N)
    plot_random(r, N, 2)

if __name__ == '__main__':
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in number:
        if i == '1':
            Part_1()
        if i == '2':
            Part_2()

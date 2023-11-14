import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

def particle_moves(x, y, grid):
    grid_new = grid.copy()
    p = False
    r = random.uniform(0, 2)
    if r < 0.5:
        y_trial = y + 1
        x_trial = x
    elif 0.5 <= r < 1:
        x_trial = x + 1
        y_trial = y
    elif 1 <= r < 1.5:
        y_trial = y - 1
        x_trial = x
    else:
        x_trial = x - 1
        y_trial = y
    if (0 <= x_trial < 60) and (0 <= y_trial < 40) and (grid[y_trial, x_trial] == 0):
        grid_new[y_trial, x_trial] = grid_new[y, x]
        grid_new[y, x] = 0
        p = True
    return grid_new, p

def Part_1():
    width = 60
    height = 40
    grid = np.zeros((height, width))
    grid[:, :width // 3] = 1
    grid[:, 2 * width // 3:] = 2
    iterations = 100000
    timestep = np.array([0, 100, 1000, 10000, 100000])
    snapshots = []
    snapshots.append(grid)
    for i in range(1, iterations + 1):
        while True:
            while True:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if (grid[y, x] != 0):
                    break
            grid, p = particle_moves(x, y, grid)
            if p:
                break
        if i in timestep:
            snapshots.append(grid)
    return grid, np.array(snapshots), timestep

def Part_2():
    grid, snapshots, timestep = Part_1()
    n_A = np.sum(grid == 1, axis=0) / 40
    n_B = np.sum(grid == 2, axis=0) / 40
    plt.figure(figsize=(12, 6))
    plt.plot(n_A, label='Population density of A')
    plt.plot(n_B, label='Population density of B')
    plt.xlabel('x')
    plt.ylabel('Population densities')
    plt.legend()
    plt.title('Linear population densities')
    plt.savefig('Part4_1.jpeg', format='jpeg')
    plt.show()
    colormap = ListedColormap(['white', 'red', 'blue'])
    for i in range(len(timestep)):
        plt.figure()
        plt.imshow(snapshots[i], cmap=colormap)
        plt.xlabel('x')
        plt.ylabel('y')
        if timestep[i] == 0:
            plt.title('Initial configuration')
        else:
            plt.title(f'Configuration at timestep={timestep[i]}')
        plt.savefig(f'Part_4_2_timestep={timestep[i]}.jpeg', format='jpeg')
        plt.show()

def Part_3():
    trials = 100
    nA = np.zeros(60)
    nB = np.zeros(60)
    for i in range(trials):
        grid, snapshots, timestep = Part_1()
        nA += np.sum(grid == 1, axis=0) / 40
        nB += np.sum(grid == 2, axis=0) / 40
    nA /= trials
    nB /= trials
    plt.figure(figsize=(12, 6))
    plt.plot(nA, label='Population density of A')
    plt.plot(nB, label='Population density of B')
    plt.xlabel('x')
    plt.ylabel('Population densities')
    plt.legend()
    plt.title('Linear population densities with 100 trials')
    plt.savefig('Part4_3.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in number:
        if i == '2':
            Part_2()
        if i == '3':
            Part_3()

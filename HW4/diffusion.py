import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def normal_distribution(x, sigma):
    return (1 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp((-x ** 2) / (2 * (sigma ** 2)))

def diffusion():
    D = 2
    T = 100
    dx = 0.5
    dt = 0.01
    x = np.arange(-50, 50, dx)
    N = len(x)
    initial_density = np.zeros(N)
    initial_density[N // 2 - 1:N // 2 + 1] = 1
    t = np.arange(0, T + dt, dt)
    density = np.zeros((N, len(t)))
    density[:, 0] = initial_density
    count = 0
    snapshots = []
    snapshots_time = []
    for n in range(1, len(t)):
        for i in range(1, N - 1):
            density[i, n] = density[i, n - 1] + (D * dt / (dx ** 2)) * (density[i + 1, n - 1] - 2 * density[i, n - 1] + density[i - 1, n - 1])
        count += 1
        if count == 2000:
            snapshots.append(density[:, n])
            snapshots_time.append(n * dt)
            count = 0
    plt.figure(figsize=(16, 6))
    for i in range(len(snapshots)):
        plt.plot(x, snapshots[i], label=f'Density profile at t={snapshots_time[i]:.0f}s')
        sigma_theoretical = np.sqrt(2 * D * snapshots_time[i])
        popt, pcov = curve_fit(normal_distribution, x, snapshots[i], p0=[sigma_theoretical])
        plt.plot(x, normal_distribution(x, *popt), '--', label='Fit at t={0}s with $\sigma$={1}, theoretical value of $\sigma$={2}'.format(snapshots_time[i], round(popt[0], 2), round(sigma_theoretical, 2)))
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Density profile of 1D diffustion')
    plt.savefig('Part3_2.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    diffusion()

import sys
import numpy as np
import matplotlib.pyplot as plt

def trajectory(theta):
    vi = 70
    density = 1.29
    area = 0.0014
    g = 9.8
    deltaT = 0.005
    mass = 0.046
    for j in range(0, len(theta)):
        plt.figure()
        t = np.arange(0, 10.005, deltaT)
        vx_i = vi * np.cos(theta[j] * np.pi / 180)
        vy_i = vi * np.sin(theta[j] * np.pi / 180)
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        vx = np.zeros(len(t))
        vy = np.zeros(len(t))
        vx[0] = vx_i
        vy[0] = vy_i
        for i in range(1, len(t)):
            vx[i] = vx[i - 1]
            x[i] = x[i - 1] + vx[i - 1] * deltaT
            vy[i] = vy[i - 1] - g * deltaT
            y[i] = y[i - 1] + vy[i - 1] * deltaT
        plt.plot(x, y , label='ideal trajectory')
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        vx = np.zeros(len(t))
        vy = np.zeros(len(t))
        vx[0] = vx_i
        vy[0] = vy_i
        v = vi
        for i in range(1, len(t)):
            vx[i] = vx[i - 1] - ((1 / 2) * density * np.abs(v) * area / mass) * vx[i - 1] * deltaT
            x[i] = x[i - 1] + vx[i - 1] * deltaT
            vy[i] = vy[i - 1] - g * deltaT - ((1 / 2) * density * np.abs(v) * area / mass) * vy[i - 1] * deltaT
            y[i] = y[i - 1] + vy[i - 1] * deltaT
            v = np.sqrt((vx[i] ** 2) + (vy[i] ** 2))
        plt.plot(x, y, label='smooth ball with drag')
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        vx = np.zeros(len(t))
        vy = np.zeros(len(t))
        vx[0] = vx_i
        vy[0] = vy_i
        v = vi
        C = 7.0 / v
        for i in range(1, len(t)):
            vx[i] = vx[i - 1] - (C * density * np.abs(v) * area / mass) * vx[i - 1] * deltaT
            x[i] = x[i - 1] + vx[i - 1] * deltaT
            vy[i] = vy[i - 1] - g * deltaT - (C * density * np.abs(v) * area / mass) * vy[i - 1] * deltaT
            y[i] = y[i - 1] + vy[i - 1] * deltaT
            v = np.sqrt((vx[i] ** 2) + (vy[i] ** 2))
            if v >= 14:
                C = 7.0 / v
            else: 
                C = 1 / 2
        plt.plot(x, y, label='dimpled golf ball with drag')
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        vx = np.zeros(len(t))
        vy = np.zeros(len(t))
        vx[0] = vx_i
        vy[0] = vy_i
        v = vi
        C = 7.0 / v
        for i in range(1, len(t)):
            vx[i] = vx[i - 1] - (C * density * np.abs(v) * area / mass) * vx[i - 1] * deltaT - 0.25 * vy[i - 1] * deltaT
            x[i] = x[i - 1] + vx[i - 1] * deltaT
            vy[i] = vy[i - 1] - g * deltaT - (C * density * np.abs(v) * area / mass) * vy[i - 1] * deltaT + 0.25 * vx[i - 1] * deltaT
            y[i] = y[i - 1] + vy[i - 1] * deltaT
            v = np.sqrt((vx[i] ** 2) + (vy[i] ** 2))
            if v >= 14:
                C = 7.0 / v
            else:
                C = 1 / 2
        plt.plot(x, y, label='dimpled golf ball with drag and spin')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trajectory of Golf')
        plt.legend()
        plt.savefig('Trajectory_of_Golf_with_theta={}.jpeg'.format(theta[j]), format = 'jpeg')
        plt.show()


if __name__ == "__main__":
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    theta = np.zeros(len(number))
    for i in range(0, len(number)):
        theta[i] = int(number[i])
    trajectory(theta)

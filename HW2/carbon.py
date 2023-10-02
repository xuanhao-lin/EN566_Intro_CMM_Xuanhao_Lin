import sys
import numpy as np
import matplotlib.pyplot as plt

def decay(deltaT):
    years = 20000
    decay_constant = -5700 / np.log(1 / 2)
    N0 = (10 ** (-12) / 14) * (6.022 * (10 ** 23))
    theoretical = N0 * np.exp(-1 * years / decay_constant)
    for j in range(0, len(deltaT)):
        plt.figure()
        t = np.arange(0, years + deltaT[j], deltaT[j])
        N = np.zeros(len(t))
        N[0] = N0
        for i in range(1, len(t)):
            N[i] = N[i - 1] - 1 / decay_constant * N[i - 1] * deltaT[j]
        plt.plot(t, N, label = 'time step=' + str(deltaT[j]) + ' years')
        plt.plot(t, N0 * np.exp(-t / decay_constant), label='analytical result')
        print(N[len(N) - 1])
        error = np.abs(N[len(N) - 1] - theoretical) / theoretical * 100
        print('the percentage deviation={}%'.format(error))
        plt.ylabel('Number of Isotope')
        plt.xlabel('time [years]')
        plt.title('Decay of Carbon Isotope')
        plt.legend()
        plt.savefig('Decay_of_Carbon_Isotope_with_Time_Step_Width={}.jpeg'.format(deltaT[j]), format = 'jpeg')
        plt.show()

if __name__ == "__main__":
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    deltaT = np.zeros(len(number))
    for i in range(0, len(number)):
        deltaT[i] = int(number[i])
    decay(deltaT)

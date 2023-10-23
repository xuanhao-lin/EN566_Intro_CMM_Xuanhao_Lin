import sys
import numpy as np
import matplotlib.pyplot as plt

def nonlinear(theta, omega, t, omega_D, alpha_D):
    return omega, -g / l * np.sin(theta) - 2 * gamma * omega + alpha_D * np.sin(omega_D*t)

def linear(theta, omega, t, omega_D):
    return omega, -g / l * theta - 2 * gamma * omega + alpha_D * np.sin(omega_D*t)

def linear_Runge_Kutta(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = 0
    omega[0] = 0
    for i in range(1, len(t)):
        x1_theta, x1_omega = linear(theta[i - 1], omega[i - 1], t[i - 1], omega_D)
        x2_theta, x2_omega = linear(theta[i - 1] + x1_theta * deltat / 2, omega[i - 1] + x1_omega * deltat / 2, t[i - 1] + deltat /2, omega_D)
        x3_theta, x3_omega = linear(theta[i - 1] + x2_theta * deltat / 2, omega[i - 1] + x2_omega * deltat / 2, t[i - 1] + deltat / 2, omega_D)
        x4_theta, x4_omega = linear(theta[i - 1] + x3_theta * deltat, omega[i - 1] + x3_omega * deltat, t[i - 1] + deltat, omega_D)
        theta[i] = theta[i - 1] + (x1_theta + 2 * x2_theta + 2 * x3_theta + x4_theta) * deltat / 6
        omega[i] = omega[i - 1] + (x1_omega + 2 * x2_omega + 2 * x3_omega + x4_omega) * deltat / 6
    return theta,omega

def nonlinear_Runge_Kutta(omega_D, alpha_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = 0
    omega[0] = 0
    for i in range(1, len(t)):
            x1_theta, x1_omega = nonlinear(theta[i - 1], omega[i - 1], t[i - 1], omega_D, alpha_D)
            x2_theta, x2_omega = nonlinear(theta[i - 1] + x1_theta * deltat / 2, omega[i - 1] + x1_omega * deltat / 2, t[i - 1] + deltat /2, omega_D, alpha_D)
            x3_theta, x3_omega = nonlinear(theta[i - 1] + x2_theta * deltat / 2, omega[i - 1] + x2_omega * deltat / 2, t[i - 1] + deltat / 2, omega_D, alpha_D)
            x4_theta, x4_omega = nonlinear(theta[i - 1] + x3_theta * deltat, omega[i - 1] + x3_omega * deltat, t[i - 1] + deltat, omega_D, alpha_D)
            theta[i] = theta[i - 1] + (x1_theta + 2 * x2_theta + 2 * x3_theta + x4_theta) * deltat / 6
            omega[i] = omega[i - 1] + (x1_omega + 2 * x2_omega + 2 * x3_omega + x4_omega) * deltat / 6
    return theta, omega

def linear_Euler_Cromer(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = 0
    omega[0] = 0
    for i in range(1, len(t)):
        omega[i] = omega[i - 1] + linear(theta[i - 1], omega[i - 1], t[i - 1], omega_D)[1] * deltat
        theta[i] = theta[i - 1] + omega[i] * deltat
    return theta, omega

def nonlinear_Euler_Cromer(omega_D, alpha_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    theta[0] = 0
    omega[0] = 0
    t = np.arange(0, 70 + deltat, deltat)
    for i in range(1, len(t)):
        omega[i] = omega[i - 1] + nonlinear(theta[i - 1], omega[i - 1], t[i - 1], omega_D, alpha_D)[1] * deltat
        theta[i] = theta[i - 1] + omega[i] * deltat
    return theta, omega

def Part_2(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta, omega = linear_Euler_Cromer(omega_D)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle('Euler-Cromer method')
    plt.savefig('Figure 1.jpeg', format='jpeg')
    plt.show()
    theta, omega = linear_Runge_Kutta(omega_D)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle('Runge-Kutta 4th order method')
    plt.savefig('Figure 2.jpeg', format='jpeg')
    plt.show()
    D = np.arange(0.5, 1.6, 0.1)
    theta_0 = np.zeros(len(D))
    phi = np.zeros(len(D))
    for i in range(len(D)):
        theta, omega = linear_Runge_Kutta(D[i])
        steady_theta = theta[-1000:]
        theta_0[i] = (max(steady_theta) - min(steady_theta)) / 2
        zero_crossings = np.where(np.diff(np.signbit(steady_theta)))[0]
        half_period = (len(steady_theta) - zero_crossings[-1]) * deltat
        remaining_time = (len(steady_theta) - zero_crossings[-1]) * deltat
        if steady_theta[-1] <= 0:
            phi[i] = (half_period - remaining_time) / half_period * np.pi * D[i]
        else:
            phi[i] = (half_period - remaining_time) / half_period * np.pi * D[i] + np.pi
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(D, theta_0, 'o-')
    plt.xlabel(r'$\omega_D$')
    plt.ylabel(r'$\theta_0$')
    plt.subplot(2, 1, 2)
    plt.plot(D, phi, 'o-')
    plt.xlabel(r'$\omega_D$')
    plt.ylabel(r'$\phi$')
    plt.suptitle('Resonance structure')
    plt.savefig('Figure 3.jpeg', format='jpeg')
    plt.show()
    half_max = max(theta_0) / 2
    above_half_max = []
    for i in range(len(theta_0)):
        if theta_0[i] > half_max:
            above_half_max.append(i)
    FWHM = D[above_half_max[-1]] - D[above_half_max[0]]
    print('FWHM=', FWHM)

def Part_3(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta, omega = linear_Runge_Kutta(omega_D)
    K = np.zeros(len(t))
    U = np.zeros(len(t))
    E = np.zeros(len(t))
    for i in range(len(t)):
        K[i] = 1 / 2 * m * (omega[i] * l) ** 2
        U[i] = m * g * l * (1 - np.cos(theta[i]))
        E[i] = K[i] + U[i]
    plt.figure(figsize=(10,15))
    plt.subplot(3, 1, 1)
    plt.plot(t, U)
    plt.xlabel('t [s]')
    plt.ylabel('Potential energy [J]')
    plt.subplot(3, 1, 2)
    plt.plot(t, K)
    plt.xlabel('t [s]')
    plt.ylabel('Kinetic energy [J]')
    plt.subplot(3, 1, 3)
    plt.plot(t, E)
    plt.xlabel('t [s]')
    plt.ylabel('Total energy [J]')
    plt.suptitle('Potential, kinetic, and total energy vs. time')
    plt.savefig('Figure 4.jpeg', format='jpeg')
    plt.show()

def Part_4(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    theta, omega = nonlinear_Euler_Cromer(omega_D, alpha_D)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle(r'Euler-Cromer method with $\alpha_D=0.2$')
    plt.savefig('Figure 5.jpeg', format='jpeg')
    plt.show()
    theta, omega = nonlinear_Runge_Kutta(omega_D, alpha_D)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle(r'Runge-Kutta 4th order method with $\alpha_D=0.2$')
    plt.savefig('Figure 6.jpeg', format='jpeg')
    plt.show()
    theta, omega = nonlinear_Euler_Cromer(omega_D, 1.2)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle(r'Euler-Cromer method with $\alpha_D=1.2$')
    plt.savefig('Figure 7.jpeg', format='jpeg')
    plt.show()
    theta, omega = nonlinear_Runge_Kutta(omega_D, 1.2)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\theta$(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\omega$(t)')
    plt.suptitle(r'Runge-Kutta 4th order method with $\alpha_D=1.2$')
    plt.savefig('Figure 8.jpeg', format='jpeg')
    plt.show()

def Part_5(omega_D):
    deltat = 0.01
    t = np.arange(0, 70 + deltat, deltat)
    alpha_D = np.array([0.2, 0.5, 1.2])
    theta_in = 0.001
    plt.figure()
    for j in alpha_D:
        theta, omega = nonlinear_Euler_Cromer(omega_D, j)
        theta_chaos = np.zeros(len(t))
        omega_chaos = np.zeros(len(t))
        omega_chaos[0] = 0
        theta_chaos[0] = theta[0] + theta_in
        for i in range(1, len(t)):
            omega_chaos[i] = omega_chaos[i - 1] + nonlinear(theta_chaos[i - 1], omega_chaos[i - 1], t[i - 1], omega_D, j)[1] * deltat
            theta_chaos[i] = theta_chaos[i - 1] + omega_chaos[i] * deltat
        delta_theta = np.abs(theta - theta_chaos)
        plt.plot(t, delta_theta, label=r'$\alpha_D={}$'.format(i))
        print('Lyapunov exponent:', np.polyfit(t, np.log(delta_theta), 1)[0])
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel(r'|$\Delta\theta(t)$|')
    plt.title('Non-linear pendulum with chaos')
    plt.savefig('Figure 9.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    m = 1
    g = 9.8
    l = 9.8
    gamma = 0.25
    alpha_D = 0.2
    omega_0 = np.sqrt(g / l)
    omega_D_resonance = omega_0
    string = sys.argv[1].split('=')
    number = string[1].split(',')
    for i in number:
        if i == '2':
            Part_2(omega_D_resonance)
        if i == '3':
            Part_3(omega_D_resonance)
        if i == '4':
            Part_4(omega_D_resonance)
        if i == '5':
            Part_5(0.666)

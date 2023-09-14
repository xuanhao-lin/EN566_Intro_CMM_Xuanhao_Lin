import sys
import numpy as np
import matplotlib.pyplot as plt

function_to_plot = {
    'cos': np.cos,
    'sin': np.sin,
    'sinc': lambda x: np.sinc(x / np.pi),
}

def plot_func(functions):
    x = np.arange(-10, 10.05, 0.05)
    plt.figure()
    for i in functions:
        plt.plot(x, function_to_plot[i](x), label=i)
    plt.title('Plot of Trigonometric Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    p = 0
    for i in range(1, len(sys.argv)):
        if '--print=' in sys.argv[i]:
            string = sys.argv[i].split('=')
            fmts = string[1].split(',')
            for j in fmts:
                plt.savefig("Plot_of_Trigonometric_Functions.{}".format(j), format=j)
            p = 1
    if not p:
        plt.show()

def write(filename, functions):
    x = np.arange(-10, 10.05, 0.05)
    with open(filename, 'w', encoding='ascii') as f:
        f.write('x ' + ' '.join(functions) + '\n')
        for i in x:
            f.write(f"{i} ")
            tmp = 0
            for j in functions:
                tmp += 1
                if tmp != len(functions):
                    f.write(f"{function_to_plot[j](i)} ")
                else:
                    f.write(f"{function_to_plot[j](i)}")
            f.write('\n')
        f.close()

def read_file(filename):
    with open(filename, 'r', encoding='ascii') as f:
        line = f.readline()
        values = line.split(' ')
        functions = values[1:]
        functions[len(functions)-1] = functions[len(functions)-1].strip()
        f.close()
    data = np.loadtxt(filename, skiprows=1)
    x = data[:, 0]
    y = data[:, 1:]
    plt.figure()
    for i in range(len(functions)):
        plt.plot(x, y[:, i], label=functions[i])
    plt.title('Plot of Trigonometric Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        if '--function=' in sys.argv[i]:
            string = sys.argv[i].split('=')
            functions = string[1].split(',')
            plot_func(functions)
    for i in range(1, len(sys.argv)):
        if '--write=' in sys.argv[i]:
            string = sys.argv[i].split('=')
            filename = string[1]
            write(filename, functions)
        if '--read_from_file=' in sys.argv[i]:
            string = sys.argv[i].split('=')
            filename = string[1]
            read_file(filename)


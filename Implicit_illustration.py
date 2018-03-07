"""
Illustration for understanding the behaviour of Implicit SGD
Shows how Implicit SGD and standard SGD find the optimum
of f(x) = x^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Parameters
limit = 2
grid_size = 0.001
learning_rate = 1.7
num_iterations = 5
x_initial = 1.75
plot_algorithm = "BOTH_ISGD_SGD"
assert plot_algorithm in {"BOTH_ISGD_SGD", "ISGD", "SGD"}

# Calculate file name for saving pdf
pdf_file_name = ('./Results/Plots/' + plot_algorithm + '.pdf')

with PdfPages(pdf_file_name) as pdf:
    plt.figure(figsize=(4, 2.5))  # Was (6, 4) in the paper, (4, 2.5) in the poster
    # Create grid and plot function values
    x_range = np.arange(-limit, limit, grid_size)
    plt.plot(x_range, x_range ** 2/2, 'k')

    # Calculate ISGD iterates
    x_ISGD = np.zeros(num_iterations)
    x_ISGD[0] = x_initial
    for t in range(num_iterations - 1):
        x_ISGD[t + 1] = x_ISGD[t] / (1 + learning_rate)

    # Calculate SGD iterates
    x_SGD = np.zeros(num_iterations)
    x_SGD[0] = x_initial
    for t in range(num_iterations - 1):
        x_SGD[t + 1] = x_SGD[t] * (1 - learning_rate)

    # Plot
    if plot_algorithm == "BOTH_ISGD_SGD":
        # Calculate Implicit SGD iterates
        plt.plot(x_SGD, x_SGD ** 2/2, '--bd', label='SGD')
        plt.plot(x_ISGD, x_ISGD ** 2/2, '-ro', markerfacecolor='none', label='ISGD')
        plt.legend(loc='upper center')
        plt.title('Implicit vs Vanilla SGD')

    elif plot_algorithm == "SGD":
        plt.plot(x_SGD, x_SGD ** 2/2, '-ro')
        plt.title('SGD iterates')

    elif plot_algorithm == "ISGD":
        plt.plot(x_ISGD, x_ISGD ** 2/2, '-ro')
        plt.title('ISGD iterates')

    # Create labels and show the plot
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$f(\theta) = \theta^2/2$')

    # Save figure
    pdf.savefig(bbox_inches='tight')
    # plt.show()

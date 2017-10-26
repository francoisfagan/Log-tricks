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
SGD_True_ISGD_False = False

# Calculate file name for saving pdf
pdf_file_name = ('./Results/Plots/'
                 + ('SGD' if SGD_True_ISGD_False else 'ISGD')
                 + '.pdf')

with PdfPages(pdf_file_name) as pdf:
    plt.figure(figsize=(6, 4))
    # Create grid and plot function values
    x_range = np.arange(-limit, limit, grid_size)
    plt.plot(x_range, x_range**2)

    # Calculate SGD iterates
    if SGD_True_ISGD_False:
        x_SGD = np.zeros(num_iterations)
        x_SGD[0] = x_initial
        for t in range(num_iterations - 1):
            x_SGD[t+1] = x_SGD[t] * (1 - learning_rate)
        plt.plot(x_SGD, x_SGD**2, '-ro')
        plt.title('SGD iterates')

    else:
        # Calculate Implicit SGD iterates
        x_ISGD = np.zeros(num_iterations)
        x_ISGD[0] = x_initial
        for t in range(num_iterations - 1):
            x_ISGD[t+1] = x_ISGD[t] / (1 + learning_rate)
        plt.plot(x_ISGD, x_ISGD**2, '-ro')
        plt.title('Implicit SGD iterates')

    # Create labels and show the plot
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$f(\theta) = \theta^2$')

    # Save figure
    pdf.savefig(bbox_inches='tight')
# plt.show()
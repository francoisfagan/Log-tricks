""" Code used to empirically understand the distribution of u in the learned denomenator softmax
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters

dist = random.normalvariate
params = [0,10]
repeat = 1000
scale_by_log_K = True

for num_classes in [10,100,1000,10000]:
    num_samples = num_classes * repeat

    # Sample u
    Z_inv_samples = []
    for repetition in range(repeat):
        new_samples = np.exp([dist(*params) for _ in range(num_classes)])
        new_samples = new_samples / sum(new_samples)
        Z_inv_samples.append(new_samples)
    Z_inv_samples = np.concatenate(Z_inv_samples)
    u_samples = - np.log(Z_inv_samples)
    u_samples = np.sort(u_samples)
    if scale_by_log_K:
        u_samples = u_samples - np.log(num_classes)

    # # Plot pdf
    # weights=np.ones_like(u_samples)/float(len(u_samples)) # Normalizes pdf
    # n, bins, patches = plt.hist(u_samples, 50, weights=weights)
    # mode = bins[np.argmax(n)]
    # print('Mode: ' + str(mode))

    # Plot cdf
    plt.plot(u_samples, np.arange(num_samples)/float(num_samples), label=num_classes)

#lower_bound_x = np.arange(0,1.0/num_classes, 0.001/num_classes)
#plt.plot(lower_bound_x, lower_bound_x*num_classes)

# Show plot
plt.title('Distribution of $u - \log(K)$')
plt.legend()
plt.show()
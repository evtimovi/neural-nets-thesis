import matplotlib.pyplot as plt 
import numpy as np
execfile('foo/euclidean_dist_colorferet_1000.txt')

flipped_dist=map(lambda x: 1/x, euclidean_dist)
bins = np.arange(0.02,0.065,0.005/4)
plt.hist(flipped_dist[:500], bins=bins, label='genuine')
plt.hist(flipped_dist[500:], bins=bins, label='imposter')
plt.xlabel('similarity (reciprocal euclidean)')
plt.ylabel('number of occurences')
plt.title('Genuine and Imposter Distributions')
plt.legend()
plt.show()

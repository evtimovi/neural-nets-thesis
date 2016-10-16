import matplotlib.pyplot as plt 
import numpy as np
execfile("vgg_vectors.txt")

flipped_dist=map(lambda x: 1/x, dist_distribution)
bins = np.arange(0.025,0.070,0.005/4)
plt.hist(flipped_dist[:500], bins=bins, label='genuine')
plt.hist(flipped_dist[500:], bins=bins, label='imposter')
plt.xlabel('similarity')
plt.ylabel('number of occurences')
plt.title('Genuine and Imposter Distributions')
plt.legend()
plt.show()

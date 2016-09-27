"""
Example application of digestive ripening algorithm for the partitioning of a 2 atoms-thick disc of Si atoms of diameter 100 Angstrom.
Script compares results between spectral clustering, k-means and digestive ripening.
"""

from __future__ import print_function

from quippy import Atoms, distance_map
from sklearn.cluster import KMeans, SpectralClustering
import sys
from time import time
import networkx as nx
import numpy as np

sys.path.insert(1, '../src')
import digestive_ripening

at = Atoms('si-pie-thin.xyz')

at.set_cutoff(3.06)
at.calc_connect()
dm = np.array(distance_map(at, at.n, at.n))
pos = at.get_positions()

print("Spectral clustering...")
t = time()
labels = SpectralClustering(7).fit_predict(pos)
time_elapsed = time() - t
at.set_array('assign_spectral', labels)
at.write("si-pie-thin-assign.xyz")
print("Time elapsed: %.2f s" % time_elapsed)
print('='*40)

print("k-means clustering...")
t = time()
labels = KMeans(7, max_iter=int(1e6)).fit_predict(pos)
time_elapsed = time() - t
at.set_array('assign_kmeans', labels)
at.write("si-pie-thin-assign.xyz")
print("Time elapsed: %.2f s" % time_elapsed)
print('='*40)

print("Digestive ripening...")
t = time()
assign = digestive_ripening.digestive_ripening(dm, at.get_array('assign_kmeans'), algorithm='fullrand', verbose=True)
time_elapsed = time() - t
at.set_array('assign_rand', assign)
at.write('si-pie-thin-assign.xyz')
print("Time elapsed: %.2f s" % time_elapsed)
print('='*40)


# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:29:54 2024

@author: Giles
"""

from pystates import all_spectrums, snapshot_dist
from scipy.spatial import distance
import pickle
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random
from random import randint

def LMDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	if w.size==0:
		print('Error: matrix is negative definite.')
		return []

	V = evecs[:,w]
	N = D.shape[1]
	Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
	Dm = D - np.tile(np.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= np.tile(np.mean(X,axis=1),(N, 1)).T

	_, evecs = sp.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T

# Example usage
with open("corrs.pkl","rb") as f:
    corr_dict = pickle.load(f)

eigenspectrums = all_spectrums(corr_dict,norm=True).T

# Using old distance implementation - note this is much slower with the same result
dist1 = snapshot_dist(eigenspectrums.T,norm=False)
mds = MDS(n_components=2,dissimilarity='precomputed',random_state=0)
results = mds.fit(dist1)
coords = results.embedding_

agg = AgglomerativeClustering().fit(coords)
labels = agg.labels_
colours = []
for i in range(np.max(labels)+1):
    colours.append('#%06X' % randint(0, 0xFFFFFF))
cmap=[]
for i in labels:
    cmap.append(colours[i])
plt.figure()
plt.scatter(coords[:,0],coords[:,1],c=cmap)

plt.figure()
plt.plot(labels,'k.')

# Using new distance implementation with MDS
dist2 = distance.cdist(eigenspectrums,eigenspectrums,'euclidean')
mds = MDS(n_components=2,dissimilarity='precomputed',random_state=0)
results = mds.fit(dist1)
coords = results.embedding_

agg = AgglomerativeClustering().fit(coords)
labels = agg.labels_
colours = []
for i in range(np.max(labels)+1):
    colours.append('#%06X' % randint(0, 0xFFFFFF))
cmap=[]
for i in labels:
    cmap.append(colours[i])
plt.figure()
plt.scatter(coords[:,0],coords[:,1],c=cmap)

plt.figure()
plt.plot(labels,'k.')

# Example using LMDS instead of MDS
landmarks = random.sample(range(0,eigenspectrums.shape[0],1),100)
dist3 = distance.cdist(eigenspectrums[landmarks,:],eigenspectrums,'euclidean')
coords = LMDS(dist3,landmarks,2)

agg = AgglomerativeClustering().fit(coords)
labels = agg.labels_
colours = []
for i in range(np.max(labels)+1):
    colours.append('#%06X' % randint(0, 0xFFFFFF))
cmap=[]
for i in labels:
    cmap.append(colours[i])
plt.figure()
plt.scatter(coords[:,0],coords[:,1],c=cmap)

plt.figure()
plt.plot(labels,'k.')

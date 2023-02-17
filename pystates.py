# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:52:20 2023

@author: Giles
"""

import numpy as np
import pickle
import lzma
from tqdm import tqdm,trange
import os

def claplacian(M,norm=True):
    if norm == True:
        L = np.diag(sum(M)) - M
        v = 1./np.sqrt(sum(M))
        return np.diag(v) * L * np.diag(v)
    else:
        return np.diag(sum(M)) - M
    
def eigenspectrum(L):
    eigvals = np.real(np.linalg.eig(L)[0])
    return -np.sort(-eigvals)

def all_spectrums(A_dict):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i = 0
    for key in tqdm(dict_keys):
        L = claplacian(A_dict[key],norm=False)
        eigenspectrums[:,i] = eigenspectrum(L)
        i+=1
    return eigenspectrums
    
def snapshot_dist(eigenspectrums,norm=True):
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N,N))
    for i in trange(N):
        for j in range(N):
            dist[i,j] = np.sqrt(np.sum(np.power((eigenspectrums[:,i]-eigenspectrums[:,j]),2)))
            if norm == True:
                if max(max(eigenspectrums[:,i]),max(eigenspectrums[:,j])) > 1e-10:
                    dist[i,j] = dist[i,j]/np.sqrt(max((np.sum(np.power(eigenspectrums[:,i],2))),(np.sum(np.power(eigenspectrums[:,j],2)))))
                else:
                    dist[i,j] = 0
                    
    return dist

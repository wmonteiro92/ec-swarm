# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:20:34 2020

@author: wmont
"""

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from scipy.optimize import differential_evolution
from SwarmPackagePy import ba, fa, cso
from scipy.stats import friedmanchisquare

import numpy as np
import pandas as pd

def ackley(x):
    a = (x ** 2)
    b = np.cos(2 * np.pi * x)
    
    if len(x.shape) == 1:
        n = float(x.shape[0])
        return -20.0 * np.exp(-0.2 * np.sqrt((1 / n) * a.sum())) - np.exp(
                (1 / n) * b.sum()) + 20.0 + np.exp(1)
    else:
        # pyswarms work with multiple solutions per function call
        n = float(x.shape[1])
        return -20.0 * np.exp(-0.2 * np.sqrt((1 / n) * a.sum(axis=1))) - np.exp(
                (1 / n) * b.sum(axis=1)) + 20.0 + np.exp(1)

def rastrigin(x):
    a = 10.0
    j = (x ** 2.0 - a * np.cos(2.0 * np.pi * x))
    
    if len(x.shape) == 1:
        return (a * x.shape[0]) + j.sum()
    else:
        # pyswarms work with multiple solutions per function call
        return (a * x.shape[1]) + j.sum(axis=1)

def sphere(x):
    if len(x.shape) == 1:
        return (x ** 2.0).sum()
    else:
        # pyswarms work with multiple solutions per function call
        return (x ** 2.0).sum(axis=1)

def run_algorithm(fun, algorithm):
    max_iter = 1000
    pop_size = 10
    
    function_name = fun[0]
    bounds = fun[1]
    
    if algorithm == 'de':
        #bounds = [(0, 1), (0, 1)]
        result = differential_evolution(function_name, bounds=[tuple(bounds)]*2, 
                                        maxiter=max_iter, popsize=pop_size)
        optimizer = None
        stats = (result.fun, result.x)
    elif algorithm == 'pso-global':
        # https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html#module-pyswarms.single.global_best
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=pop_size, dimensions=2,
                                            options=options,
                                            bounds=([bounds[0]]*2, [bounds[1]]*2))
        stats = optimizer.optimize(function_name, iters=max_iter)
    elif algorithm == 'pso-local':
        # https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html#module-pyswarms.single.local_best
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        optimizer = ps.single.LocalBestPSO(n_particles=pop_size, dimensions=2,
                                           options=options,
                                           bounds=([bounds[0]]*2, [bounds[1]]*2))
        stats = optimizer.optimize(function_name, iters=max_iter)
    elif algorithm == 'firefly':
        optimizer = fa(n=pop_size, function=function_name, lb=bounds[0], ub=bounds[1],
                 dimension=2, iteration=max_iter)
        
        best_agent = np.array(optimizer.get_Gbest())
        stats = (function_name(best_agent), best_agent)
    elif algorithm == 'cuckoo':
        optimizer = cso(n=pop_size, function=function_name, lb=bounds[0], ub=bounds[1],
                 dimension=2, iteration=max_iter)
        
        best_agent = np.array(optimizer.get_Gbest())
        stats = (function_name(best_agent), best_agent)
    elif algorithm == 'bat':
        optimizer = ba(n=pop_size, function=function_name, lb=bounds[0], ub=bounds[1],
                 dimension=2, iteration=max_iter)
        
        best_agent = np.array(optimizer.get_Gbest())
        stats = (function_name(best_agent), best_agent)
    
    return stats

# creating a DataFrame to store the data
stats = []
fitness_matrix = []

function_list = [(ackley, [-32, 32]), (rastrigin, [-5.12, 5.12]), (sphere, [-10, 10])]
algorithm_list = ['de', 'pso-global', 'pso-local', 'firefly', 'cuckoo', 'bat']

for function_name in function_list:
    for algorithm in algorithm_list:
        # creating a empty list to store all the results found for this combination
        fitness = []
    
        for i in range(30):
            # appending the results of the current run to the fitness list
            fitness += [run_algorithm(function_name, algorithm)[0]]
        
        # including the results in the matrix
        fitness_matrix.append([function_name[0].__name__, algorithm, fitness])
        
        # getting the results out of the current combination
        stats.append([function_name[0].__name__, algorithm,
                     np.min(fitness), np.max(fitness), np.mean(fitness),
                     np.median(fitness), np.std(fitness)])

# creating a DataFrame (an easily manageable table) containing all the results
df_stats = pd.DataFrame(stats, columns=['Benchmark', 'Algorithm', 'Min',
                                        'Max', 'Mean', 'Median', 'StdDev'])

# generating the Friedman test results for each problem  
# https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/anova/how-to/friedman-test/interpret-the-results/key-results/
stats_measurements = []
for function_name in function_list:
    measurements = []
    for combination in fitness_matrix:
        if combination[0] == function_name[0].__name__:
            measurements.append(combination[2])
    
    _, pvalue = friedmanchisquare(*measurements)
    stats_measurements.append([function_name[0].__name__, pvalue])

# creating a DataFrame (an easily manageable table) containing all the results
df_stats_measurements = pd.DataFrame(stats_measurements, 
                                     columns=['Benchmark', 'Friedman p-value'])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


WORKING_DIR = 'build'

TEST_CONFIG = { 'KF' : ['test_KF', 'testKF_target.txt', 'testKF_cor_mean.txt'], # [folder name, log files]
    'UKF' : ['test_UKF_nonlinear', 'testUKF_target.txt', 'testUKF_cor_mean.txt'],
    'SIS' : ['test_SIS_nonlinear', 'testSIS_target.txt', 'testSIS_cor_particles.txt', 'testSIS_cor_weights.txt'] }

def read_log(filename):
    data = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            floats = [float(x) for x in line.split()] 
            data.append(floats)
    return data

def best_particle_data(corrected_data, weights_data):
    num_particles = len(weights_data[0])
    best_data = []
    for i in range(len(weights_data)):
        idx = weights_data[i].index(max(weights_data[i]))
        best_data.append(corrected_data[i*num_particles+idx])
    return best_data

def plot_target(target_data, ax):
    x = []
    #y = []
    for data in target_data:
        x.append(data[0])
        #y.append(data[2])
        #ax.scatter(data[0], data[2],color='k',marker='o')
    
    ax.plot(x,color='k',marker='o')

def plot_corrected(corrected_data, ax, color):
    x = []
    #y = []
    for data in corrected_data:
        x.append(data[0])
        #y.append(data[2])
    #ax.plot(x, y,color=color,marker='x',linestyle='-')
    ax.plot(x, color=color,marker='x',linestyle='-')

def read_data(conf_name):
    target_data = read_log(os.path.join(WORKING_DIR, TEST_CONFIG[conf_name][0], TEST_CONFIG[conf_name][1]))
    corrected_data = read_log(os.path.join(WORKING_DIR, TEST_CONFIG[conf_name][0], TEST_CONFIG[conf_name][2]))

    if len(corrected_data) != len(target_data):
        if len(corrected_data) > len(target_data) and len(corrected_data) % len(target_data) == 0:
            num_particles = len(corrected_data) / len(target_data)

            weights_data = read_log(os.path.join(WORKING_DIR, TEST_CONFIG[conf_name][0], TEST_CONFIG[conf_name][3]))
            assert len(weights_data) == len(target_data) # same simulation steps
            assert len(weights_data[0]) == num_particles # same particle counts

            print ('num_particles', num_particles) 
            corrected_data = best_particle_data(corrected_data, weights_data)
        else:
            print ('bad records', len(target_data), len(corrected_data))
            corrected_data = None
    
    return target_data, corrected_data

# recursively compare list data    
def compare_data(a, b):
    if len(a) == len(b):
        same = True
        for i in range(len(a)):
            if hasattr(a[i], '__len__') and hasattr(b[i], '__len__'):
                same = same and compare_data(a[i], b[i])
            else:
                same = same and (a[i] == b[i])
        return same
    else:
        return False


# create a figure and axis
fig, ax = plt.subplots()

target_data0, corrected_data = read_data('UKF')
plot_target(target_data0, ax)
plot_corrected(corrected_data, ax, [0.7,0.7,0.7])

target_data, corrected_data = read_data('SIS')
assert compare_data(target_data0, target_data) #same target
plot_corrected(corrected_data, ax, [0.5,0.5,0.5])

# set a title and labels
ax.set_title('Dataset')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

plt.show()

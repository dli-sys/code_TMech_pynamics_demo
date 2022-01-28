# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:22:12 2022

@author: dongting
"""

import numpy
import socket
import time
import os
import datetime
import matplotlib.pyplot as plt
import math
import yaml
import logging
import time
import matplotlib
import matplotlib.pyplot as plt


from scipy.signal import butter,filtfilt

w_locking_force = numpy.genfromtxt('w_lock.csv')
w_locking_force[:,-4:-1] = w_locking_force[:,-4:-1]-w_locking_force[0,-4:-1]

wo_locking_force = numpy.genfromtxt('wo_lock.csv')
wo_locking_force[:,-4:-1] = wo_locking_force[:,-4:-1]-wo_locking_force[0,-4:-1]
# Filter requirements.
T = w_locking_force[-1,0]-w_locking_force[0,0]     # Sample Period

fs = len(w_locking_force)/T     # sample rate, Hz
cutoff = 0.5   # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    from scipy.signal import butter,filtfilt
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
y1 = butter_lowpass_filter(w_locking_force[:,3], cutoff, fs, order)
y2 = butter_lowpass_filter(wo_locking_force[:,3], cutoff, fs, order)

dis1 = w_locking_force[:,-2]*numpy.sqrt(2)*1000-40
dis2 = wo_locking_force[:,-2]*numpy.sqrt(2)*1000-40

plt.figure()
plt.plot(dis1,w_locking_force[:,3],'b',alpha=0.2)
plt.plot(dis1,y1,'b-')
plt.plot(dis2,wo_locking_force[:,3],'r',alpha=0.2)
plt.plot(dis2,y2,'r--')

plt.plot(20,-0.46,marker="o",alpha=0.25,markersize=10)

plt.text(15, -6, "Before contact")
plt.text(20, -6, "After contact")
plt.text(20.5, 0.5, "Lock")

plt.xlim(0,80)

plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("Distance (mm)")
plt.ylabel("Force (N)")
plt.show()
plt.vlines(20,-10,1)


plt.plot(dis1,w_locking_force[:,3],'b',alpha=0.2)

plt.plot(w_locking_force[:,0]-w_locking_force[0,0],w_locking_force[:,3],'b-')
plt.vlines(w_locking_force[3648,0]-w_locking_force[0,0],-10,1)


plt.plot(dis2,wo_locking_force[:,3],'r',alpha=0.2)
plt.plot(dis2,y2,'r--')




def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return array[idx]
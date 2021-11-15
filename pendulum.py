#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:29:47 2021

@author: andreas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


########################## Inputs #############################################
# It is quite important that the length of the pendulum is measured accurately 
# for good results
L = 0.44 #m
g = 9.81 #m/s**2

# Reads in the data as it comes from Tracker. The angle is supposed to be in 
# degree
# Further, it seems that the code gives the best results if the dataset starts 
# with the pendulum at one of the turning points, i.e. zero angular velocity
dataset = np.loadtxt("83degree.csv", skiprows=2, delimiter='\t')
# There is also an option for the students to read in the x coordinate only
# However, reading in the angle is preferable
input_is_x_coordinates = False

# This converts the input into the angle of the pendulum in radians
if(input_is_x_coordinates):
    phi_exp = np.arcsin(dataset[:,1]/L)
else:
    phi_exp = dataset[:,1]*np.pi/180


# This function calculates the angle and angular speed of a mathematical 
# pendulum. Input parameters are:
# t - time in seconds (array)
# amplitude - in radians
# phi0 - initial angle in radians
# returns: Array containing the angle and angular velocity as a fct of time
def mathematical_pendulum(t, amplitude, phi0=0):
    omega = np.sqrt(g/L)
    return [amplitude*np.sin(omega*t+phi0/amplitude*np.pi/2), \
            amplitude*omega*np.cos(omega*t+phi0/amplitude*np.pi/2)]


# pend calculates the derivatives of the angle and the angular velocity
# for every time step in t. This is needed for the numerical 
# calculation of the pendulums motion.
# Inputs:
# y - Array that contains the angle and the angular velocity at the timestep t 
# t - time in seconds
# c - characteristic constant g/L
# returns the derivatives of y
def pend(y, t, c=g/L):
    theta, omega = y
    dydt = [omega, - c*np.sin(theta)]
    return dydt


# This function calculates the angle and angular speed of a pendulum. 
# For this it solves the differential equation numerically, instead of 
# using an approximation for the sine function.
# Note that it does not consider friction.
# Input parameters are:
# t - time in seconds (array)
# amplitude - in radians
# phi0 - initial angle in radians
# returns: Array containing the angle and angular velocity as a fct of time
def numerical_pendulum(t, phi0, omega0):
    solution = odeint(pend, [phi0, omega0], t)
    return solution[:,0], solution[:,1]
    

# Extracting the time of the experimental data
t_exp = dataset[:,0]
# Phi0 is the initial angle of the pendulum
phi0 = phi_exp[0]
# Calculating the derivative of phi as it is needed for the numerical 
# calculation and plots
omega_exp = np.gradient(phi_exp, t_exp, edge_order=2)
# Initial angular velocity
omega0 = omega_exp[0]
# Maximum amplitude of the pendulums motion
amplitude = np.max(abs(phi_exp))

# Extraction of the timeframe the measurement data was taken from. 
# This is needed for the plots
xlims = [np.min(t_exp),np.max(t_exp)]
# Creation of an array that represents smaller timesteps than the experimental 
# data.This is done to get a smoother graph for the theoretical calculations
t = np.linspace(xlims[0], xlims[1], 1000)


# Thereoretical calculations for the mathematical pendulum and the numerical 
# solution.
phi_mathematical, omega_mathematical = mathematical_pendulum(t, amplitude, phi0)
phi_numerical, omega_numerical = numerical_pendulum(t, phi0, omega0)


# Creating plots for the angle as function of time
# Here I thought that the students should put their own labels for the axis
# into the code
plt.plot(t_exp, phi_exp, "o",  label='experiment label')
plt.plot(t, phi_numerical, label='numerical label')
plt.plot(t, phi_mathematical, label='mathematical label')
# The location of the legend can also be manually be chosen according to this 
# chart: https://www.geeksforgeeks.org/change-the-legend-position-in-matplotlib/
plt.legend(loc='best')
plt.ylabel('y-label')
plt.xlabel('x-label')
plt.xlim(xlims)
plt.grid()
plt.show()
plt.close()

# Creating plots for the anglular velocity as function of time
# Here I thought that the students should put their own labels for the axis
# into the code
plt.plot(t_exp, omega_exp, "o",  label='experiment label')
plt.plot(t, omega_numerical, label='numerical label')
plt.plot(t, omega_mathematical, label='mathematical label')
plt.legend(loc='best')
plt.ylabel('y-label')
plt.xlabel('x-label')
plt.xlim(xlims)
plt.grid()
plt.show()


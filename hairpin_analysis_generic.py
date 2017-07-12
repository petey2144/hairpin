# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:21:18 2017
FOR HELIUM DATA - analyzes pressure and mixture sweeps
@author: Dave Peterson
"""
#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
from scipy import optimize
import os
import subprocess
import glob
import itertools
from mpl_toolkits.axes_grid.axislines import Subplot
import prettyplotlib as ppl
 #from fityk import Fityk -- one day this will work...

#Reads in the fitted hairpin peaks data
#and changes it into a format that can be fed into the sheath & pressure correction

fwhm = np.genfromtxt('/home/david/Downloads/hairpin_processed_FWHM_floating_argon_60W.csv', delimiter=' ')
center = np.genfromtxt('/home/david/Downloads/hairpin_processed_Center_floating_argon_60W.csv', delimiter=' ')

Q_Measured = np.zeros([len(fwhm)])
Q_Plasma = np.zeros([len(fwhm)])
Q_Vacuum = center[0]/fwhm[0]

Pressure_60W = np.linspace(0.1,1.0,10) #this will change for different measurement points

for i in range(1,len(fwhm)):
    Q_Measured[i] = center[i]/fwhm[i]
    Q_Plasma[i] = (1/((1/Q_Measured[i]) - (1/Q_Vacuum))) #Calculate plasma Q factor from measurements
    
Q_Measured = np.trim_zeros(Q_Measured)
Q_Plasma = np.trim_zeros(Q_Plasma)

ppl.plot(Pressure_60W, Q_Plasma, 'bo')
plt.xlabel('Pressure [Torr]')
plt.ylabel('Q Plasma [a.u.]')
plt.show()

#Does the sheath and pressure correction

N_collisionfreq = 1000 #mesh size for collision frequency constant sweep
N = len(Pressure_60W) #Number of data points in pressure to be processed
T_e_guess = 3.2 # electron temperature guess (will be replaced with estimation from loop probe) [eV] 

a = 0.00022 # hairpin wire radius [m]
w = 0.00185 # width between hairpin tines [m]
sigma_i = 1E-14 #argon e-n collision cross section [cm^2] (may need to include this in iteration cycle, or make temperature dependent)
Avagadro = 6.022E23 # [atoms/mol]
neutral_gas_temp = 298.15 #neutral gas temperature used to calculate neutral gas density [K]
R_gas = 62.36367 #gas constant [L*Torr/K*mol]
convergence_criteria = 1E-5 #used in sheath correction loop
c = 2.9979E8 #speed of light [m/s]
epsilon_naught = 8.854E-12 #permittivity of free space [F/m]

n_g = (Pressure_60W/(R_gas*neutral_gas_temp))*Avagadro*(1/1000) #obtain neutral gas density array

collfreqconstant = np.linspace(1.0E9, 7E9, N_collisionfreq) #matrix of assumed pressure normalized collision frequency values
collfreq = np.outer(collfreqconstant, Pressure_60W) #creates matrix of collision frequencies at a specific pressure

Density = 1E10*((center/1E9)**2 - (center[0]/1E9)**2)/0.8062 #uncorrected electron density from centerline resonance frequency, converts resonance freq in Hz to GHz [cm^-3]
Density = np.trim_zeros(Density) #removes leading zero from vacuum resonance

Final_Density = np.zeros((N_collisionfreq, N))
Final_sheath_width = np.zeros(N)
Final_b = np.zeros(N)

for j in range(0,N_collisionfreq):
    
    collision_correction = 1/(1 + (collfreq[j]/(2*np.pi*center[0]))**2)
    Density_pressure_corrected = Density/collision_correction
    
    for i in range(0, N): 
        
        convergence = 1
        Density_corrected = Density_pressure_corrected[i]
        
        while convergence > convergence_criteria:
            
            lambda_ds = 7.43*np.sqrt(T_e_guess/Density_corrected) #debye length at sheath edge in [m]
            sheath_width = 2.0*lambda_ds #this sheath assumes a 2 Debye length sheath
            b = sheath_width + a 
                
            sheath_correction = 1 - ((center[0]**2)/(center[i+1]**2))*((np.log((w-a)/(w-b)) + np.log(b/a))/np.log((w-a)/a)) #uses more copmlicated sheath definition, assumes sheath is completely devoid of electrons
            
            Density_updated = Density_pressure_corrected[i]/sheath_correction
            convergence = abs(Density_updated - Density_corrected)/Density_updated #update convergence criteria
            
            if convergence < convergence_criteria:
                Final_Density[j][i] = Density_updated
                Final_sheath_width[i] = sheath_width
                Final_b[i] = b
                
            Density_corrected = Density_updated #updates the last iterations density

fig, ax = plt.subplots(1)
ax.plot(Pressure_60W, Density, 'bo', label='Uncorrected Density',)
spines_to_remove = ['top', 'right']
for spine in spines_to_remove:
    ax.spines[spine].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
for i in range(0,N_collisionfreq):
    plt.plot(Pressure_60W, Final_Density[i],)
plt.xlabel('Pressure [Torr]')
plt.ylabel('Electron Density [cm^-3]')
plt.legend(loc='upper center')
#plt.savefig('Pressure_and_Sheath_Correction_Sweep.png', bbox_inches='tight') #saves plot
plt.show()

#Takes all sheath corrections (with different collision frequency assumptions)
#and iteratively solves for the correct true e-n collision frequency

measured_collision_freq = np.zeros((N_collisionfreq, N))
difference = np.zeros((N, N_collisionfreq))
true_collision_freq = np.zeros(N)

for j in range(0,N_collisionfreq):
    for i in range(0,N):
        def f(x):
            return ((((2*np.pi*center[i+1])**2 + x**2)/((2*np.pi*9000*np.sqrt(Final_Density[j][i]))**2) -1)*(2*np.pi*center[i+1]/x) - Q_Plasma[i]) #assumes loaded transmission line
        
        measured_collision_freq[j][i] = optimize.newton(f, 1E9, tol=6E-6, maxiter=750) #finds zero crossing of function f
        
        difference[i][j] = 100*abs(measured_collision_freq[j][i] - collfreq[j][i])/collfreq[j][i]#find minimum of %difference of input collision freq to output
        
        true_collision_freq[i] = collfreqconstant[difference[i].argmin()] #selects the pressure normalized collision frequency that corresponds to minimum of %different

final_measured_collision_freq_60W = Pressure_60W*true_collision_freq #multiply array of measured pressure normalized collision frequencies by pressure to obtain measured collision frequency

marker = ['.', '', '']
colors = ['r', 'b', 'g', 'c', 'k',]
linestyle = ['dashed', 'solid','dashdot', 'dotted', ':']

for i in range(0,N):
    ppl.plot(collfreqconstant, difference[i], linestyle=linestyle[i%5], color=colors[i%5])
plt.xlabel('Assumed Normalized Collision Frequency [Hz/Torr]')
plt.ylabel('Change from assumed to measured [%]')
plt.show()

#determination of error bars based on max and min assumed electron temps and floating potentials - T_e range:1-10 eV  V_f range: 5-40 V - larger values produce smaller collision freqs
#final_measured_collision_freq_max = final_measured_collision_freq_60W
#y_err_max = final_measured_collision_freq_max - final_measured_collision_freq_60W

#final_measured_collision_freq_min = final_measured_collision_freq_60W
#y_err_min = final_measured_collision_freq_60W - final_measured_collision_freq_min #one must be commented out during run

argon_legend_list = ['0.2 Torr', '0.3 Torr', '0.4 Torr', '0.5 Torr', '0.6 Torr', '0.7 Torr', '0.8 Torr', '0.9 Torr']

fig, ax = plt.subplots(1)
for i in range(1,N):
    true_collision_freq[i] = collfreqconstant[difference[i].argmin()]
    ax.plot(collfreqconstant, difference[i], marker=marker[i%3], linestyle=linestyle[i%5], color=colors[i%5])
spines_to_remove = ['top', 'right']
for spine in spines_to_remove:
    ax.spines[spine].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.ylim([0, 40])
plt.xlim([0.5E9, 7E9])
plt.xlabel('Assumed Normalized Collision Frequency [Hz/Torr]')
plt.ylabel('Change from assumed to measured [%]')
plt.legend(argon_legend_list, loc='best', prop={'size':10})
#plt.savefig('Percent_Difference_from_assumed_to_measured_zoomed.png', bbox_inches='tight') #saves plot
plt.show()

ppl.plot(Pressure_60W, final_measured_collision_freq_60W, 'bo',)
plt.xlabel('Pressure [Torr]')
plt.ylabel('Collision Frequency [Hz]')
#plt.errorbar(Pressure, final_measured_collision_freq, yerr=[y_err_min, y_err_max], fmt='bo')
#plt.savefig('Final_Measured_Collision_Freq.png', bbox_inches='tight') 
plt.show()

#These values have been obtained from HPEM (plasma simulation)
HPEM_PRESSURE = [0.2, 0.6, 0.8, 1.0, 1.4, 1.8, 2.0]
HPEM_COLF = [4.83E8, 1.19E9, 1.50E9, 1.72E9, 2.21E9, 2.89E9, 3.0E9]

HPEM_MAX_PRESSURE = [0.2, 0.6, 0.8, 1.0, 1.4, 1.8, 2.0]
HPEM_MAX_COLF = [6.27E8, 2.01E9, 2.65E9, 3.33E9, 4.33E9, 5.58E9, 6.09E9]

#Plot Data
fig, ax = plt.subplots(1)
ax.plot(Pressure_60W, final_measured_collision_freq_60W, 'bo',)
plt.xlabel('Pressure [Torr]')
plt.ylabel('Collision Frequency [Hz]')
spines_to_remove = ['top', 'right']
for spine in spines_to_remove:
    ax.spines[spine].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')    
#plt.errorbar(Pressure, final_measured_collision_freq, yerr=[y_err_min, y_err_max], fmt='bo')
plt.plot(HPEM_MAX_PRESSURE, HPEM_MAX_COLF, '--', color = 'g')
plt.plot(HPEM_PRESSURE, HPEM_COLF, color = 'r')
plt.xlim([0, 2.1])
plt.legend(['Hairpin','HPEM', 'BOLSIG+'], loc='upper center')
#plt.savefig('Argon_Pressure_Sweep.png', bbox_inches='tight')
plt.show()

ppl.plot(Pressure_60W, true_collision_freq, 'bo', label = 'Loaded Transmission Line')
plt.xlabel('Pressure [Torr]')
plt.ylabel('Pressure Normalized Collision Frequency [Hz/Torr]')
#plt.ylim([1.5E9, 4.0E9])
#plt.savefig('Argon_Pressure_Normalized_Collision_Freq.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1)
ax.plot(Pressure_60W, Density, 'bo', label='Uncorrected Density',)
Final_True_Density_argon_floating = np.zeros(N)
spines_to_remove = ['top', 'right']
for spine in spines_to_remove:
    ax.spines[spine].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
for i in range(0,N):
    Final_True_Density_argon_floating[i] = Final_Density[difference[i].argmin()][i]
plt.plot(Pressure_60W, Final_True_Density_argon_floating, 'rs', label='Final Density')
plt.xlabel('Pressure [Torr]')
plt.ylabel('Electron Density [cm^-3]')
plt.legend(loc='best')
#plt.savefig('Argon_Floating_Density_Profile.png', bbox_inches='tight')
plt.show()


Uncorr_Density_Argon_floating = Density

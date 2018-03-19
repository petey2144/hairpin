# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:15:43 2017

@author: David J Peterson
"""
#----------------
#import libraries
#----------------
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
#from fityk import Fityk -- this library does not work (but if it did work, the Fityk fitting process could be done from this script)

#------------------------------------------------------
#Read in the raw hairpin data and apply the calibration
#------------------------------------------------------

dirname = '/media/david/DAVE USB/February 24/' #location of raw data
calibration_dirname = '/media/david/DAVE USB/Grounded Hairpin Calibration/' #location of calibration data

helium_20W_pressure_sweep_files = glob.glob(dirname + "/Feb_24_grounded_helium_20W_*") #
mixture_sweep_20W_files = glob.glob(dirname + "/Feb_24_grounded_argon_helium_20W_*")
vacuum_file = glob.glob(dirname+ "/*vacuum2.csv") #files need this naming convention
plasma_calibration_files = glob.glob(calibration_dirname + "/*plasma_cal.csv") #plasma calibration data (hairpin tines shorted and immersed in plasma)
vacuum_calibration_file = glob.glob(calibration_dirname + "/*vacuum_cal.csv") #vacuum calibration data

vacuum = np.genfromtxt(vacuum_file[0], skip_header=7, usecols=1) #raw data from vacuum sweep
frequency = np.recfromcsv(helium_20W_pressure_sweep_files[0], skip_header=6, usecols=0) #frequency range used, uses recfromcsv because genfromtxt did not recognize data, so header is slightly different
helium_20W_pressure_sweep = np.zeros([len(helium_20W_pressure_sweep_files), len(vacuum)]) #raw data for different plasma conditions/experiments
mixture_sweep_20W = np.zeros([len(mixture_sweep_20W_files), len(vacuum)]) #raw data for different plasma conditions/experiments
plasma_calibration = np.zeros([len(plasma_calibration_files[0]), len(vacuum)]) #calibration sweep for plasma, typically only 1
helium_20W_pressure_sweep_calibrated = np.zeros([len(helium_20W_pressure_sweep_files), len(vacuum)])
mixture_sweep_calibrated_20W = np.zeros([len(mixture_sweep_20W_files), len(vacuum)])

vacuum_calibration = np.genfromtxt(vacuum_calibration_file[0], skip_header=7, usecols=[1])

for i in range(0,len(plasma_calibration_files)):
    plasma_calibration[i] = np.genfromtxt(plasma_calibration_files[i], skip_header=7, usecols=[1]) #the loop is included just in case multiple plasma calibration files are needed
    
for i in range(0,len(helium_20W_pressure_sweep_files)):
    helium_20W_pressure_sweep[i] = np.genfromtxt(helium_20W_pressure_sweep_files[i], skip_header=7, usecols=[1]) #location of data in csv file may vary
    helium_20W_pressure_sweep_calibrated[i] = helium_20W_pressure_sweep[i] - plasma_calibration[0] #calibration is done by simply subtracting off the transmission line waveform
    
for i in range(0,len(mixture_sweep_20W_files)):
    mixture_sweep_20W[i] = np.genfromtxt(mixture_sweep_20W_files[i], skip_header=7, usecols=[1])
    mixture_sweep_calibrated_20W[i] = mixture_sweep_20W[i] - plasma_calibration[0]

#--------------------
#plot calibrated data
#--------------------

vacuum_calibrated = vacuum - vacuum_calibration #calibrated vacuum file

helium_20W_pressure_sweep_legend_list = [] #initialize legend list
mixture_sweep_legend_list_20W = [] #initialize legend list

for i in range(0,len(helium_20W_pressure_sweep_files)): #plot resulting calibrations
    ppl.plot(frequency, helium_20W_pressure_sweep_calibrated[i])
    helium_20W_pressure_sweep_legend_list.append(helium_20W_pressure_sweep_files[i].split(dirname+'Feb_24_grounded_helium_20W_')[1].split('.csv')[0]) #creates legend from file names
ppl.plot(frequency, vacuum_calibrated,) #plot vacuum data 
helium_20W_pressure_sweep_legend_list.append('vacuum') #add last item to legend list
plt.legend(helium_20W_pressure_sweep_legend_list, loc='best', prop={'size':5})
plt.xlabel('Frequency [Hz]')
plt.ylabel('Reflected Signal [a.u.]')
plt.show()

for i in range(0,len(mixture_sweep_20W_files)): #plot resulting calibrations
    ppl.plot(frequency, mixture_sweep_calibrated_20W[i])
    mixture_sweep_legend_list_20W.append(mixture_sweep_20W_files[i].split(dirname+'Feb_24_grounded_argon_helium_20W_')[1].split('.csv')[0]) #creates legend from file names
ppl.plot(frequency, vacuum_calibrated,) #plot frequency 
mixture_sweep_legend_list_20W.append('vacuum') #add last item to legend list
plt.legend(mixture_sweep_legend_list_20W, loc='best', prop={'size':5})
plt.xlabel('Frequency [Hz]')
plt.ylabel('Reflected Signal [a.u.]')
plt.show()
    
#---------------------------------------------------------------------------    
#Save calibrated data as a csv file, which will be read in by a fityk script
#---------------------------------------------------------------------------

k = np.hstack((np.c_[frequency.astype(float)], np.c_[vacuum_calibrated.astype(float)]))
for i in range(len(helium_20W_pressure_sweep)):
    k = np.hstack((k, np.c_[helium_20W_pressure_sweep_calibrated[i].astype(float)]))
np.savetxt('Fitykloadfile_grounded_helium_20W.csv', k, delimiter=",")

k = np.hstack((np.c_[frequency.astype(float)], np.c_[vacuum_calibrated.astype(float)]))
for i in range(len(mixture_sweep_20W)):
    k = np.hstack((k, np.c_[mixture_sweep_calibrated_20W[i].astype(float)]))
np.savetxt('Fitykloadfile_grounded_mixture_20W.csv', k, delimiter=",")

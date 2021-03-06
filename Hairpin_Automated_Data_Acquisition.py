# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:32:04 2018

@author: David Peterson
"""

#-------------------------------------------------------------------------------
#   Save data from a DPO5104 Tektronix oscilloscope using GPIB or USB
#   Determine vacuum resonant frequency, steady state resonant frequency, and pulsed resonant frequency (time dependent)   
#   Determine steady state and time dependent electron densit for a hairpin resonator probe
#   
#
# python        3.6         (http://www.python.org/)
# pyvisa        1.4         (http://pyvisa.sourceforge.net/)
# numpy         1.6.2       (http://numpy.scipy.org/)
# MatPlotLib    1.0.1       (http://matplotlib.sourceforge.net/)
#-------------------------------------------------------------------------------

import visa
import numpy as np
from struct import unpack
import pylab
import pandas as pd
import sys, os
sys.path.append('C:\\Users\\lab-admin\\.spyder-py3')
from synth import synth
from scipy.optimize import curve_fit
from scipy import optimize

####
## Determine vacuum resonant frequency
####

freqstart = 2030   #starting frequency in MHz
freqstop = 2040 #stopping frequency in MHz
freqstep = .25 #step in frequency in MHz
freqdelay = 0.5 #time delay for changing frequency in seconds
numsteps = int((freqstop-freqstart)/freqstep)
ss_plasma_freq_sweep_length = 250 #width of frequency sweep for steady state plasma
frequency = np.linspace(freqstart,freqstop,numsteps)

S1= synth("AH01SZGB")
S1.syn_cmd("clkint")
S1.syn_cmd("setpwr 10")
S1.syn_cmd("setfreqfrom" + " " + str(freqstart))
S1.syn_cmd("setfreqto" + " " + str(freqstop))
S1.syn_cmd("setfreqstep" + " " + str(freqstep))
S1.syn_cmd("setfreqdelay" + " " + str(freqdelay))
S1.syn_cmd("freqstep")

rm = visa.ResourceManager()
print(rm.list_resources())
ports = rm.list_resources()
#Replace this with code that determines if device is GPIB or USB connected and selects appropriate port name
scope = rm.open_resource('GPIB0::1::INSTR') #Currently assumes GPIB connection, will be made flexible to detect both GPIB and USB

scope.write('DATA:SOU CH4') #set channel to take data from
scope.write('DATA:START 0')
scope.write('DATA:STOP 1E5')
scope.write('DATA:WIDTH 1')
scope.write('DATA:ENC RPB')

ymult = float(scope.ask('WFMPRE:YMULT?'))
yzero = float(scope.ask('WFMPRE:YZERO?'))
yoff = float(scope.ask('WFMPRE:YOFF?'))
xincr = float(scope.ask('WFMPRE:XINCR?'))

#Take sample data set to get number of data points
scope.write('CURVE?')
data = scope.read_raw()
headerlen = 2 + int(data[1])
header = data[:headerlen]
ADC_wave = data[headerlen:-1]
ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))
Volts = (ADC_wave - yoff) * ymult  + yzero
Time = np.arange(0, xincr * len(Volts), xincr)

Raw_Reflected_Voltage = np.zeros((len(frequency), len(Volts)))

for i in range(0,len(frequency)):
    scope.write('CURVE?')
    data = scope.read_raw()
    headerlen = 2 + int(data[1])
    header = data[:headerlen]
    ADC_wave = data[headerlen:-1]

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))

    Volts = (ADC_wave - yoff) * ymult  + yzero
        
    Raw_Reflected_Voltage[i] = Volts
    S1.syn_cmd("freqstep")
    

S1.syn_cmd("stop")
S1.syn_cmd("shutdown")
S1.syn_close() #CLOSES RFCONTROL WINDOWS 

#Data Analysis
Reflected_Voltage_Averaged = np.average(Raw_Reflected_Voltage, axis=1)
Vacuum_Resonant_Frequency = frequency[Reflected_Voltage_Averaged.argmin()] #determines vacuum resonant frequency using minimum reflected voltage     
pylab.plot(frequency, Reflected_Voltage_Averaged)
pylab.show()

#fit w/ Lorentz curve
# initial parameter guesses
# [height, HWHM, center, background]
params = np.array([0.15, 1, Vacuum_Resonant_Frequency, 0.1], dtype=np.double)   #fit guess, hwhm in MHz
solp, ier = fit(params, frequency, Reflected_Voltage_Averaged)
Vacuum_Resonant_Frequency_Fit = solp[2]
Vacuum_fwhm = 2*abs(solp[1])
Vacuum_Fit = lorentz(frequency, *solp)

Q_Vacuum = Vacuum_Resonant_Frequency_Fit/Vacuum_fwhm

pylab.plot(frequency, Reflected_Voltage_Averaged, frequency, Vacuum_Fit)
pylab.show()

#Reference sweep (can also be a calibration sweep, but need to make into separate cell)
ref_sweep_offset = 10 #offset from vacuum resonance frequency in MHz
freqstart = ref_sweep_offset + int(Vacuum_Resonant_Frequency) #starting frequency in MHz
freqstop = freqstart+ss_plasma_freq_sweep_length #stopping frequency in MHz, may need to adjust this
#freqstep = 0.5 #step in frequency in MHz
freqdelay = 0.5 #time delay for changing frequency in seconds
numsteps = int((freqstop-freqstart)/freqstep)
frequency = np.linspace(freqstart,freqstop,numsteps)

S1= synth("AH01SZGB")
S1.syn_cmd("clkint")
S1.syn_cmd("setpwr -4")
S1.syn_cmd("setfreqfrom" + " " + str(freqstart))
S1.syn_cmd("setfreqto" + " " + str(freqstop))
S1.syn_cmd("setfreqstep" + " " + str(freqstep))
S1.syn_cmd("setfreqdelay" + " " + str(freqdelay))
S1.syn_cmd("freqstep")

scope.write('CURVE?')
data = scope.read_raw()
headerlen = 2 + int(data[1])
header = data[:headerlen]
ADC_wave = data[headerlen:-1]
ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))
Volts = (ADC_wave - yoff) * ymult  + yzero
Time = np.arange(0, xincr * len(Volts), xincr)

Ref_Reflected_Voltage = np.zeros((len(frequency), len(Volts)))

for i in range(0,len(frequency)):
    scope.write('CURVE?')
    data = scope.read_raw()
    headerlen = 2 + int(data[1])
    header = data[:headerlen]
    ADC_wave = data[headerlen:-1]

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))

    Volts = (ADC_wave - yoff) * ymult  + yzero
        
    Ref_Reflected_Voltage[i] = Volts
    S1.syn_cmd("freqstep")
    
S1.syn_cmd("stop")
S1.syn_cmd("shutdown")
S1.syn_close()

Ref_Reflected_Voltage_Averaged = np.average(Ref_Reflected_Voltage, axis=1) #Reference reflected voltage to be used in steady state electron density determinatoin

#%%

####
## Determine resonant frequency and electron density in a steady state plasma
####
S1.syn_close()
freqstart = int(Vacuum_Resonant_Frequency) #starting frequency in MHz
freqstop = freqstart+ss_plasma_freq_sweep_length #stopping frequency in MHz, may need to adjust this
#freqstep = 0.5 #step in frequency in MHz
freqdelay = 0.01 #time delay for changing frequency in seconds
numsteps = int((freqstop-freqstart)/freqstep)
frequency = np.linspace(freqstart,freqstop,numsteps)

S1= synth("AH01SZGB")
S1.syn_cmd("clkint")
S1.syn_cmd("setpwr -4")
S1.syn_cmd("setfreqfrom" + " " + str(freqstart))
S1.syn_cmd("setfreqto" + " " + str(freqstop))
S1.syn_cmd("setfreqstep" + " " + str(freqstep))
S1.syn_cmd("setfreqdelay" + " " + str(freqdelay))
S1.syn_cmd("freqstep")

rm = visa.ResourceManager()
print(rm.list_resources())
scope = rm.open_resource('GPIB0::1::INSTR')

scope.write('DATA:SOU CH4') #set channel to take data from
scope.write('DATA:START 0')
scope.write('DATA:STOP 1E5')
scope.write('DATA:WIDTH 1')
scope.write('DATA:ENC RPB')

ymult = float(scope.ask('WFMPRE:YMULT?'))
yzero = float(scope.ask('WFMPRE:YZERO?'))
yoff = float(scope.ask('WFMPRE:YOFF?'))
xincr = float(scope.ask('WFMPRE:XINCR?'))

#Take sample data set to get number of data points
scope.write('CURVE?')
data = scope.read_raw()
headerlen = 2 + int(data[1])
header = data[:headerlen]
ADC_wave = data[headerlen:-1]
ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))
Volts = (ADC_wave - yoff) * ymult  + yzero
Time = np.arange(0, xincr * len(Volts), xincr)

Raw_Reflected_Voltage = np.zeros((len(frequency), len(Volts)))

for i in range(0,len(frequency)):
    scope.write('CURVE?')
    data = scope.read_raw()
    headerlen = 2 + int(data[1])
    header = data[:headerlen]
    ADC_wave = data[headerlen:-1]

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))

    Volts = (ADC_wave - yoff) * ymult  + yzero
        
    Raw_Reflected_Voltage[i] = Volts
    while '1' in scope.ask("BUSY?"):
        time.sleep(0.01)
    S1.syn_cmd("freqstep")
    
S1.syn_cmd("stop")
S1.syn_cmd("shutdown")
S1.syn_close()

SS_Reflected_Voltage_Averaged = np.average(Raw_Reflected_Voltage, axis=1)
Corrected_Reflected_Voltage = SS_Reflected_Voltage_Averaged - Ref_Reflected_Voltage_Averaged #subtracts off vacuum transmission line curve
Steady_State_Resonant_Frequency = frequency[Corrected_Reflected_Voltage.argmin()] #determines vacuum resonant frequency using minimum reflected voltage    
Uncorrected_Electron_Density = 1E10*(np.square(Steady_State_Resonant_Frequency/1000) - np.square(Vacuum_Resonant_Frequency/1000))/0.81 #calculates electron density in cm^-3
pylab.plot(frequency, SS_Reflected_Voltage_Averaged)
pylab.show()

#%%


####
## Determine time resolved resonant frequency and electron density in a pulsed plasmsa
####
freqstart = int(Vacuum_Resonant_Frequency) #starting frequency in MHz
freqstop = freqstart+ss_plasma_freq_sweep_length #stopping frequency in MHz, may need to adjust this
#freqstep = 0.5 #step in frequency in MHz
freqdelay = 0.5 #time delay for changing frequency in seconds
numsteps = int((freqstop-freqstart)/freqstep)
frequency = np.linspace(freqstart,freqstop,numsteps)

S1= synth("AH01SZGB")
S1.syn_cmd("clkint")
S1.syn_cmd("setpwr -4")
S1.syn_cmd("setfreqfrom" + " " + str(freqstart))
S1.syn_cmd("setfreqto" + " " + str(freqstop))
S1.syn_cmd("setfreqstep" + " " + str(freqstep))
S1.syn_cmd("setfreqdelay" + " " + str(freqdelay))
S1.syn_cmd("freqstep")

rm = visa.ResourceManager()
print(rm.list_resources())
scope = rm.open_resource('GPIB0::1::INSTR')

scope.write('DATA:SOU CH4') #set channel to take data from
scope.write('DATA:START 0')
scope.write('DATA:STOP 1E5')
scope.write('DATA:WIDTH 1')
scope.write('DATA:ENC RPB')

ymult = float(scope.ask('WFMPRE:YMULT?'))
yzero = float(scope.ask('WFMPRE:YZERO?'))
yoff = float(scope.ask('WFMPRE:YOFF?'))
xincr = float(scope.ask('WFMPRE:XINCR?'))

#Take sample data set to get number of data points
#scope.write("ACQ:STOPA SEQ")
#scope.write("ACQ:STATE ON")


scope.write('CURVE?')
data = scope.read_raw()
headerlen = 2 + int(data[1])
header = data[:headerlen]
ADC_wave = data[headerlen:-1]
ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))
Volts = (ADC_wave - yoff) * ymult  + yzero
Time = np.arange(0, xincr * len(Volts), xincr)
while '1' in scope.ask("BUSY?"):
        time.sleep(0.01)


Raw_Reflected_Voltage_Pulsed = np.zeros((len(frequency), len(Volts)))
Corrected_Reflected_Voltage = np.zeros((len(frequency), len(Volts)))

for i in range(0,len(frequency)):
    #scope.write("ACQ:STOPA SEQ")
    #scope.write("ACQ:STATE ON")
    
    scope.write('CURVE?')
    data = scope.read_raw()
    headerlen = 2 + int(data[1])
    header = data[:headerlen]
    ADC_wave = data[headerlen:-1]

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))

    Volts = (ADC_wave - yoff) * ymult  + yzero
        
    Raw_Reflected_Voltage_Pulsed[i] = Volts
    Corrected_Reflected_Voltage[i] = Raw_Reflected_Voltage_Pulsed[i]-Ref_Reflected_Voltage[i] #corrects for vacuum transmission line
    
    while '1' in scope.ask("BUSY?"):
        time.sleep(0.01)
    S1.syn_cmd("freqstep")
    
S1.syn_cmd("stop")
S1.syn_cmd("shutdown")
S1.syn_close()
    
#Data Analysis
Time_Resolved_Resonant_Frequency = frequency[Corrected_Reflected_Voltage.argmin(axis=0)] #determines vacuum resonant frequency using minimum reflected voltage    
Uncorrected_Electron_Density = 1E10*(np.square(Time_Resolved_Resonant_Frequency/1000) - np.square(Vacuum_Resonant_Frequency/1000))/0.81

############
#Fits data and finds Q in Plasma
# Lorentzian fitting function
def lorentz(x, *p):
    I, gamma, x0, bg = p
    return I * gamma**2 / ((x - x0)**2 + gamma**2) + bg

def fit(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)

# initial parameter guesses
# [height, HWHM, center, background]
# Get the fitting parameters for the best lorentzian
Plasma_Resonant_Frequency_Fit = np.zeros(len(Time))
plasma_fwhm = np.zeros(len(Time))
solution = np.zeros((len(frequency), len(Time)))

for i in range(0,len(Time)):
    params = np.array([0.15, 1, Time_Resolved_Resonant_Frequency[i], 0.0], dtype=np.double)   #fit guess, hwhm in MHz
    solp, ier = fit(params, frequency, Corrected_Reflected_Voltage[:,i])
    Plasma_Resonant_Frequency_Fit[i] = solp[2]
    plasma_fwhm[i] = 2*abs(solp[1])
    solution[:,i] = lorentz(frequency, *solp)

# error stuff
# coefficient of determination
def calc_r2(y, f):
    avg_y = y.mean()
    sstot = ((y - avg_y)**2).sum()
    ssres = ((y - f)**2).sum()
    return 1 - ssres/sstot

# calculate the errors
#r2 = calc_r2(frequency, lorentz(frequency, *solp)) #r_squared for fit
Uncorrected_Electron_Density_Fit = 1E10*(np.square(Plasma_Resonant_Frequency_Fit/1000) - np.square(Vacuum_Resonant_Frequency/1000))/0.81

Q_Measured = Plasma_Resonant_Frequency_Fit/plasma_fwhm
Error_in_Fit_Plasma_Resonant_Frequency = 100*(Time_Resolved_Resonant_Frequency-Plasma_Resonant_Frequency_Fit)/Time_Resolved_Resonant_Frequency

Q_Plasma = (1/((1/Q_Measured) - (1/Q_Vacuum)))

#############

Time_micro = 1E6*Time
pylab.plot(Time_micro, Uncorrected_Electron_Density)
#pylab.axis([0, 200, 0, 1.3E10])
pylab.xlabel('Time [${\mu}s$ ]')
pylab.ylabel('Electron Density [cm$^{-3}$]')
#xticks(np.linspace(Time_micro[0],max(Time_micro),5))
pylab.show()

Time_micro = 1E6*Time
pylab.plot(Time_micro, Uncorrected_Electron_Density_Fit)
#pylab.axis([0, 200, 0, 1.3E10])
pylab.xlabel('Time [${\mu}s$ ]')
pylab.ylabel('Electron Density using Fit [cm$^{-3}$]')
#xticks(np.linspace(Time_micro[0],max(Time_micro),5))
pylab.show()

data_out = np.transpose([Time_micro, Uncorrected_Electron_Density])
np.savetxt("C:\\Users\\lab-admin\\Documents\\Hairpin Automated DAQ Data\\Oxygen_4-10-2018\\Ar Si -75V.csv", data_out, delimiter=",")
#%%

####
## Forward and Reflected Power and Trigger Voltage Data Acquisition
####


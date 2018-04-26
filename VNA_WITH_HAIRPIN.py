# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:58:35 2018

@author: David Peterson
"""
#####
#RVNA WITH HAIRPIN RESONATOR PROBE
#####

#This modified script allows for measurement of vacuum resonant frequency and quality factor for the hairpin resonator probe
#The second section calculates resonant frequency and quality factor for probe in plasma, as well as electron density and collision frequency

#########################################################
##                                                     ##
##  Program Name: vna.py                               ##
##                                                     ##
##  Python programming example for CMT VNA             ##
##                                                     ##
##  Version: January 2016                              ##
##  Author: CMT support                                ##
##                                                     ##
##  Support:  support@coppermountaintech.com           ##
##                                                     ##
#########################################################
##   Program Description: This example demonstrates    ##
##   how to connect to the instrument, configure a     ##
##   measurement, retrieve a result, and store it to   ##
##   file.  Measurements are collected in a loop which ##
##   runs num_iter times.                              ##
##                                                     ##
## 06/10/2015 Updated to use new python function input() instead of raw_input().
## 06/22/2015 Change the syntax to interactive UI. Update format and parameter issue.
## 09/28/2015 Update to increase the readability of the program
## 01/11/2016 Modified to use new COM server names (RVNA, TRVNA, S2VNA, and S4VNA)
#########################################################

# Allows communication via COM interface
try:
	import win32com.client
except:
	print("You will first need to import the pywin32 extension")
	print("to get COM interface support.")
	print("Try http://sourceforge.net/projects/pywin32/files/ )?")
	input("\nPress Enter to Exit Program\n")
	exit()

# Allows time.sleep() command
import time
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import LorentzianModel
from lmfit import Model, minimize, Parameters, report_fit, Minimizer
from scipy.optimize import curve_fit

###########################
# Prompt for user's input
# Input parameters
#
instrlist= ['R54 or R140',
            'TR1300, TR5048, or TR7530',
            'S5048, S7530, Planar804, or Planar304',
            'S8081']
familylist=['RVNA',
            'TRVNA',
            'S2VNA',
            'S4VNA']
#print('\n','0 - ',instrlist[0],'\n','1 - ',instrlist[1],'\n','2 - ',instrlist[2],'\n','3 - ',instrlist[3],'\n')

#choose the instrument
#instrument=familylist[int(input('Please select your instrument(only enter the first number):'))]
instrument=familylist[0] #This auto selects the VNA type, which is R54

#choose frequency type, 0 for Start/Stop Frequency, 1 for Center/Span Frequency
#use_center_and_span = int(input('\nPlease enter whether 0 - Start/Stop Frequency \t 1-Center/Span Frequency:'))
use_center_and_span = 0 #This auto selects start/stop frequency range instead of center/span

#power level
#power_level_dbm = float(input('\nPlease enter power level(dbm):'))
#power_level_dbm = input('\nPlease enter power level(dbm):')

power_level_dbm = "HIGH" #keep as high for better SNR, only 2 options with the R54, "HIGH" or "LOW"

#fstart=400e6 or center, as per above, in Hz
f1_hz = float(input('\nPlease enter start/center frequency (Hz):'))

#fstop=600e6 or span, as per above, in Hz
f2_hz = float(input('\nPlease enter stop/span frequency (Hz):'))

#number of measurement points
#num_points = int(input('\nPlease enter number of measurement points:'))
num_points = 10000 #This can be changed but is not left as an input because it will likely be infrequently changed
#parameter = input('\nPlease enter the parameter (e.g. S11, S21, S12, S22, etc):')
parameter = "S11" #always selected for S11 for the R54
# "S21", "S11", "S12", etc. R54/140 must use
# "S11"; TR devices must use "S11" or "S21";
#  Ports 3 and 4 available for S8081 only

#"mlog" or "phase" or"smith chart"
#format = input('\nPlease enter the format (e.g. mlog, phase, smith):')
#format = "mlog" #"mlin" for linear
format = "mlin"

#measurement interval
#time_per_iter_sec = float(input('\nPlease enter measurement interval(second):'))
time_per_iter_sec = 1 #this can be changed, not sure how this influences measurement yet, may be capable of time resolution but it unlikely, may be useful for automated experiments

#number of times to loop
#num_iter = int(input('\nPlease enter number of times to loop:'))
num_iter = 2 #not sure how this impacts measurements (do they average?)

#number of function iterations to store
#num_iter_to_store = int(input('\nPlease enter number of function iterations to store:'))
num_iter_to_store = 2 #not sure how this works either
###########################
#
#  Example code
#

#Instantiate COM client
try:
	app = win32com.client.Dispatch(instrument + ".application")
except:
	print("Error establishing COM server connection to " + instrument + ".")
	print("Check that the VNA application COM server was registered")
	print("at the time of software installation.")
	print("This is described in the VNA programming manual.")
	input("\nPress Enter to Exit Program\n")
	exit()

#Wait up to 20 seconds for instrument to be ready
if app.Ready == 0:
    print("Instrument not ready! Waiting...")
    for k in range (1, 21):
        time.sleep(1)
        if app.Ready != 0:
            break
        print("%d" % k)

# If the software is still not ready, cancel the program
if app.Ready == 0:
	print("Error, timeout waiting for instrument to be ready.")
	print("Check that VNA is powered on and connected to PC.")
	print("The status Ready should appear in the lower right")
	print("corner of the VNA application window.")
	input("\nPress Enter to Exit Program\n")
	exit()
else:
    print("Instrument ready! Continuing...")

#Get and echo the instrument name, serial number, etc.
#
#  [This is a simple example of getting an ActiveX property in Python]
#
print(app.name)

# Sets the instrument to a preset state
#
#  [This is an example of executing an ActiveX "method" in Python]
#
app.scpi.system.preset()

#Configure the stimulus
if use_center_and_span == 1:
#
#  [This is a simple example of setting an ActiveX property in Python. Note
#	that when indexed parameters are referenced, the Get prefix and SCPI
#	 capitalization must be used (e.g. GetSENSe(1) rather than simply sense(1) )]
	app.scpi.GetSENSe(1).frequency.center = f1_hz
	app.scpi.GetSENSe(1).frequency.span = f2_hz
else:
	app.scpi.GetSENSe(1).frequency.start = f1_hz
	app.scpi.GetSENSe(1).frequency.stop = f2_hz

app.scpi.GetSENSe(1).sweep.points = num_points

if instrument[0] == "R": #use this for RVNA equipment
	app.scpi.GetSOURce(1).power.level.state = power_level_dbm
 
#if instrument[0] != "R": #use this if using R60, expects float value
#	app.scpi.GetSOURce(1).power.level.immediate.amplitude = power_level_dbm

#Configure the measurement
app.scpi.GetCALCulate(1).GetPARameter(1).define = parameter
app.scpi.GetCALCulate(1).GetPARameter(1).select()
app.scpi.GetCALCulate(1).selected.format = format
app.scpi.trigger.sequence.source = "bus"

for iter in range(1,num_iter):

	#Execute the measurement
	app.scpi.trigger.sequence.single()

	app.scpi.GetCALCulate(1).GetPARameter(1).select()
	Y = app.scpi.GetCALCulate(1).selected.data.Fdata

	#Discard complex-valued points
	Y = Y[0::2]

	F = app.scpi.GetSENSe(1).frequency.data

	if iter <= num_iter_to_store:
		app.scpi.mmemory.store.image = str(iter) + ".png"
		app.scpi.mmemory.store.fdata = str(iter) + ".csv"

	time.sleep(time_per_iter_sec)

#plt.plot(F,Y) #plot measured spectrum

###########
###This section will do the analysis/curve fitting
##########

#Finds vacuum resonant frequency
Vacuum_Resonant_Frequency = F[np.argmin(Y)]

#Fits data and finds Q Vacuum
F_array = np.asarray(F)
Y_array = np.asarray(Y)

# Fitting the data
# Lorentzian fitting function
def lorentz(x, *p):
    I, gamma, x0, bg = p
    return I * gamma**2 / ((x - x0)**2 + gamma**2) + bg

# initial parameter guesses
# [height, HWHM, center, background]
params = np.array([0.8, 1E6, F[np.argmin(Y)], 0.1], dtype=np.double)   #fit guess

def fit(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)

# Get the fitting parameters for the best lorentzian
solp, ier = fit(params, F_array, Y_array)

# error stuff
# coefficient of determination
def calc_r2(y, f):
    avg_y = y.mean()
    sstot = ((y - avg_y)**2).sum()
    ssres = ((y - f)**2).sum()
    return 1 - ssres/sstot

# calculate the errors
r2 = calc_r2(Y_array, lorentz(F_array, *solp)) #r_squared for fit

solution = lorentz(F_array, *solp)
vacuum_fwhm = 2*abs(solp[1])
Q_Vacuum = Vacuum_Resonant_Frequency/vacuum_fwhm
Vacuum_Resonant_Frequency_Fit = solp[2]
Error_in_Fit_Vacuum_Resonant_Frequency = 100*abs(Vacuum_Resonant_Frequency-Vacuum_Resonant_Frequency_Fit)/Vacuum_Resonant_Frequency #percent difference

plt.plot(F,Y,F,solution)
plt.xlabel('Frquency [Hz]')
plt.ylabel('Reflected Signal [V]')
plt.legend(('Raw Data', 'Lorentz Fit'))

print("Vacuum Resonant Frequency = " + str(Vacuum_Resonant_Frequency/1E9) + " GHz")
print("Q_Vacuum = " + str(Q_Vacuum))


#%%
#This section does the same sweep/analysis but in plasma

#Since the in plasma resonant frequency isn't known, a large sweep above the vacuum resonance 
#is taken and the minimum of that is then used to remeasure the resonance over a narrower range, giving better results.
#The benefit of this is that it does not need to  be done manually.

f1_hz = Vacuum_Resonant_Frequency + 3E7

#fstop=600e6 or span, as per above, in Hz
f2_hz = Vacuum_Resonant_Frequency + 3E8

#Instantiate COM client
try:
	app = win32com.client.Dispatch(instrument + ".application")
except:
	print("Error establishing COM server connection to " + instrument + ".")
	print("Check that the VNA application COM server was registered")
	print("at the time of software installation.")
	print("This is described in the VNA programming manual.")
	input("\nPress Enter to Exit Program\n")
	exit()

#Wait up to 20 seconds for instrument to be ready
if app.Ready == 0:
    print("Instrument not ready! Waiting...")
    for k in range (1, 21):
        time.sleep(1)
        if app.Ready != 0:
            break
        print("%d" % k)

# If the software is still not ready, cancel the program
if app.Ready == 0:
	print("Error, timeout waiting for instrument to be ready.")
	print("Check that VNA is powered on and connected to PC.")
	print("The status Ready should appear in the lower right")
	print("corner of the VNA application window.")
	input("\nPress Enter to Exit Program\n")
	exit()
else:
    print("Instrument ready! Continuing...")

#Get and echo the instrument name, serial number, etc.
#
#  [This is a simple example of getting an ActiveX property in Python]
#
print(app.name)

# Sets the instrument to a preset state
#
#  [This is an example of executing an ActiveX "method" in Python]
#
app.scpi.system.preset()

#Configure the stimulus
if use_center_and_span == 1:
#
#  [This is a simple example of setting an ActiveX property in Python. Note
#	that when indexed parameters are referenced, the Get prefix and SCPI
#	 capitalization must be used (e.g. GetSENSe(1) rather than simply sense(1) )]
	app.scpi.GetSENSe(1).frequency.center = f1_hz
	app.scpi.GetSENSe(1).frequency.span = f2_hz
else:
	app.scpi.GetSENSe(1).frequency.start = f1_hz
	app.scpi.GetSENSe(1).frequency.stop = f2_hz

app.scpi.GetSENSe(1).sweep.points = num_points

if instrument[0] == "R": #use this for RVNA equipment
	app.scpi.GetSOURce(1).power.level.state = power_level_dbm
 
#if instrument[0] != "R": #use this if using R60, expects float value
#	app.scpi.GetSOURce(1).power.level.immediate.amplitude = power_level_dbm

#Configure the measurement
app.scpi.GetCALCulate(1).GetPARameter(1).define = parameter
app.scpi.GetCALCulate(1).GetPARameter(1).select()
app.scpi.GetCALCulate(1).selected.format = format
app.scpi.trigger.sequence.source = "bus"

for iter in range(1,num_iter):

	#Execute the measurement
	app.scpi.trigger.sequence.single()

	app.scpi.GetCALCulate(1).GetPARameter(1).select()
	Y = app.scpi.GetCALCulate(1).selected.data.Fdata

	#Discard complex-valued points
	Y = Y[0::2]

	F = app.scpi.GetSENSe(1).frequency.data

	if iter <= num_iter_to_store:
		app.scpi.mmemory.store.image = str(iter) + ".png"
		app.scpi.mmemory.store.fdata = str(iter) + ".csv"

	time.sleep(time_per_iter_sec)

###########
###This section will do the analysis/curve fitting
##########

#Finds vacuum resonant frequency
Plasma_Resonant_Frequency = F[np.argmin(Y)]

#Fits data and finds Q in Plasma
F_array = np.asarray(F)
Y_array = np.asarray(Y)

# Fitting the data
# Lorentzian fitting function
def lorentz(x, *p):
    I, gamma, x0, bg = p
    return I * gamma**2 / ((x - x0)**2 + gamma**2) + bg

# initial parameter guesses
# [height, HWHM, center, background]
params = np.array([0.8, 1E6, F[np.argmin(Y)], 0.1], dtype=np.double)   #fit guess

def fit(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)

# Get the fitting parameters for the best lorentzian
solp, ier = fit(params, F_array, Y_array)

# error stuff
# coefficient of determination
def calc_r2(y, f):
    avg_y = y.mean()
    sstot = ((y - avg_y)**2).sum()
    ssres = ((y - f)**2).sum()
    return 1 - ssres/sstot

# calculate the errors
r2 = calc_r2(Y_array, lorentz(F_array, *solp)) #r_squared for fit

solution = lorentz(F_array, *solp)
plasma_fwhm = 2*abs(solp[1])
Plasma_Resonant_Frequency_Fit = solp[2]
Q_Measured = Plasma_Resonant_Frequency_Fit/plasma_fwhm
Error_in_Fit_Plasma_Resonant_Frequency = 100*(Plasma_Resonant_Frequency-Plasma_Resonant_Frequency_Fit)/Plasma_Resonant_Frequency
plt.plot(F,Y,F,solution)
plt.xlabel('Frquency [Hz]')
plt.ylabel('Reflected Signal [V]')
plt.legend(('Raw Data', 'Lorentz Fit'))

print("Plasma Resonant Frequency = " + str(Plasma_Resonant_Frequency/1E9) + " GHz")
print("Q_Measured = " + str(Q_Measured))

#Calculate electron density
Uncorrected_Electron_Density = 1E10*(np.square(Plasma_Resonant_Frequency/1E9) - np.square(Vacuum_Resonant_Frequency/1E9))/0.8062 #calculates electron density in cm^-3

#Calculate sheath correction and collision frequency
#This assumes a sheath thickness of ~4 debye lengths (will change but is a measureable quantity)
#Note that this uses the resonant frequency from the fit (difference between the fit and raw data is also calculated)

Q_Plasma = (1/((1/Q_Measured) - (1/Q_Vacuum))) #this comes directly from transmission line theory

N_collisionfreq = 1000 #mesh size for collision frequency constant sweep
T_e_guess = 3.2 # electron temperature guess [eV] 

a = 0.00022 # hairpin wire radius [m]
w = 0.00185 # width between hairpin tines [m]
Avagadro = 6.022E23 # [atoms/mol]
neutral_gas_temp = 298.15 #neutral gas temperature used to calculate neutral gas density [K] (can use N2 rotational temps for this)
R_gas = 62.36367 #gas constant [L*Torr/K*mol]
convergence_criteria = 1E-5 #used in sheath correction loop
c = 2.9979E8 #speed of light [m/s]
epsilon_naught = 8.854E-12 #permittivity of free space [F/m]
Pressure = 2 #[Torr] THIS WILL NEED TO BE CHANGED

n_g = (Pressure/(R_gas*neutral_gas_temp))*Avagadro*(1/1000) #obtain neutral gas density, 1/1000 for converting from L to cm^-3 (THIS IS CURRENTLY NOT USED)

collfreq = Pressure*np.linspace(1.0E8, 7E9, N_collisionfreq) #array of pressure normalized collision frequencies [Hz/Torr] * pressure[Torr] to create collision frequency array [Hz] to sweep over to find solution 

for j in range(0,N_collisionfreq):
    
    collision_correction = 1/(1 + (collfreq/(2*np.pi*Vacuum_Resonant_Frequency_Fit))**2) #uses vacuum frequency here
    Density_pressure_corrected = Density/collision_correction
        
    convergence = 1
    Density_corrected = Density_pressure_corrected
        
    while convergence > convergence_criteria:
            
        lambda_ds = 7.43*np.sqrt(T_e_guess/Density_corrected) #debye length at sheath edge in [m]
        sheath_width = 4.0*lambda_ds #this sheath assumes a 4 Debye length sheath
        b = sheath_width + a 
                
        sheath_correction = 1 - ((Vacuum_Resonant_Frequency_Fit**2)/(Plasma_Resonant_Frequency_Fit**2))*((np.log((w-a)/(w-b)) + np.log(b/a))/np.log((w-a)/a)) #uses more copmlicated sheath definition, assumes sheath is completely devoid of electrons
            
        Density_updated = Density_pressure_corrected/sheath_correction
        convergence = abs(Density_updated - Density_corrected)/Density_updated #update convergence criteria
            
        if convergence < convergence_criteria:
            Final_Density[j] = Density_updated
            Final_sheath_width = sheath_width
            Final_b = b
                
        Density_corrected = Density_updated #updates the last iterations density

    def f(x):
        return ((((2*np.pi*Plasma_Resonant_Frequency_Fit)**2 + x**2)/((2*np.pi*9000*np.sqrt(Final_Density[j]))**2) -1)*(2*np.pi*Plasma_Resonant_Frequency_Fit/x) - Q_Plasma) #Solves for nu_effective from Q_Plasma
        
    measured_collision_freq[j] = optimize.newton(f, 1E9, tol=6E-6, maxiter=750) #finds zero crossing of function f
        
    difference[j] = 100*abs(measured_collision_freq[j] - collfreq[j])/collfreq[j]#find minimum of percent-difference of input collision freq to output
        
    true_collision_freq = collfreq[difference.argmin()] #selects the pressure normalized collision frequency that corresponds to minimum of %different

pressure_normalized_measured_collision_freq = true_collision_freq/Pressure #pressure normalized collision freq [Hz/Torr]
Pressure_and_Sheath_Corrected_Electron_Density = Final_Density[difference.argmin()] #corresponding measured electron density [cm^-3] using measured collision frequency

print("Pressure and Sheath Corrected Electron Density = " + str(Pressure_and_Sheath_Corrected_Electron_Density) + '[cm^-3]')
print("Effective Collision Frequency = " + str(true_collision_freq) + '[Hz]')

########
##This section constrains the possible EEDFs being measured using equation (8) in Peterson et al 2017 PSST
########

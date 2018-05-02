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
import pandas as pd
import pylab
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy import optimize

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
plt.show()

print("Vacuum Resonant Frequency = " + str(Vacuum_Resonant_Frequency/1E9) + " GHz")
print("Q_Vacuum = " + str(Q_Vacuum))


#Remove resonance to fit transmission line curve
#Use 4*FWHM for total width to remove resonance, centered around resonance
#set range of data removal
Y_array_Modified_For_Interpolation = Y_array
F_array_Modified_For_Interpolation = F_array

Y_array_Modified_For_Interpolation = np.delete(Y_array, range(np.abs(F_array-(Vacuum_Resonant_Frequency_Fit-4*vacuum_fwhm)).argmin(), np.abs(F_array-(Vacuum_Resonant_Frequency_Fit+4*vacuum_fwhm)).argmin()))
F_array_Modified_For_Interpolation = np.delete(F_array, range(np.abs(F_array-(Vacuum_Resonant_Frequency_Fit-4*vacuum_fwhm)).argmin(), np.abs(F_array-(Vacuum_Resonant_Frequency_Fit+4*vacuum_fwhm)).argmin()))

#for i in range(np.abs(F_array-(Vacuum_Resonant_Frequency_Fit-4*vacuum_fwhm)).argmin(), np.abs(F_array-(Vacuum_Resonant_Frequency_Fit+4*vacuum_fwhm)).argmin()):
#
#    Y_array_Modified_For_Interpolation[i] = 0
    
#Now spline to fit data    

Y_interpolated = UnivariateSpline(F_array_Modified_For_Interpolation, Y_array_Modified_For_Interpolation)
Y_interpolated.set_smoothing_factor(0.06) #THIS MAY HAVE TO CHANGE

pylab.plot(F_array, Y_array, F_array, Y_interpolated(F_array))
plt.xlabel('Frquency [Hz]')
plt.ylabel('Reflected Signal [V]')
plt.legend(('Raw Data', 'Spline with no resonance'))
plt.show()

#Calibrated Reflected Voltage
Y_Calibrated = Y_array - Y_interpolated(F_array)

#Fit calibrated curve

solp_cal, ier_cal = fit(params, F_array, Y_Calibrated)

solution_cal = lorentz(F_array, *solp_cal)
vacuum_fwhm_cal = 2*abs(solp_cal[1])
Vacuum_Resonant_Frequency_Fit_cal = solp_cal[2]
Q_Vacuum_cal = Vacuum_Resonant_Frequency_Fit_cal/vacuum_fwhm_cal

# calculate the errors
r2_cal = calc_r2(Y_Calibrated, lorentz(F_array, *solp_cal)) #r_squared for fit

plt.plot(F_array,Y_Calibrated,F,solution_cal)
plt.xlabel('Frquency [Hz]')
plt.ylabel('Reflected Signal [V]')
plt.legend(('Calibrated Raw Data', 'Lorentz Fit'))
plt.show()

#%%
#This section does the same sweep/analysis but in plasma

#Since the in plasma resonant frequency isn't known, a large sweep above the vacuum resonance 
#is taken and the minimum of that is then used to remeasure the resonance over a narrower range, giving better results.
#The benefit of this is that it does not need to  be done manually.

f1_hz = float(input('\nPlease enter start/center frequency (Hz):'))
f2_hz = float(input('\nPlease enter stop/span frequency (Hz):'))

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

Pressure = float(input('\nPlease enter the pressure[Torr]:')) #[Torr]

n_g = (Pressure/(R_gas*neutral_gas_temp))*Avagadro*(1/1000) #obtain neutral gas density, 1/1000 for converting from L to cm^-3

collfreq = Pressure*np.linspace(1.0E8, 7E9, N_collisionfreq) #array of pressure normalized collision frequencies [Hz/Torr] * pressure[Torr] to create collision frequency array [Hz] to sweep over to find solution 

Final_Density = np.zeros(N_collisionfreq)
measured_collision_freq = np.zeros(N_collisionfreq)
difference = np.zeros(N_collisionfreq)

for j in range(0,N_collisionfreq):
    
    collision_correction = 1/(1 + (collfreq[j]/(2*np.pi*Vacuum_Resonant_Frequency_Fit))**2) #uses vacuum frequency here
    Density_pressure_corrected = Uncorrected_Electron_Density/collision_correction
        
    convergence = 1
    Density_corrected = Density_pressure_corrected
        
    while convergence > convergence_criteria:
            
        lambda_ds = 7.43*np.sqrt(T_e_guess/Density_corrected) #debye length at sheath edge in [m]
        sheath_width = 1.5*lambda_ds #this sheath assumes a 2 Debye length sheath (NEEDS TO BE MEASURED)
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
Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed = Pressure_and_Sheath_Corrected_Electron_Density*1E6 #convert to [m^-3] for later use

print("Pressure and Sheath Corrected Electron Density = " + str(Pressure_and_Sheath_Corrected_Electron_Density/1E10) + 'E10 [cm^-3]')
print("Effective Collision Frequency = " + str(true_collision_freq) + '[Hz]')


########
##This section constrains the possible EEDFs being measured using equation (8) in Peterson et al 2017 PSST
########

#Set elastic collision cross section using input from user - cross sections from Biagi (Bolsig+)
Gas = input('\nPlease enter the gas type(Options: Ar, He, N2):')

if Gas == 'Ar':
    energy = '1.000000e-5 1.000000e-3 2.000000e-3 5.000000e-3 1.000000e-2 1.741900e-2 3.514200e-2 5.317400e-2 7.151900e-2 9.018400e-2 1.091700e-1 1.285000e-1 1.481500e-1 1.681500e-1 1.885000e-1 2.092100e-1 2.302700e-1 2.517000e-1 2.735000e-1 2.956900e-1 3.182600e-1 3.412200e-1 3.645800e-1 3.883500e-1 4.125400e-1 4.371400e-1 4.621800e-1 4.876500e-1 5.135600e-1 5.399300e-1 5.667500e-1 5.940400e-1 6.218100e-1 6.500600e-1 6.788000e-1 7.080500e-1 7.378000e-1 7.680700e-1 7.988700e-1 8.302100e-1 8.620900e-1 8.945200e-1 9.275200e-1 9.611000e-1 9.952600e-1 1.030000e+0 1.065400e+0 1.101400e+0 1.138000e+0 1.175200e+0 1.213100e+0 1.251600e+0 1.290900e+0 1.330800e+0 1.371400e+0 1.412700e+0 1.454700e+0 1.497500e+0 1.541000e+0 1.585200e+0 1.630300e+0 1.676100e+0 1.722700e+0 1.770100e+0 1.818400e+0 1.867500e+0 1.917400e+0 1.968200e+0 2.020000e+0 2.072600e+0 2.126100e+0 2.180500e+0 2.235900e+0 2.292300e+0 2.349700e+0 2.408000e+0 2.467400e+0 2.527800e+0 2.589200e+0 2.651700e+0 2.715400e+0 2.780100e+0 2.845900e+0 2.912900e+0 2.981100e+0 3.050400e+0 3.121000e+0 3.192800e+0 3.265800e+0 3.340100e+0 3.415700e+0 3.492600e+0 3.570900e+0 3.650500e+0 3.731500e+0 3.813900e+0 3.897800e+0 3.983100e+0 4.069900e+0 4.158200e+0 4.248100e+0 4.339500e+0 4.432500e+0 4.527100e+0 4.623400e+0 4.721400e+0 4.821000e+0 4.922400e+0 5.025600e+0 5.130600e+0 5.237300e+0 5.346000e+0 5.456500e+0 5.569000e+0 5.683400e+0 5.799900e+0 5.918300e+0 6.038800e+0 6.161400e+0 6.286200e+0 6.413100e+0 6.542200e+0 6.673600e+0 6.807300e+0 6.943300e+0 7.081600e+0 7.365700e+0 7.659600e+0 7.964000e+0 8.279000e+0 8.605100e+0 8.942600e+0 9.292000e+0 9.653700e+0 1.002800e+1 1.061400e+1 1.256800e+1 1.304400e+1 1.353800e+1 1.404900e+1 1.457800e+1 1.512500e+1 1.569200e+1 1.627800e+1 1.657900e+1 1.688500e+1 1.719700e+1 1.751400e+1 1.783600e+1 1.816500e+1 1.849800e+1 1.883800e+1 1.918400e+1 1.953500e+1 1.989300e+1 2.062700e+1 2.138700e+1 2.217400e+1 2.257800e+1 2.298800e+1 2.340600e+1 2.383100e+1 2.426400e+1 2.470400e+1 2.515200e+1 2.607100e+1 2.702200e+1 2.751000e+1 2.800700e+1 2.851200e+1 2.902600e+1 2.954900e+1 3.008100e+1 3.117400e+1 3.230400e+1 3.347500e+1 3.468600e+1 3.530800e+1 3.594000e+1 3.658400e+1 3.723800e+1 3.790500e+1 3.858200e+1 3.927200e+1 3.997300e+1 4.141300e+1 4.290400e+1 4.444600e+1 4.604400e+1 4.686300e+1 4.769700e+1 4.854500e+1 4.940800e+1 5.028600e+1 5.208800e+1 5.395400e+1 5.491100e+1 5.588500e+1 5.687600e+1 5.788400e+1 5.891000e+1 5.995400e+1 6.101500e+1 6.209600e+1 6.319500e+1 6.431300e+1 6.545100e+1 6.660800e+1 6.778600e+1 6.898400e+1 7.020300e+1 7.270600e+1 7.529600e+1 7.797700e+1 8.075200e+1 8.217600e+1 8.362500e+1 8.509900e+1 8.659900e+1 8.812500e+1 8.967800e+1 9.125700e+1 9.449900e+1 9.616300e+1 9.785500e+1 9.957700e+1 1.031100e+2 1.067700e+2 1.105600e+2 1.144800e+2 1.185400e+2 1.227400e+2 1.270900e+2 1.315900e+2 1.362500e+2 1.410700e+2 1.435400e+2 1.460600e+2 1.486200e+2 1.538800e+2 1.593200e+2 1.649600e+2 1.707900e+2 1.768300e+2 1.830800e+2 1.895500e+2 1.962400e+2 2.031700e+2 2.103500e+2 2.177800e+2 2.254600e+2 2.334200e+2 2.416600e+2 2.545600e+2 2.681500e+2 2.824700e+2 2.975400e+2 3.134100e+2 3.301300e+2 3.477400e+2 3.662800e+2 3.858100e+2 4.063800e+2 4.355200e+2 4.587300e+2 4.831700e+2 5.089200e+2 5.360300e+2 5.645900e+2 5.946600e+2 6.263400e+2 6.596900e+2 6.948200e+2 7.318200e+2 7.575800e+2 7.842400e+2 8.118300e+2 8.404000e+2 8.699600e+2 9.005700e+2 9.322500e+2 9.650500e+2'
    energy = energy.split()
    energy = np.asarray(energy)
    energy = energy.astype(np.float)

    sigma_c_energy = '6.298400e-20 6.298400e-20 5.835000e-20 4.977500e-20 4.113000e-20 3.290600e-20 2.144900e-20 1.474100e-20 1.032600e-20 7.273200e-21 5.117000e-21 3.589000e-21 2.519100e-21 1.793200e-21 1.330700e-21 1.073000e-21 9.762900e-22 1.007000e-21 1.139000e-21 1.351700e-21 1.628900e-21 1.957400e-21 2.326700e-21 2.728200e-21 3.155200e-21 3.602100e-21 4.064500e-21 4.539100e-21 5.023100e-21 5.514600e-21 6.012300e-21 6.515300e-21 7.023300e-21 7.536300e-21 8.054600e-21 8.579000e-21 9.110500e-21 9.650300e-21 1.020000e-20 1.076200e-20 1.133700e-20 1.192800e-20 1.253700e-20 1.316800e-20 1.382100e-20 1.431600e-20 1.479100e-20 1.527500e-20 1.576700e-20 1.626700e-20 1.677000e-20 1.727100e-20 1.778100e-20 1.830000e-20 1.882800e-20 1.936500e-20 1.991100e-20 2.046700e-20 2.107400e-20 2.169300e-20 2.232400e-20 2.296500e-20 2.358000e-20 2.416500e-20 2.476000e-20 2.536600e-20 2.598200e-20 2.660800e-20 2.729100e-20 2.805900e-20 2.884100e-20 2.963600e-20 3.044500e-20 3.126800e-20 3.210500e-20 3.295700e-20 3.382400e-20 3.472800e-20 3.567400e-20 3.663700e-20 3.761600e-20 3.861300e-20 3.962700e-20 4.065900e-20 4.170900e-20 4.275600e-20 4.381500e-20 4.489100e-20 4.598700e-20 4.710200e-20 4.823600e-20 4.938900e-20 5.056300e-20 5.175800e-20 5.297300e-20 5.420900e-20 5.546700e-20 5.674700e-20 5.832800e-20 6.000600e-20 6.171300e-20 6.345000e-20 6.521800e-20 6.701600e-20 6.884500e-20 7.070600e-20 7.260000e-20 7.452600e-20 7.651200e-20 7.861100e-20 8.074700e-20 8.292000e-20 8.513100e-20 8.738000e-20 8.966900e-20 9.199700e-20 9.436600e-20 9.673800e-20 9.906700e-20 1.014400e-19 1.038500e-19 1.063000e-19 1.088000e-19 1.113400e-19 1.139200e-19 1.163100e-19 1.208500e-19 1.255500e-19 1.304200e-19 1.354600e-19 1.406800e-19 1.460800e-19 1.513800e-19 1.568100e-19 1.621700e-19 1.656900e-19 1.620300e-19 1.586500e-19 1.547000e-19 1.505600e-19 1.458000e-19 1.408700e-19 1.357700e-19 1.305000e-19 1.277900e-19 1.250300e-19 1.222300e-19 1.193700e-19 1.164700e-19 1.137700e-19 1.112600e-19 1.087100e-19 1.061200e-19 1.034900e-19 1.008000e-19 9.717800e-20 9.375800e-20 9.021700e-20 8.840100e-20 8.655300e-20 8.467200e-20 8.275900e-20 8.081300e-20 7.883200e-20 7.704500e-20 7.428800e-20 7.143400e-20 6.996900e-20 6.848000e-20 6.696400e-20 6.542100e-20 6.385200e-20 6.235400e-20 6.038700e-20 5.835200e-20 5.624600e-20 5.406500e-20 5.294600e-20 5.180800e-20 5.064900e-20 4.947100e-20 4.827200e-20 4.705200e-20 4.581100e-20 4.454800e-20 4.315800e-20 4.174200e-20 4.027600e-20 3.875900e-20 3.798000e-20 3.718800e-20 3.638200e-20 3.556200e-20 3.480000e-20 3.353800e-20 3.223200e-20 3.156200e-20 3.088000e-20 3.018700e-20 2.948100e-20 2.876300e-20 2.803200e-20 2.744100e-20 2.684700e-20 2.624300e-20 2.562800e-20 2.500200e-20 2.436500e-20 2.371800e-20 2.305900e-20 2.244900e-20 2.182400e-20 2.117600e-20 2.050600e-20 1.977400e-20 1.934700e-20 1.891200e-20 1.847000e-20 1.802000e-20 1.756200e-20 1.709700e-20 1.674900e-20 1.610000e-20 1.576700e-20 1.542900e-20 1.508500e-20 1.465100e-20 1.424200e-20 1.381800e-20 1.337800e-20 1.292400e-20 1.245300e-20 1.201600e-20 1.162000e-20 1.121000e-20 1.078600e-20 1.056800e-20 1.034700e-20 1.012100e-20 9.829200e-21 9.589700e-21 9.341800e-21 9.085200e-21 8.819600e-21 8.544600e-21 8.260000e-21 7.965300e-21 7.711100e-21 7.510200e-21 7.302300e-21 7.087000e-21 6.864200e-21 6.633500e-21 6.317800e-21 6.073200e-21 5.815600e-21 5.544300e-21 5.312200e-21 5.078200e-21 4.831700e-21 4.637200e-21 4.441900e-21 4.252100e-21 4.033600e-21 3.859500e-21 3.676200e-21 3.501000e-21 3.351800e-21 3.194800e-21 3.029400e-21 2.894700e-21 2.761200e-21 2.620700e-21 2.472700e-21 2.369700e-21 2.263100e-21 2.164500e-21 2.078800e-21 1.990100e-21 1.898300e-21 1.803200e-21 1.704800e-21'
    sigma_c_energy = sigma_c_energy.split()
    sigma_c_energy = np.asarray(sigma_c_energy)
    sigma_c_energy = sigma_c_energy.astype(np.float)

if Gas == 'He':
    energy = '1.000000e-6 1.000000e-4 3.514000e-2 7.152000e-2 1.091700e-1 1.371370e+0 5.918310e+0 8.440610e+0 1.102264e+1 1.379108e+1 1.657924e+1 1.918366e+1 2.217395e+1 2.560725e+1 2.954921e+1 3.407519e+1 3.790451e+1 4.215191e+1 4.686301e+1 5.588529e+1 6.660830e+1 7.935261e+1 9.449926e+1 1.125011e+2 1.338963e+2 1.593245e+2 1.895461e+2 2.254644e+2 2.681535e+2 3.188895e+2 3.791894e+2 4.508559e+2 5.360318e+2 6.372635e+2 7.575776e+2 9.005711e+2'
    energy = energy.split()
    energy = np.asarray(energy)
    energy = energy.astype(np.float)

    sigma_c_energy = '4.903500e-20 4.903500e-20 5.501140e-20 5.747600e-20 5.896700e-20 6.938560e-20 6.025320e-20 5.209010e-20 4.444340e-20 3.734320e-20 3.184430e-20 2.766530e-20 2.383470e-20 1.998990e-20 1.667870e-20 1.385490e-20 1.190580e-20 1.024370e-20 8.806780e-21 6.702370e-21 5.073960e-21 3.808260e-21 2.878530e-21 2.204960e-21 1.622750e-21 1.224330e-21 9.142570e-22 6.756040e-22 4.949030e-22 3.701550e-22 2.748810e-22 2.023320e-22 1.477860e-22 1.101690e-22 8.165410e-23 5.983320e-23'
    sigma_c_energy = sigma_c_energy.split()
    sigma_c_energy = np.asarray(sigma_c_energy)
    sigma_c_energy = sigma_c_energy.astype(np.float)        

if Gas == 'N2':
    energy = '1.000000e-4 3.514217e-2 7.151931e-2 1.091748e-1 1.481536e-1 1.885022e-1 2.302688e-1 2.735031e-1 3.182567e-1 3.645831e-1 4.125375e-1 4.621772e-1 5.135612e-1 5.667511e-1 6.218101e-1 7.378008e-1 1.065380e+0 1.213095e+0 1.540973e+0 1.630268e+0 1.722701e+0 1.818383e+0 1.917427e+0 2.019952e+0 2.126079e+0 2.235937e+0 2.349654e+0 2.467369e+0 2.589219e+0 2.715352e+0 2.845918e+0 3.120975e+0 3.265795e+0 3.415704e+0 3.570882e+0 3.731513e+0 3.897788e+0 4.069907e+0 4.248075e+0 4.432503e+0 4.821032e+0 5.456542e+0 5.918310e+0 6.673615e+0 8.120108e+0 1.144515e+1 2.138721e+1 2.298833e+1 2.470396e+1 2.654229e+1 2.851209e+1 3.062278e+1 3.288442e+1 3.530781e+1 3.790451e+1 4.068694e+1 4.366836e+1 4.686301e+1 4.854502e+1 5.208844e+1 5.588529e+1 5.995369e+1 6.431306e+1 6.898420e+1 7.398942e+1 7.935261e+1 8.217638e+1 8.509938e+1 8.812509e+1 9.125714e+1 9.785531e+1 1.049254e+2 1.125011e+2 1.206186e+2 1.248925e+2 1.338963e+2 1.386368e+2 1.435440e+2 1.486236e+2 1.538817e+2 1.649587e+2 1.707908e+2 1.768279e+2 1.895461e+2 1.962423e+2 2.031738e+2 2.103489e+2 2.177762e+2 2.254644e+2 2.334229e+2 2.416610e+2 2.501886e+2 2.590160e+2 2.681535e+2 2.776121e+2 2.874032e+2 2.975383e+2 3.080295e+2 3.188895e+2 3.301311e+2 3.417678e+2 3.538134e+2 3.662823e+2 3.791894e+2 3.925501e+2 4.063803e+2 4.206965e+2 4.355158e+2 4.508559e+2 4.667351e+2 4.831724e+2 5.001872e+2 5.178000e+2 5.360318e+2 5.549043e+2 5.744399e+2 5.946621e+2 6.155950e+2 6.372635e+2 6.596934e+2 6.829116e+2 7.069458e+2 7.318245e+2 7.575776e+2 7.842356e+2 8.118305e+2 8.403951e+2 8.699636e+2 9.005711e+2 9.322543e+2 9.650509e+2'
    energy = energy.split()
    energy = np.asarray(energy)
    energy = energy.astype(np.float)
    
    sigma_c_energy = '1.145700e-20 3.671700e-20 5.147100e-20 6.179400e-20 7.060000e-20 7.721800e-20 8.263200e-20 8.735000e-20 9.127800e-20 9.452100e-20 9.775200e-20 1.007300e-19 1.038100e-19 1.070100e-19 1.097600e-19 1.130700e-19 1.096900e-19 1.062400e-19 1.123000e-19 1.238000e-19 1.414500e-19 1.651500e-19 1.893000e-19 1.828100e-19 2.016900e-19 2.377500e-19 1.930600e-19 2.508800e-19 1.835900e-19 2.302500e-19 1.782400e-19 1.590000e-19 1.384200e-19 1.215500e-19 1.111600e-19 1.047400e-19 1.000400e-19 9.788100e-20 9.503100e-20 9.208000e-20 8.971600e-20 8.717400e-20 8.532700e-20 8.365300e-20 8.200000e-20 8.372300e-20 8.161300e-20 8.001200e-20 7.829600e-20 7.614900e-20 7.378500e-20 7.137700e-20 6.911600e-20 6.676000e-20 6.473400e-20 6.260500e-20 6.045900e-20 5.819600e-20 5.701800e-20 5.412000e-20 5.070300e-20 4.704200e-20 4.398100e-20 4.071100e-20 3.800500e-20 3.532400e-20 3.391200e-20 3.245000e-20 3.093700e-20 2.956000e-20 2.725100e-20 2.541600e-20 2.375000e-20 2.196400e-20 2.102400e-20 1.939900e-20 1.854500e-20 1.766200e-20 1.674800e-20 1.606500e-20 1.482500e-20 1.417100e-20 1.356000e-20 1.258900e-20 1.207700e-20 1.158800e-20 1.113200e-20 1.065900e-20 1.017000e-20 9.664300e-21 9.140400e-21 8.602500e-21 8.251200e-21 7.887500e-21 7.511000e-21 7.121400e-21 6.718000e-21 6.403200e-21 6.110000e-21 5.806500e-21 5.492300e-21 5.196800e-21 4.957400e-21 4.709600e-21 4.453000e-21 4.219400e-21 4.016100e-21 3.805700e-21 3.590800e-21 3.419300e-21 3.241700e-21 3.058600e-21 2.924700e-21 2.786200e-21 2.642700e-21 2.494300e-21 2.340600e-21 2.220500e-21 2.110000e-21 1.995600e-21 1.877200e-21 1.765700e-21 1.678600e-21 1.588500e-21 1.495200e-21 1.410400e-21 1.339000e-21 1.265100e-21 1.188900e-21 1.128700e-21 1.066400e-21'
    sigma_c_energy = sigma_c_energy.split()
    sigma_c_energy = np.asarray(sigma_c_energy)
    sigma_c_energy = sigma_c_energy.astype(np.float)
    
m_e = 9.109E-31 #kg
e = 1.602E-19 #coulombs
omega = 2*np.pi*Plasma_Resonant_Frequency_Fit #angular frequency
v = np.sqrt(2*e*energy/m_e) #electron velocity m/s
n_g_meters_cubed = n_g*1E6 #n_g in [m^-3]
nu_energy = n_g_meters_cubed*v*sigma_c_energy #energy differential collision cross section for momentum transfer
T_e_sweep = np.linspace(0.1, 15, 1000)#0.1-15eV; 1000 steps

#MAXWELLIAN
f_maxwellian = np.zeros((len(T_e_sweep), len(energy)))
normalization_term = np.zeros((len(T_e_sweep), len(energy)))
f_prime_maxwellian = np.zeros((len(T_e_sweep), len(energy)))

for i in range(0, len(T_e_sweep)):
    f_maxwellian[i] = 2*(np.pi**(-1/2))*T_e_sweep[i]**(-3/2)*np.exp(-energy/T_e_sweep[i])
    normalization_term[i] = np.trapz(np.power(energy, 1/2) * f_maxwellian[i], energy) #Normalize f(energy) such that integral ( energy^1/2 * f(energy)) = 1 = T_eff
    f_maxwellian[i] = f_maxwellian[i]/normalization_term[i] #normalized maxwellian EEDF
    f_prime_maxwellian[i] = np.gradient(f_maxwellian[i])/np.gradient(energy) #derivative of EEDF using gradient method

#BI-MAXWELLIAN
f_bimaxwellian = np.zeros((len(T_e_sweep), len(energy)))
normalization_term_bimaxwellian = np.zeros((len(T_e_sweep), len(energy)))
f_prime_bimaxwellian = np.zeros((len(T_e_sweep), len(energy)))
A = np.zeros((len(T_e_sweep), len(energy)))

beta = 1/9 # n_2/n_1; ratio of hot/cold electron populations
T_1 = 0.5 #eV; lower temperature, kept fixed
T_2_bimaxwellian_sweep = T_e_sweep

for i in range(0, len(T_e_sweep)):
    A = 1/(((np.pi**1/2)/2)*((1-beta)*T_1**(3/2) + beta*T_2_bimaxwellian_sweep[i]**(3/2)))
    f_bimaxwellian[i] = A*((1-beta)*np.exp(-energy/T_1) + beta*np.exp(-energy/T_2_bimaxwellian_sweep[i]))
    normalization_term_bimaxwellian[i] = np.trapz(np.power(energy, 1/2) * f_bimaxwellian[i], energy) #normalize T_eff = integral ( energy^3/2 * f(energy)) = 1
    f_bimaxwellian[i] = f_bimaxwellian[i]/normalization_term_bimaxwellian[i]
    f_prime_bimaxwellian[i] = np.gradient(f_bimaxwellian[i])/np.gradient(energy) #derivative of EEDF using gradient method

#DRUYVESTEYN
f_druyvesteyn = np.zeros((len(T_e_sweep), len(energy)))
normalization_term_druyvesteyn = np.zeros((len(T_e_sweep), len(energy)))
f_prime_druyvesteyn = np.zeros((len(T_e_sweep), len(energy)))

for i in range(0, len(T_e_sweep)):
    f_druyvesteyn[i] = (0.5648*Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*np.sqrt(energy)/np.power(T_e_sweep[i], 3/2))*np.exp(-0.243*np.power(energy/T_e_sweep[i], 2))
    normalization_term[i] = np.trapz(np.power(energy, 1/2) * f_druyvesteyn[i], energy) #normalize T_eff = integral ( energy^3/2 * f(energy)) = 1
    f_druyvesteyn[i] = f_druyvesteyn[i]/normalization_term[i]
    f_prime_druyvesteyn[i] = np.gradient(f_druyvesteyn[i])/np.gradient(energy) #derivative of EEDF using gradient method

#CONDUCTIVITY
integrand_maxwellian = np.zeros((len(T_e_sweep), len(energy)))
sigma_e_maxwellian = np.zeros(len(T_e_sweep))
integrand_bimaxwellian = np.zeros((len(T_e_sweep), len(energy)))
sigma_e_bimaxwellian = np.zeros(len(T_e_sweep))
integrand_druyvesteyn = np.zeros((len(T_e_sweep), len(energy)))
sigma_e_druyvesteyn = np.zeros(len(T_e_sweep))
nu_eff_maxwellian = np.zeros(len(T_e_sweep))
omega_eff_maxwellian = np.zeros(len(T_e_sweep))
nu_eff_bimaxwellian = np.zeros(len(T_e_sweep))
omega_eff_bimaxwellian = np.zeros(len(T_e_sweep))
nu_eff_druyvesteyn = np.zeros(len(T_e_sweep))
omega_eff_druyvesteyn = np.zeros(len(T_e_sweep))

for i in range(0, len(T_e_sweep)):
    #CONDUCTIVITY
    integrand_maxwellian[i] = ((np.power(energy, 3/2)) * f_prime_maxwellian[i])/(nu_energy + 1j*omega)
    sigma_e_maxwellian[i] = -(2*Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*e**2/(3*m_e))*np.trapz(integrand_maxwellian[i], energy)

    integrand_bimaxwellian[i] = ((np.power(energy, 3/2)) * f_prime_bimaxwellian[i])/(nu_energy + 1j*omega)
    sigma_e_bimaxwellian[i] = -(2*Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*e**2/(3*m_e))*np.trapz(integrand_bimaxwellian[i], energy)

    integrand_druyvesteyn[i] = ((np.power(energy, 3/2)) * f_prime_druyvesteyn[i])/(nu_energy + 1j*omega)
    sigma_e_druyvesteyn[i] = -(2*Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*e**2/(3*m_e))*np.trapz(integrand_druyvesteyn[i], energy)

    #Effective collision frequency and angular frequency
    nu_eff_maxwellian[i] = np.real(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_maxwellian[i]*m_e))
    omega_eff_maxwellian[i] = np.imag(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_maxwellian[i]*m_e))

    nu_eff_bimaxwellian[i] = np.real(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_bimaxwellian[i]*m_e))
    omega_eff_bimaxwellian[i] = np.imag(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_bimaxwellian[i]*m_e))

    nu_eff_druyvesteyn[i] = np.real(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_druyvesteyn[i]*m_e))
    omega_eff_druyvesteyn[i] = np.imag(Pressure_and_Sheath_Corrected_Electron_Density_meters_cubed*(e**2)/(sigma_e_druyvesteyn[i]*m_e))

#Find minimum of (nu_eff_EEDF_calculated - nu_eff_measured) and set T_e_EEDF at that value
T_e_maxwellian = T_e_sweep[abs(nu_eff_maxwellian - true_collision_freq).argmin()]
T_2_bimaxwellian = T_e_sweep[abs(nu_eff_bimaxwellian - true_collision_freq).argmin()]
T_e_druyvesteyn = T_e_sweep[abs(nu_eff_druyvesteyn - true_collision_freq).argmin()]

print("T_e_Maxwellian = " + str(T_e_maxwellian) + '[eV]')
print("T_2_bimaxwellian = " + str(T_2_bimaxwellian) + '[eV]')
print("T_e_druyvesteyn = " + str(T_e_druyvesteyn) + '[eV]')

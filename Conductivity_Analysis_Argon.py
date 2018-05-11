# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:27:27 2018

@author: djpeter5
"""
import numpy as np
import pandas as pd
import pylab
import scipy

#Energy and cross section from BOLSIG+, BIAGI - CHANGED 0eV TERM TO 1E-5eV
energy = '1.000000e-5 1.000000e-3 2.000000e-3 5.000000e-3 1.000000e-2 1.741900e-2 3.514200e-2 5.317400e-2 7.151900e-2 9.018400e-2 1.091700e-1 1.285000e-1 1.481500e-1 1.681500e-1 1.885000e-1 2.092100e-1 2.302700e-1 2.517000e-1 2.735000e-1 2.956900e-1 3.182600e-1 3.412200e-1 3.645800e-1 3.883500e-1 4.125400e-1 4.371400e-1 4.621800e-1 4.876500e-1 5.135600e-1 5.399300e-1 5.667500e-1 5.940400e-1 6.218100e-1 6.500600e-1 6.788000e-1 7.080500e-1 7.378000e-1 7.680700e-1 7.988700e-1 8.302100e-1 8.620900e-1 8.945200e-1 9.275200e-1 9.611000e-1 9.952600e-1 1.030000e+0 1.065400e+0 1.101400e+0 1.138000e+0 1.175200e+0 1.213100e+0 1.251600e+0 1.290900e+0 1.330800e+0 1.371400e+0 1.412700e+0 1.454700e+0 1.497500e+0 1.541000e+0 1.585200e+0 1.630300e+0 1.676100e+0 1.722700e+0 1.770100e+0 1.818400e+0 1.867500e+0 1.917400e+0 1.968200e+0 2.020000e+0 2.072600e+0 2.126100e+0 2.180500e+0 2.235900e+0 2.292300e+0 2.349700e+0 2.408000e+0 2.467400e+0 2.527800e+0 2.589200e+0 2.651700e+0 2.715400e+0 2.780100e+0 2.845900e+0 2.912900e+0 2.981100e+0 3.050400e+0 3.121000e+0 3.192800e+0 3.265800e+0 3.340100e+0 3.415700e+0 3.492600e+0 3.570900e+0 3.650500e+0 3.731500e+0 3.813900e+0 3.897800e+0 3.983100e+0 4.069900e+0 4.158200e+0 4.248100e+0 4.339500e+0 4.432500e+0 4.527100e+0 4.623400e+0 4.721400e+0 4.821000e+0 4.922400e+0 5.025600e+0 5.130600e+0 5.237300e+0 5.346000e+0 5.456500e+0 5.569000e+0 5.683400e+0 5.799900e+0 5.918300e+0 6.038800e+0 6.161400e+0 6.286200e+0 6.413100e+0 6.542200e+0 6.673600e+0 6.807300e+0 6.943300e+0 7.081600e+0 7.365700e+0 7.659600e+0 7.964000e+0 8.279000e+0 8.605100e+0 8.942600e+0 9.292000e+0 9.653700e+0 1.002800e+1 1.061400e+1 1.256800e+1 1.304400e+1 1.353800e+1 1.404900e+1 1.457800e+1 1.512500e+1 1.569200e+1 1.627800e+1 1.657900e+1 1.688500e+1 1.719700e+1 1.751400e+1 1.783600e+1 1.816500e+1 1.849800e+1 1.883800e+1 1.918400e+1 1.953500e+1 1.989300e+1 2.062700e+1 2.138700e+1 2.217400e+1 2.257800e+1 2.298800e+1 2.340600e+1 2.383100e+1 2.426400e+1 2.470400e+1 2.515200e+1 2.607100e+1 2.702200e+1 2.751000e+1 2.800700e+1 2.851200e+1 2.902600e+1 2.954900e+1 3.008100e+1 3.117400e+1 3.230400e+1 3.347500e+1 3.468600e+1 3.530800e+1 3.594000e+1 3.658400e+1 3.723800e+1 3.790500e+1 3.858200e+1 3.927200e+1 3.997300e+1 4.141300e+1 4.290400e+1 4.444600e+1 4.604400e+1 4.686300e+1 4.769700e+1 4.854500e+1 4.940800e+1 5.028600e+1 5.208800e+1 5.395400e+1 5.491100e+1 5.588500e+1 5.687600e+1 5.788400e+1 5.891000e+1 5.995400e+1 6.101500e+1 6.209600e+1 6.319500e+1 6.431300e+1 6.545100e+1 6.660800e+1 6.778600e+1 6.898400e+1 7.020300e+1 7.270600e+1 7.529600e+1 7.797700e+1 8.075200e+1 8.217600e+1 8.362500e+1 8.509900e+1 8.659900e+1 8.812500e+1 8.967800e+1 9.125700e+1 9.449900e+1 9.616300e+1 9.785500e+1 9.957700e+1 1.031100e+2 1.067700e+2 1.105600e+2 1.144800e+2 1.185400e+2 1.227400e+2 1.270900e+2 1.315900e+2 1.362500e+2 1.410700e+2 1.435400e+2 1.460600e+2 1.486200e+2 1.538800e+2 1.593200e+2 1.649600e+2 1.707900e+2 1.768300e+2 1.830800e+2 1.895500e+2 1.962400e+2 2.031700e+2 2.103500e+2 2.177800e+2 2.254600e+2 2.334200e+2 2.416600e+2 2.545600e+2 2.681500e+2 2.824700e+2 2.975400e+2 3.134100e+2 3.301300e+2 3.477400e+2 3.662800e+2 3.858100e+2 4.063800e+2 4.355200e+2 4.587300e+2 4.831700e+2 5.089200e+2 5.360300e+2 5.645900e+2 5.946600e+2 6.263400e+2 6.596900e+2 6.948200e+2 7.318200e+2 7.575800e+2 7.842400e+2 8.118300e+2 8.404000e+2 8.699600e+2 9.005700e+2 9.322500e+2 9.650500e+2'
energy = energy.split()
energy = np.asarray(energy)
energy = energy.astype(np.float)

sigma_c_energy = '6.298400e-20 6.298400e-20 5.835000e-20 4.977500e-20 4.113000e-20 3.290600e-20 2.144900e-20 1.474100e-20 1.032600e-20 7.273200e-21 5.117000e-21 3.589000e-21 2.519100e-21 1.793200e-21 1.330700e-21 1.073000e-21 9.762900e-22 1.007000e-21 1.139000e-21 1.351700e-21 1.628900e-21 1.957400e-21 2.326700e-21 2.728200e-21 3.155200e-21 3.602100e-21 4.064500e-21 4.539100e-21 5.023100e-21 5.514600e-21 6.012300e-21 6.515300e-21 7.023300e-21 7.536300e-21 8.054600e-21 8.579000e-21 9.110500e-21 9.650300e-21 1.020000e-20 1.076200e-20 1.133700e-20 1.192800e-20 1.253700e-20 1.316800e-20 1.382100e-20 1.431600e-20 1.479100e-20 1.527500e-20 1.576700e-20 1.626700e-20 1.677000e-20 1.727100e-20 1.778100e-20 1.830000e-20 1.882800e-20 1.936500e-20 1.991100e-20 2.046700e-20 2.107400e-20 2.169300e-20 2.232400e-20 2.296500e-20 2.358000e-20 2.416500e-20 2.476000e-20 2.536600e-20 2.598200e-20 2.660800e-20 2.729100e-20 2.805900e-20 2.884100e-20 2.963600e-20 3.044500e-20 3.126800e-20 3.210500e-20 3.295700e-20 3.382400e-20 3.472800e-20 3.567400e-20 3.663700e-20 3.761600e-20 3.861300e-20 3.962700e-20 4.065900e-20 4.170900e-20 4.275600e-20 4.381500e-20 4.489100e-20 4.598700e-20 4.710200e-20 4.823600e-20 4.938900e-20 5.056300e-20 5.175800e-20 5.297300e-20 5.420900e-20 5.546700e-20 5.674700e-20 5.832800e-20 6.000600e-20 6.171300e-20 6.345000e-20 6.521800e-20 6.701600e-20 6.884500e-20 7.070600e-20 7.260000e-20 7.452600e-20 7.651200e-20 7.861100e-20 8.074700e-20 8.292000e-20 8.513100e-20 8.738000e-20 8.966900e-20 9.199700e-20 9.436600e-20 9.673800e-20 9.906700e-20 1.014400e-19 1.038500e-19 1.063000e-19 1.088000e-19 1.113400e-19 1.139200e-19 1.163100e-19 1.208500e-19 1.255500e-19 1.304200e-19 1.354600e-19 1.406800e-19 1.460800e-19 1.513800e-19 1.568100e-19 1.621700e-19 1.656900e-19 1.620300e-19 1.586500e-19 1.547000e-19 1.505600e-19 1.458000e-19 1.408700e-19 1.357700e-19 1.305000e-19 1.277900e-19 1.250300e-19 1.222300e-19 1.193700e-19 1.164700e-19 1.137700e-19 1.112600e-19 1.087100e-19 1.061200e-19 1.034900e-19 1.008000e-19 9.717800e-20 9.375800e-20 9.021700e-20 8.840100e-20 8.655300e-20 8.467200e-20 8.275900e-20 8.081300e-20 7.883200e-20 7.704500e-20 7.428800e-20 7.143400e-20 6.996900e-20 6.848000e-20 6.696400e-20 6.542100e-20 6.385200e-20 6.235400e-20 6.038700e-20 5.835200e-20 5.624600e-20 5.406500e-20 5.294600e-20 5.180800e-20 5.064900e-20 4.947100e-20 4.827200e-20 4.705200e-20 4.581100e-20 4.454800e-20 4.315800e-20 4.174200e-20 4.027600e-20 3.875900e-20 3.798000e-20 3.718800e-20 3.638200e-20 3.556200e-20 3.480000e-20 3.353800e-20 3.223200e-20 3.156200e-20 3.088000e-20 3.018700e-20 2.948100e-20 2.876300e-20 2.803200e-20 2.744100e-20 2.684700e-20 2.624300e-20 2.562800e-20 2.500200e-20 2.436500e-20 2.371800e-20 2.305900e-20 2.244900e-20 2.182400e-20 2.117600e-20 2.050600e-20 1.977400e-20 1.934700e-20 1.891200e-20 1.847000e-20 1.802000e-20 1.756200e-20 1.709700e-20 1.674900e-20 1.610000e-20 1.576700e-20 1.542900e-20 1.508500e-20 1.465100e-20 1.424200e-20 1.381800e-20 1.337800e-20 1.292400e-20 1.245300e-20 1.201600e-20 1.162000e-20 1.121000e-20 1.078600e-20 1.056800e-20 1.034700e-20 1.012100e-20 9.829200e-21 9.589700e-21 9.341800e-21 9.085200e-21 8.819600e-21 8.544600e-21 8.260000e-21 7.965300e-21 7.711100e-21 7.510200e-21 7.302300e-21 7.087000e-21 6.864200e-21 6.633500e-21 6.317800e-21 6.073200e-21 5.815600e-21 5.544300e-21 5.312200e-21 5.078200e-21 4.831700e-21 4.637200e-21 4.441900e-21 4.252100e-21 4.033600e-21 3.859500e-21 3.676200e-21 3.501000e-21 3.351800e-21 3.194800e-21 3.029400e-21 2.894700e-21 2.761200e-21 2.620700e-21 2.472700e-21 2.369700e-21 2.263100e-21 2.164500e-21 2.078800e-21 1.990100e-21 1.898300e-21 1.803200e-21 1.704800e-21'
sigma_c_energy = sigma_c_energy.split()
sigma_c_energy = np.asarray(sigma_c_energy)
sigma_c_energy = sigma_c_energy.astype(np.float)

#Define Constants
n_e = 5E9*1E6 #m^-3
m_e = 9.109E-31 #kg
e = 1.602E-19 #coulombs
applied_frequency = 2E9 #Hz
omega = 2*np.pi*applied_frequency #angular frequency
Pressure = 1 #Torr
Pressure_sweep = np.linspace(0.001, 100, 100000) #1mTorr-10Torr; 100 steps
N = Pressure*3.25E16*1E6 #neutral gas number density m^-3 @ 300K
N_sweep = Pressure_sweep*3.25E16*1E6 
v = np.sqrt(2*e*energy/m_e) #electron velocity m/s
nu_energy = N*v*sigma_c_energy #energy differential collision cross section for momentum transfer
nu_energy_sweep = np.outer(N_sweep, v*sigma_c_energy) #energy differential collision cross section for momentum transfer

#MAXWELLIAN
T_e_sweep = np.linspace(0.1, 10, 100)#0.1-10eV; 100 steps
T_e = 1.9
f_maxwellian = 2*(np.pi**(-1/2))*T_e**(-3/2)*np.exp(-energy/T_e)
normalization_term = np.trapz(np.power(energy, 1/2) * f_maxwellian, energy) #Normalize f(energy) such that integral ( energy^1/2 * f(energy)) = 1 = T_eff
f_maxwellian = f_maxwellian/normalization_term
avg_electron_energy = np.trapz(np.power(energy, 3/2) * f_maxwellian, energy)

f_maxwellian_comparison = 2*(np.pi**(-1/2))*T_e**(-3/2)*np.exp(-energy/T_e)
normalization_term = (2/3)*np.trapz(np.power(energy, 3/2) * f_maxwellian_comparison, energy) #Normalize f(energy) such that integral ( energy^3/2 * f(energy)) = 1 = T_eff
f_maxwellian_comparison = f_maxwellian/normalization_term #maxwellian to be used to compare to bi-maxwellian and Druyvesteyn

f_prime_maxwellian = np.gradient(f_maxwellian)/np.gradient(energy) #derivative of EEDF using gradient method
f_prime_maxwellian_comparison = np.gradient(f_maxwellian_comparison)/np.gradient(energy) #derivative of EEDF using gradient method

#BI-MAXWELLIAN
beta = 1/9 # n_2/n_1; ratio of hot/cold electron populations
T_1 = 0.5 #eV; lower temperature
T_2 = 5 #eV; higher temperature
A = 1/(((np.pi**1/2)/2)*((1-beta)*T_1**(3/2) + beta*T_2**(3/2)))
f_bimaxwellian = A*((1-beta)*np.exp(-energy/T_1) + beta*np.exp(-energy/T_2))
normalization_term = (2/3)*np.trapz(np.power(energy, 3/2) * f_bimaxwellian, energy) #normalize T_eff = integral ( energy^3/2 * f(energy)) = 1
f_bimaxwellian = f_bimaxwellian/normalization_term
f_prime_bimaxwellian = np.gradient(f_bimaxwellian)/np.gradient(energy) #derivative of EEDF using gradient method

#DRUYVESTEYN
#NEED TO NORMALIZE EEDF
T_druyvesteyn = 1.58 #eV; higher temperature
f_druyvesteyn = (0.5648*n_e*np.sqrt(energy)/np.power(T_druyvesteyn, 3/2))*np.exp(-0.243*np.power(energy/T_druyvesteyn, 2))
normalization_term = (2/3)*np.trapz(np.power(energy, 3/2) * f_druyvesteyn, energy) #normalize T_eff = integral ( energy^3/2 * f(energy)) = 1
f_druyvesteyn = f_druyvesteyn/normalization_term
f_prime_druyvesteyn = np.gradient(f_druyvesteyn)/np.gradient(energy) #derivative of EEDF using gradient method


#CONDUCTIVITY
integrand = ((np.power(energy, 3/2)) * f_prime_maxwellian)/(nu_energy_sweep + 1j*omega)
sigma_e = -(2*n_e*e**2/(3*m_e))*np.trapz(integrand, energy)

integrand = ((np.power(energy, 3/2)) * f_prime_bimaxwellian)/(nu_energy_sweep + 1j*omega)
sigma_e_bimaxwellian = -(2*n_e*e**2/(3*m_e))*np.trapz(integrand, energy)

integrand = ((np.power(energy, 3/2)) * f_prime_druyvesteyn)/(nu_energy_sweep + 1j*omega)
sigma_e_druyvesteyn = -(2*n_e*e**2/(3*m_e))*np.trapz(integrand, energy)

###########
nu_eff_5eV = np.real(n_e*(e**2)/(sigma_e*m_e))
omega_eff = np.imag(n_e*(e**2)/(sigma_e*m_e))

nu_eff_bimaxwellian_5eV = np.real(n_e*(e**2)/(sigma_e_bimaxwellian*m_e))
omega_eff_bimaxwellian = np.imag(n_e*(e**2)/(sigma_e_bimaxwellian*m_e))

nu_eff_druyvesteyn_5eV = np.real(n_e*(e**2)/(sigma_e_druyvesteyn*m_e))
omega_eff_druyvesteyn = np.imag(n_e*(e**2)/(sigma_e_druyvesteyn*m_e))

##########
nu_eff_3eV_6GHz = np.real(n_e*(e**2)/(sigma_e*m_e))
omega_eff_6GHz = np.imag(n_e*(e**2)/(sigma_e*m_e))

nu_eff_bimaxwellian_3eV_6GHz = np.real(n_e*(e**2)/(sigma_e_bimaxwellian*m_e))
omega_eff_bimaxwellian_6GHz = np.imag(n_e*(e**2)/(sigma_e_bimaxwellian*m_e))

nu_eff_druyvesteyn_3eV_6GHz = np.real(n_e*(e**2)/(sigma_e_druyvesteyn*m_e))
omega_eff_druyvesteyn_6GHz = np.imag(n_e*(e**2)/(sigma_e_druyvesteyn*m_e))

###########
integrand_dc = ((energy**(3/2)) * f_prime_maxwellian)/(nu_energy_sweep)
nu_dc_3eV = 1/(-(2/3)*np.trapz(integrand_dc, energy))

integrand_dc_bimaxwellian = ((energy**(3/2)) * f_prime_bimaxwellian)/(nu_energy_sweep)
nu_dc_bimaxwellian_3eV = 1/(-(2/3)*np.trapz(integrand_dc_bimaxwellian, energy))

integrand_dc_druyvesteyn = ((np.power(energy,3/2)) * f_prime_druyvesteyn)/(nu_energy_sweep)
nu_dc_druyvesteyn_3eV = 1/(-(2/3)*np.trapz(integrand_dc_druyvesteyn, energy))

pylab.loglog(Pressure_sweep, nu_eff_1eV/nu_dc_1eV, Pressure_sweep, nu_eff_2eV/nu_dc_2eV,Pressure_sweep, nu_eff_3eV/nu_dc_3eV)
pylab.xlabel('Pressure [Torr]')
pylab.ylabel(r'$\nu_{eff} / \nu_{dc}$')
pylab.ylim(0.7, 3)
pylab.xlim(0.1, 100)
pylab.legend(('5eV - 1.44GHz', '5eV - 2.73GHz','5eV - 6GHz'))
pylab.show() 

pylab.loglog(Pressure_sweep, nu_eff_3eV/nu_dc_3eV, Pressure_sweep, nu_eff_bimaxwellian_3eV/nu_dc_bimaxwellian_3eV, Pressure_sweep, nu_eff_druyvesteyn_3eV/nu_dc_druyvesteyn_3eV)
pylab.xlabel('Pressure [Torr]')
pylab.ylabel(r'$\nu_{eff} / \nu_{dc}$')
pylab.ylim(0.7, 20)
pylab.xlim(0.1, 100)
pylab.legend(('Maxwellian - 2GHz', 'Bi-Maxwellian - 2GHz', 'Druyvesteyn - 2GHz'))
pylab.show() 

pylab.semilogy(energy, f_maxwellian_comparison, energy, f_bimaxwellian, energy, f_druyvesteyn)
pylab.xlabel('Energy [eV]')
pylab.ylabel(r'$f(\epsilon)$')
pylab.xlim(0, 15)
pylab.ylim(0.001,3)
pylab.legend(('Maxwellian', 'Bi-Maxwellian', 'Druyvesteyn'))
pylab.show() 

pylab.loglog(Pressure_sweep, 1E-9*nu_eff_3eV/Pressure_sweep, Pressure_sweep, 1E-9*nu_eff_bimaxwellian_3eV/Pressure_sweep, Pressure_sweep, 1E-9*nu_eff_druyvesteyn_3eV/Pressure_sweep)
pylab.xlabel('Pressure [Torr]')
pylab.ylabel(r'$10^{-9}\nu_{eff}/p [s^{-1} Torr^{-1}]$')
pylab.ylim(0.1, 30)
pylab.xlim(0.001, 10)
pylab.legend(('Maxwellian - 1.5GHz', 'Bi-Maxwellian - 1.5GHz', 'Druyvesteyn - 1.5GHz'))
pylab.show() 

pylab.loglog(Pressure_sweep, 1E-9*nu_eff_bimaxwellian_3eV_1pt5GHz/Pressure_sweep, 'b--', Pressure_sweep, 1E-9*nu_eff_bimaxwellian_3eV_3GHz/Pressure_sweep, 'b-.', Pressure_sweep, 1E-9*nu_eff_bimaxwellian_3eV_6GHz/Pressure_sweep, 'b', Pressure_sweep, 1E-9*nu_eff_3eV_1pt5GHz/Pressure_sweep, 'g--', Pressure_sweep, 1E-9*nu_eff_3eV_3GHz/Pressure_sweep, 'g-.', Pressure_sweep, 1E-9*nu_eff_3eV_6GHz/Pressure_sweep, 'g', Pressure_sweep, 1E-9*nu_eff_druyvesteyn_3eV_1pt5GHz/Pressure_sweep, 'r--', Pressure_sweep, 1E-9*nu_eff_druyvesteyn_3eV_3GHz/Pressure_sweep, 'r-.', Pressure_sweep, 1E-9*nu_eff_druyvesteyn_3eV_6GHz/Pressure_sweep, 'r')
pylab.xlabel('Pressure [Torr]')
pylab.ylabel(r'$10^{-9}\nu_{eff}/p [s^{-1} Torr^{-1}]$')
pylab.ylim(0.1, 30)
pylab.xlim(0.001, 10)
#pylab.legend(('Bi-Maxwellian - 1.5GHz', 'Bi-Maxwellian - 3GHz', 'Bi-Maxwellian - 6GHz', 'Maxwellian - 1.5GHz', 'Maxwellian - 3GHz', 'Maxwellian - 6GHz', 'Druyvesteyn - 1.5GHz', 'Druyvesteyn - 3GHz', 'Druyvesteyn - 6GHz'))
pylab.show() 

#Feed in hairpin measured collision frequencies and sweep EEDFs/T_eff until they match - needs 
#This method requires a measurement of nu_dc (Langmuir probe) to match with
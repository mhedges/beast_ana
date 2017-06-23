import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt

from matplotlib import rc
from pylab import MaxNLocator

from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.plotting.style import set_style
from root_numpy import root2rec, hist2array, stack, stretch
from ROOT import TFile, TH1F, gROOT, TGraph

import iminuit, probfit

import rootpy.plotting.root2matplotlib as rplt
import ROOT

from os.path import expanduser


root_style = True

### Set matplotlib style
# Atlas style
#import atlas_style_mpl
#style = atlas_style_mpl.style_mpl()
#plt.style.use(style)

if root_style == True :
# Belle2 Style
    import belle2style_mpl
    style = belle2style_mpl.b2_style_mpl()
    plt.style.use(style)

elif root_style == False :
    import seaborn as sns
    sns.set(color_codes=True)


###############################################################################



# Get run names for various studies
def run_names(run_name):
    LER_Beamsize = []

    LER_Fill = [6100.1,6100.2,6100.3,6100.4]
    HER_Fill = []

    LER_Vacuumbump = []
    HER_Vacuumbump = []

    HER_Toushek = []
    LER_Toushek = []

    HER_Chromaticity = []

    HER_Injection = []
    LER_Injection = []
    
    HER_ToushekTPC = ['BEAST_run5100.root', 'BEAST_run9001.root']

    LER_ToushekTPC =(
            ['BEAST_run10000.root','BEAST_run10001.root','BEAST_run10002.root','BEAST_run10003.root','BEAST_run10004.root'])

    sim_LER_ToushekTPC =(
            ['mc_beast_run_10002.root','mc_beast_run_10003.root','mc_beast_run_10004.root'])

    if run_name == 'LER_ToushekTPC': return LER_ToushekTPC
    if run_name == 'sim_LER_ToushekTPC': return sim_LER_ToushekTPC
    elif run_name == 'HER_ToushekTPC': return HER_ToushekTPC


# Calculate expected rate by scaling simulation to beam parameters present in
# data runs 10002-10004 (including subrun structure)
def calc_sim_weights(datapath, simpath):

    ### Populate beam parameter arrays from data
    total_time = 0
    subrun_durations = []
    subrun_IPZ2 = []
    subrun_I2sigY = []
    subrun_IP = []
    subrun_IPsigYZ2 = []
    print('Getting weights for simulation for BEAST runs 10002-10004 ... ')
    for f in os.listdir(datapath) :
        fname = str(datapath) + str(f) 
        data = root2rec(fname, 'tout')
        for i in range(np.max(data.subrun)+1) :
            if i == 0 : continue
            P_avg = np.mean(data.SKB_LER_pressures_local_corrected[data.subrun==i])[0]

            LER_current = stretch(data[ (
                                        (data.subrun==i)
                                        & (data.SKB_LER_injectionFlag_safe==0)
                                        )],
                                ['SKB_LER_current'])['SKB_LER_current']
            #I_avg = np.mean(LER_current)
            I_avg = np.mean(data.SKB_LER_current[data.subrun==i])[0]
                
            LER_beamsize = stretch(data[ (
                                        (data.subrun==i)
                                        & (data.SKB_LER_injectionFlag_safe==0)
                                        )],
                                ['SKB_LER_correctedBeamSize_xray_Y'])['SKB_LER_correctedBeamSize_xray_Y']
                                #['SKB_LER_beamSize_xray_Y'])['SKB_LER_beamSize_xray_Y']
            #sigmaY_avg = np.mean(LER_beamsize)
            sigmaY_avg = np.mean(data.SKB_LER_correctedBeamSize_xray_Y[data.subrun==i])[0]

            LER_Zeff = stretch(data[ (
                                        (data.subrun==i)
                                        & (data.SKB_LER_injectionFlag_safe==0)
                                        )],
                                ['SKB_LER_Zeff_D02'])['SKB_LER_Zeff_D02']
            #Z_eff = np.mean(LER_Zeff)
            Z_eff = np.mean(data.SKB_LER_Zeff_D02[data.subrun==i])[0]
            #Z_eff = 2.7
                    

            total_time += len(data[data.subrun==i])
            subrun_durations.append(len(data[data.subrun==i]))
            subrun_IPZ2.append(I_avg*P_avg*(Z_eff**2))
            subrun_I2sigY.append(I_avg**2/sigmaY_avg)
            subrun_IP.append(I_avg*P_avg)
            subrun_IPsigYZ2.append(I_avg/(P_avg*sigmaY_avg*Z_eff**2))
    
    subrun_durations = np.array(subrun_durations)
    subrun_IPZ2 = np.array(subrun_IPZ2)
    subrun_I2sigY = np.array(subrun_I2sigY)
    subrun_IP = np.array(subrun_IP)
    subrun_IPsigYZ2 = np.array(subrun_IPsigYZ2)

    TouschekPlot_vals = [subrun_IPsigYZ2, subrun_IPZ2]

    return (subrun_durations, subrun_IPZ2, subrun_I2sigY, TouschekPlot_vals)


    
def neutron_rate_data(datapath):

    runs = run_names('LER_ToushekTPC')

    # Variables for Peter Toushek units
    peter_y = [] # n_neutrons/(current * local pressure)
    peter_y3 = [] # n_neutrons/(current * local pressure)
    peter_y4 = []
    peter_x = [] # current/(local pressure * beamsize)

    y3_errs = []
    y4_errs = []
    y_errs = []

    x_errs = []

    ch3_rate = []
    ch3_errs = []
    ch4_rate = []
    ch4_errs = []

    lengths = []


    branches = [
                'TPC3_N_neutrons',
                'TPC3_dEdx',
                'TPC3_npoints',
                'TPC4_N_neutrons',
                'TPC4_dEdx'
                'TPC4_npoints',
                'subrun',
                'SKB_LER_injectionFlag_safe ',
                ]


    for f in os.listdir(datapath):
        if f not in runs: continue
        ifile = datapath
        ifile += f

        print(ifile)

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile,'tout')

        for i in range(1, np.max(data.subrun)+1 ):
            if i == 0 : continue
            TPC3_npoints = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_npoints'])['TPC3_npoints']
            TPC3_dEdx = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_dEdx'])['TPC3_dEdx']
            TPC3_PID_neutrons = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_PID_neutrons'])['TPC3_PID_neutrons']
            TPC4_npoints = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_npoints'])['TPC4_npoints']
            TPC4_dEdx = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_dEdx'])['TPC4_dEdx']
            TPC4_PID_neutrons = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_PID_neutrons'])['TPC4_PID_neutrons']

            TPC3_length = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_length'])['TPC3_length']

            TPC4_length = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_length'])['TPC4_length']

            ch3 = TPC3_PID_neutrons[(
                                    (TPC3_dEdx * 1.18 > 500.0)
                                    & (TPC3_npoints > 40)
                                    #& (TPC3_length > 2000)
                                    )].sum()/len(data[data.subrun == i])
            ch4 = TPC4_PID_neutrons[(
                                    (TPC4_dEdx * 1.64 > 500.0)
                                    & (TPC4_npoints > 40)
                                    #& (TPC4_length > 2000)
                                    )].sum()/len(data[data.subrun == i])
            lengths.append(len(data[data.subrun == i]))

            ch3_rate.append(ch3)
            ch3_errs.append((len(data[data.subrun == i])))
            ch4_rate.append(ch4)
            ch4_errs.append((len(data[data.subrun == i])))


    ch3_rate = np.array(ch3_rate)
    ch4_rate = np.array(ch4_rate)

    ch3_errs = np.array(ch3_errs)
    ch4_errs = np.array(ch4_errs)

    lengths = np.array(lengths)
    return ch3_rate, ch3_errs, ch4_rate, ch4_errs


    '''
    for f in os.listdir(datapath):
        counter3 = 0
        counter4 = 0
        counter = 0
        ip = 0
        bs = 0
        ps = 0
        x = 0
        n3 = 0
        n4 = 0
        z_eff = 0
        ifile = str(datapath) + str(f)

        data = root2rec(ifile, 'tout')

        ### Old stupid way to do it
        for event in data:
            if (event.SKB_LER_injectionFlag_safe == 0 ): 
                counter += 1

                ### Method for applying additional neutron selections
                for i in range(len(event.TPC3_PID_neutrons)):
                    if (event.TPC3_PID_neutrons[i] == 1 and
                            event.TPC3_dEdx[i] * 1.18 > 500 and
                            event.TPC3_npoints[i] > 40) :
                        n3 += 1

                for i in range(len(event.TPC4_PID_neutrons)):
                    if (event.TPC4_PID_neutrons[i] == 1 and
                            event.TPC4_dEdx[i] * 1.64 > 500 and
                            event.TPC4_npoints[i] > 40) :
                        n4 += 1

                current = event.SKB_LER_current[0]
                local_pressure = event.SKB_LER_pressures_local_corrected[0]
                beam_size = event.SKB_LER_correctedBeamSize_xray_Y[0]

                ip += local_pressure * current
                ps += beam_size * local_pressure
                x += current / (local_pressure * beam_size)
                z_eff += event.SKB_LER_Zeff_D02[0]**2



        neutrons3 = n3 / len(data) #* 1000.
        neutrons4 = n4 / len(data) #* 1000.

        print('Printing avg rate in', f, '\n(old method)')
        print(neutrons3, neutrons4)
        neutrons = (neutrons3 +  neutrons4)# * 1000.
        z_effavg = z_eff / len(data)

        mean_ip = ip / counter
        #print('Printing mean IP for', f)
        #print(mean_ip)
        ip_err = mean_ip / np.sqrt(counter)
        mean_x = x / counter 
        x_err = mean_x / np.sqrt(counter)
        #mean_ps = ps / len(data)

        y = ((n3 + n4)/counter) / (ip/counter) 
        y3 = (n3/counter) / (ip/counter) 
        y4 = (n4/counter) / (ip/counter)
        print("y's:", y, y3, y4)
        input('well?')

        y3_err = y3 / np.sqrt(counter)
        y3_errs.append(y3_err)
        y4_err = y4 / np.sqrt(counter)
        y4_errs.append(y4_err)
        y_err = y / np.sqrt(counter)
        y_errs.append(y_err)

        x_errs.append(x_err)

        #print(neutrons3, neutrons4, neutrons, mean_ip)


        peter_y.append(y)
        peter_y3.append(y3)
        peter_y4.append(y4)
        peter_x.append(mean_x)

    peter_y = np.array(peter_y)
    peter_y3 = np.array(peter_y3)
    peter_y4 = np.array(peter_y4)
    peter_x = np.array(peter_x)

    y_errs = np.array(y_errs)
    y3_errs = np.array(y3_errs)
    y4_errs = np.array(y4_errs)

    x_errs = np.array(x_errs)

    #print(peter_y3, peter_y4, peter_y)
    #print(y3_errs, y4_errs, y_errs)
    #input('well?')

    fit_range = (0.01, 0.03)
    fit_range = (min(peter_x), max(peter_x))
    chi2 = probfit.Chi2Regression(probfit.linear, peter_x, peter_y,
            y_errs)
    minu = iminuit.Minuit(chi2)
    minu.migrad()
    pars = minu.values
    p_errs = minu.errors
    print(pars, p_errs)
    #input('well?')

    if root_style == True :
        color = 'black'
    elif root_style == False :
        color = 'blue'

    #f = plt.figure()
    #ax1 = f.add_subplot(111)
    #chi2.draw(minu)
    ##chi2.draw(minu, print_par=False)
    #ax1.errorbar(peter_x, peter_y, xerr=x_errs, yerr=y_errs, fmt='o', color=color)
    ##ax1.scatter(peter_x, peter_y, color=color)
    #ax1.set_xlabel('current / (pressure * beamsize * Zeff^2)')
    #ax1.set_ylabel('n_neutrons/(current * pressure * Zeff^2)')
    ##ax1.set_xlim([0.0,5E6])
    ##ax1.set_ylim([0.0,120.0])
    #ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #f.savefig('TPC_peter_toushek_measurement_simcuts.pdf')
    #plt.show()

    return peter_x, peter_y, x_errs, y_errs, chi2, minu
    '''

    #chi23 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_3,
    #        rate3_errs)
    #minu3 = iminuit.Minuit(chi23)
    #minu3.migrad()

    ##g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    #g = plt.figure()
    #bx1 = g.add_subplot(111)
    #chi23.draw(minu3, print_par=False)
    #bx1.errorbar(avg_beamsize, rate_3, yerr=rate3_errs, fmt='o', color=color)
    #bx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    #bx1.set_ylabel('Fast neutron rate in Ch. 3 (Hz)')
    #bx1.set_xlim([0.0,0.030])
    #bx1.set_ylim([0.0,0.09])

    #chi24 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_4,
    #        rate4_errs)
    #minu4 = iminuit.Minuit(chi24)
    #minu4.migrad()
    #g.savefig('TPC3_toushek_measurement.pdf')
    #plt.show()

def neutron_angles_data(datapath):

    runs = run_names('LER_ToushekTPC')

    # Variables for Peter Toushek units

    ch3_thetas = []
    ch3_phis = []
    ch4_thetas = []
    ch4_phis = []

    branches = [
                'TPC3_N_neutrons',
                'TPC3_dEdx',
                'TPC3_npoints',
                'TPC4_N_neutrons',
                'TPC4_dEdx'
                'TPC4_npoints',
                'TPC3_theta',
                'TPC3_phi',
                'TPC4_theta',
                'TPC4_phi',
                'subrun',
                'SKB_LER_injectionFlag_safe ',
                ]


    for f in os.listdir(datapath):
        if f not in runs: continue
        ifile = datapath
        ifile += f

        print(ifile)

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile,'tout')

        for i in range(1, np.max(data.subrun)+1 ):
            if i == 0 : continue
            TPC3_npoints = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_npoints'])['TPC3_npoints']
            TPC3_dEdx = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_dEdx'])['TPC3_dEdx']
            TPC3_PID_neutrons = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_PID_neutrons'])['TPC3_PID_neutrons']
            TPC4_npoints = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_npoints'])['TPC4_npoints']
            TPC4_dEdx = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_dEdx'])['TPC4_dEdx']
            TPC4_PID_neutrons = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_PID_neutrons'])['TPC4_PID_neutrons']

            TPC3_length = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_length'])['TPC3_length']

            TPC4_length = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_length'])['TPC4_length']

            TPC3_phi = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_phi'])['TPC3_phi']

            TPC3_theta = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC3_theta'])['TPC3_theta']

            TPC4_phi = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_phi'])['TPC4_phi']

            TPC4_theta = stretch(data[( (data.subrun == i)
                                        & (data.SKB_LER_injectionFlag_safe == 0) )],
                    ['TPC4_theta'])['TPC4_theta']

            ch3_thetas = np.concatenate([ch3_thetas,
                                         TPC3_theta[(
                                                      (TPC3_PID_neutrons == 1)
                                                    & (TPC3_dEdx *1.18 > 500.0)
                                                    & (TPC3_npoints > 40)
                                                    & (TPC3_length > 2000.0)
                                                    )]
                                         ])

            ch3_phis = np.concatenate([ch3_phis,
                                         TPC3_phi[(
                                                      (TPC3_PID_neutrons == 1)
                                                    & (TPC3_dEdx *1.18 > 500.0)
                                                    & (TPC3_npoints > 40)
                                                    & (TPC3_length > 2000.0)
                                                    )]
                                         ])

            ch4_thetas = np.concatenate([ch4_thetas,
                                         TPC4_theta[(
                                                      (TPC4_PID_neutrons == 1)
                                                    & (TPC4_dEdx *1.64 > 500.0)
                                                    & (TPC4_npoints > 40)
                                                    & (TPC4_length > 2000.0)
                                                    )]
                                         ])

            ch4_phis = np.concatenate([ch4_phis,
                                         TPC4_phi[(
                                                      (TPC4_PID_neutrons == 1)
                                                    & (TPC4_dEdx *1.64 > 500.0)
                                                    & (TPC4_npoints > 40)
                                                    & (TPC4_length > 2000.0)
                                                    )]
                                         ])

    # Fold angles
    ch3_folded_phis = ch3_phis
    ch3_folded_thetas = ch3_thetas

    ch3_folded_thetas[( (ch3_folded_phis < -90) )] *= -1
    ch3_folded_thetas[( (ch3_folded_phis < -90) )] += 180

    ch3_folded_thetas[( (ch3_folded_phis > 90) )] *= -1
    ch3_folded_thetas[( (ch3_folded_phis > 90) )] += 180

    ch3_folded_phis[( (ch3_folded_phis < -90) )] += 180
    ch3_folded_phis[( (ch3_folded_phis > 90) )] -= 180

    ch4_folded_phis = ch4_phis
    ch4_folded_thetas = ch4_thetas

    ch4_folded_thetas[( (ch4_folded_phis < -90) )] *= -1
    ch4_folded_thetas[( (ch4_folded_phis < -90) )] += 180

    ch4_folded_thetas[( (ch4_folded_phis > 90) )] *= -1
    ch4_folded_thetas[( (ch4_folded_phis > 90) )] += 180

    ch4_folded_phis[( (ch4_folded_phis < -90) )] += 180
    ch4_folded_phis[( (ch4_folded_phis > 90) )] -= 180

    # Calculate BP vs non-BP arrays
    ch3_bp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis < 20)]
    ch4_bp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis < 20)]

    ch3_nbp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis > 40)]
    ch4_nbp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis > 40)]

    ch3_nbp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis > 40)]
    ch4_nbp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis > 40)]

    return (ch3_thetas, ch4_thetas, ch3_phis, ch4_phis, ch3_bp_thetas,
            ch3_nbp_thetas, ch4_bp_thetas, ch4_nbp_thetas)

def neutron_rate_raw_sim(datapath):

    runs = run_names('LER_ToushekTPC')

    # Variables for Peter Toushek units
    peter_y = [] # n_neutrons/(current * local pressure)
    peter_y3 = [] # n_neutrons/(current * local pressure)
    peter_y4 = []
    peter_x = [] # current/(local pressure * beamsize)

    y3_errs = []
    y4_errs = []
    y_errs = []

    x_errs = []

    ch3_rate = []
    ch3_errs = []
    ch4_rate = []
    ch4_errs = []

    lengths = []

    branches = [
                'de_dx',
                'e_sum',
                'npoints',
                'npoints',
                ]


    for f in os.listdir(datapath):
        #if f not in runs: continue
        ifile = datapath
        ifile += f

        print(ifile)

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile,'tr')


    ch3_rate = np.array(ch3_rate)
    ch4_rate = np.array(ch4_rate)

    ch3_errs = np.array(ch3_errs)
    ch4_errs = np.array(ch4_errs)

    lengths = np.array(lengths)
    return ch3_rate, ch3_errs, ch4_rate, ch4_errs

def neutron_rate_sim(datapath):

    runs = run_names('sim_LER_ToushekTPC')

    # Variables for Peter Toushek units
    peter_y = [] # n_neutrons/(current * local pressure)
    peter_y3 = [] # n_neutrons/(current * local pressure)
    peter_y4 = []
    peter_x = [] # current/(local pressure * beamsize)

    y3_errs = []
    y4_errs = []
    y_errs = []

    x_errs = []

    ch3_rate = []
    ch3_errs = []
    ch4_rate = []
    ch4_errs = []


    branches = [
                'TPC_angular_rate_av',
                'subrun',
                ]


    for f in os.listdir(datapath):
        if f not in runs: continue
        ifile = datapath
        ifile += f

        print(ifile)

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile,'tout')

        for i in range(1, np.max(data.subrun)+1):
            ch3 = data.TPC_angular_rate_av[:,0][data.subrun==i].sum()
            ch3 /= len(data[data.subrun>0])
            ch3_rate.append(ch3)
            ch3_errs.append(len(data[data.subrun==i]))
            ch4 = data.TPC_angular_rate_av[:,1][data.subrun==i].sum()
            ch4 /= len(data[data.subrun==i])
            ch4_rate.append(ch4)
            ch4_errs.append(len(data[data.subrun==i]))

    return ch3_rate, ch3_errs, ch4_rate, ch4_errs

    '''
    # Variables for Peter Toushek units
    peter_y = [] # n_neutrons/(current * local pressure)
    peter_y3 = [] # n_neutrons/(current * local pressure)
    peter_y4 = []
    peter_x = [] # current/(local pressure * beamsize)

    y3_errs = []
    y4_errs = []
    y_errs = []

    x_errs = []

    for f in os.listdir(datapath):
        if f not in runs: continue
        ifile = datapath
        ifile += f

        print(ifile)

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile, 'tout')

        counter3 = 0
        counter4 = 0
        counter = 0
        ip = 0
        bs = 0
        ps = 0
        x = 0
        n3 = 0
        n4 = 0

        for event in data:
            if (event.SKB_LER_injectionFlag_safe == 0 and
                    event.SKB_LER_beamSize_xray_Y < 400 and
                    event.SKB_LER_beamSize_xray_Y > 35 and
                    event.SKB_LER_current > 10):
                #n3 += (event.TPC3_N_neutrons[0] if
                #        len(event.TPC3_N_neutrons) > 0 else 0)
                counter += 1

                #n3 += event.TPC_rate_av[0][0]
                #n4 += event.TPC_rate_av[1][0]

                n3 += np.sum(event.TPC_angular_rate_av[0])
                n4 += np.sum(event.TPC_angular_rate_av[1])

                current = event.SKB_LER_current[0]
                #local_pressure = event.SKB_LER_pressures_local_corrected[0]
                local_pressure = event.SKB_LER_pressures_local[0] * 3.0
                beam_size = event.SKB_LER_beamSize_xray_Y[0]

                ip += local_pressure * current
                ps += beam_size * local_pressure
                x += current / (local_pressure * beam_size)

        neutrons3 = n3 / len(data) #* 1000.
        neutrons4 = n4 / len(data) #* 1000.
        neutrons = (neutrons3 +  neutrons4)# * 1000.
        z_effavg = 2.7**2

        mean_ip = ip / counter
        ip_err = mean_ip / np.sqrt(counter)
        #mean_x = x / counter / z_effavg
        mean_x = x / counter
        x_err = mean_x / np.sqrt(counter)
        #mean_ps = ps / len(data)


        #y = ((n3 + n4)/counter) / (ip/counter) / z_effavg
        #y3 = (n3/counter) / (ip/counter) / z_effavg
        #y4 = (n4/counter) / (ip/counter) / z_effavg
        y = ((n3 + n4)/counter) / (ip/counter)
        y3 = (n3/counter) / (ip/counter)
        y4 = (n4/counter) / (ip/counter)
        #y = neutrons /  mean_ip
        #y3 = neutrons3 / mean_ip
        #y4 = neutrons4 / mean_ip
        #print('Neutrons:', neutrons3, neutrons4, neutrons)
        print("y's:", y, y3, y4)

        #y3_err = y3 * np.sqrt( ((neutrons3/np.sqrt(neutrons3)/neutrons3)**2 +
        #    (ip_err/mean_ip)**2 ) )
        #y3_errs.append(y3_err)
        #y4_err = y4 * np.sqrt( ((neutrons4/np.sqrt(neutrons4)/neutrons4)**2 +
        #    (ip_err/mean_ip)**2 ) )
        #y4_errs.append(y4_err)
        #y_err = y * np.sqrt( ((neutrons/np.sqrt(neutrons)/neutrons)**2 +
        #    (ip_err/mean_ip)**2 ) )
        #y_errs.append(y_err)

        y3_err = y3 / np.sqrt(counter)
        y3_errs.append(y3_err)
        y4_err = y4 / np.sqrt(counter)
        y4_errs.append(y4_err)
        y_err = y / np.sqrt(counter)
        y_errs.append(y_err)

        x_errs.append(x_err)

        #print(neutrons3, neutrons4, neutrons, mean_ip)


        peter_y.append(y)
        peter_y3.append(y3)
        peter_y4.append(y4)
        peter_x.append(mean_x)

    peter_y = np.array(peter_y)
    peter_y3 = np.array(peter_y3)
    peter_y4 = np.array(peter_y4)
    peter_x = np.array(peter_x)

    y_errs = np.array(y_errs)
    y3_errs = np.array(y3_errs)
    y4_errs = np.array(y4_errs)

    x_errs = np.array(x_errs)

    #print(peter_y3, peter_y4, peter_y)
    #print(y3_errs, y4_errs, y_errs)
    #input('well?')

    fit_range = (0.01, 0.03)
    fit_range = (min(peter_x), max(peter_x))
    chi2 = probfit.Chi2Regression(probfit.linear, peter_x, peter_y,
            y_errs)
    minu = iminuit.Minuit(chi2)
    minu.migrad()
    pars = minu.values
    p_errs = minu.errors
    print(pars, p_errs)
    #input('well?')

    if root_style == True :
        color = 'black'
    elif root_style == False :
        color = 'blue'

    #f = plt.figure()
    #ax1 = f.add_subplot(111)
    #chi2.draw(minu)
    ##chi2.draw(minu, print_par=False)
    #ax1.errorbar(peter_x, peter_y, xerr=x_errs, yerr=y_errs, fmt='o', color=color)
    ##ax1.scatter(peter_x, peter_y, color=color)
    #ax1.set_xlabel('current / (pressure * beamsize * Zeff^2)')
    #ax1.set_ylabel('n_neutrons/(current * pressure * Zeff^2)')
    ##ax1.set_xlim([0.0,5.0E5])
    ##ax1.set_ylim([0.0,3.0])
    #ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #f.savefig('TPC_peter_toushek_measurement_sim.pdf')
    #plt.show()

    #chi23 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_3,
    #        rate3_errs)
    #minu3 = iminuit.Minuit(chi23)
    #minu3.migrad()

    ##g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    #g = plt.figure()
    #bx1 = g.add_subplot(111)
    #chi23.draw(minu3, print_par=False)
    #bx1.errorbar(avg_beamsize, rate_3, yerr=rate3_errs, fmt='o', color=color)
    #bx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    #bx1.set_ylabel('Fast neutron rate in Ch. 3 (Hz)')
    #bx1.set_xlim([0.0,0.030])
    #bx1.set_ylim([0.0,0.09])

    #chi24 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_4,
    #        rate4_errs)
    #minu4 = iminuit.Minuit(chi24)
    #minu4.migrad()
    #g.savefig('TPC3_toushek_measurement.pdf')
    #plt.show()

    return peter_x, peter_y, x_errs, y_errs, chi2, minu
    '''

# Study neutron angular distributions
def neutron_study(datapath):
    tpc3_phis = []
    tpc3_thetas = []
    tpc3_energies = []
    tpc3_energies_bp = []
    tpc3_energies_notbp = []
    tpc3_sumtot = []
    tpc3_sumtot_bp = []
    tpc3_sumtot_notbp = []
    tpc3_tlengths = []
    tpc3_tlengths_bp = []
    tpc3_tlengths_notbp = []
    tpc3_thetas_beampipe = []
    tpc3_thetas_notbp = []

    tpc4_phis = []
    tpc4_thetas = []
    tpc4_energies = []
    tpc4_energies_bp = []
    tpc4_energies_notbp = []
    tpc4_sumtot_bp = []
    tpc4_sumtot_notbp = []
    tpc4_sumtot = []
    tpc4_tlengths = []
    tpc4_tlengths_bp = []
    tpc4_tlengths_notbp = []
    tpc4_thetas_beampipe = []
    tpc4_thetas_notbp = []

    beamsize = []

    runs = run_names('LER_ToushekTPC')
    #runs = run_names('HER_ToushekTPC')
    neutrons = 0

    ### Looping over many files
    for f in os.listdir(datapath):
        #if '.root' not in f: continue
        if f not in runs: continue
        ifile = datapath
        ifile += f

        rfile = TFile(ifile)
        tree = rfile.Get('tout')
        test = str(tree)
        if (test == '<ROOT.TObject object at 0x(nil)>' or tree.GetEntries() == 
                0): continue

        print(ifile)

        data = root2rec(ifile)

        counter_3 = 0
        counter_4 = 0

        max_theta = 180.
        min_theta = 0.
        for event in data:
            if ('TPC3_PID_neutrons' in event.dtype.names):
                tpc3_neutrons = event.TPC3_PID_neutrons
            else : continue

            for i in range(len(tpc3_neutrons)):
                if tpc3_neutrons[i] == 1 and event.TPC3_npoints[i] > 20 :
                    phi = event.TPC3_phi[i]
                    theta = event.TPC3_theta[i]

                    ### Check if theta and phi values are absurdly wrong
                    if abs(phi) > 720. or abs(theta) > 720.: continue

                    if event.TPC3_dEdx[i] < 0.15 :
                        #print('Is event in TPC3 a nutron?', tpc3_neutrons[i])
                        #print('dE/dx =', event.TPC3_dEdx[i])
                        #print('npoints = ', event.TPC3_npoints[i])
                        #print('cols = ', event.TPC3_hits_col)
                        #track = Hist2D(80, 1, 80, 336, 1, 336)
                        #for k in range(event.TPC3_npoints[i]):
                        #    track.Fill(event.TPC3_hits_col[k], 
                        #            event.TPC3_hits_row[k], 
                        #            event.TPC3_hits_tot[k])
                        #track.Draw('COLZ')
                        #input('well?')
                        continue

                    neutrons += 1

                    if phi < -360. : phi += 360.
                    elif phi > 360. : phi -= 360.

                    if theta < -360. : theta += 360.
                    elif theta > 360. : theta -= 360.

                    if phi < -90. :
                        phi += 180.
                        theta *= -1.
                        theta += 180.

                    elif phi > 90.:
                        phi -= 180.
                        theta *= -1.
                        theta += 180.
                        
                    if theta < 0. : theta += 180.
                    elif theta > 180. : theta -= 180.

                    tpc3_thetas.append(theta)
                    tpc3_phis.append(phi)

                    tpc3_energies.append(event.TPC3_sumQ[i])
                    tpc3_sumtot.append(event.TPC3_sumTOT[i])
                    tpc3_tlengths.append(event.TPC3_sumTOT[i]/event.TPC3_dEdx[i])

                    ### Select beampipe (+- 20 degrees)
                    if abs(phi) < 20.:
                        tpc3_thetas_beampipe.append(theta)
                        tpc3_energies_bp.append(event.TPC3_sumQ[i])
                        tpc3_sumtot_bp.append(event.TPC3_sumTOT[i])
                        tpc3_tlengths_bp.append(event.TPC3_sumTOT[i]/event.TPC3_dEdx[i])
                    elif abs(phi) > 40.:
                        tpc3_thetas_notbp.append(theta)
                        tpc3_energies_notbp.append(event.TPC3_sumQ[i])
                        tpc3_sumtot_notbp.append(event.TPC3_sumTOT[i])
                        tpc3_tlengths_notbp.append(event.TPC3_sumTOT[i]/event.TPC3_dEdx[i])


            if ('TPC4_PID_neutrons' in event.dtype.names):
                tpc4_neutrons = event.TPC4_PID_neutrons
            else : continue

            for j in range(len(tpc4_neutrons)):
                if tpc4_neutrons[j] == 1 and event.TPC4_npoints[j] > 20 :

                    phi = event.TPC4_phi[j]
                    theta = event.TPC4_theta[j]

                    ### Check if theta and phi values are absurdly wrong
                    if abs(phi) > 720. or abs(theta) > 720.: continue

                    if event.TPC4_dEdx[j] < 0.15 :
                        #print('Is event in TPC4 a nutron?', tpc4_neutrons[j])
                        #print('dE/dx =', event.TPC4_dEdx[j])
                        #print('npoints = ', event.TPC4_npoints[j])
                        #print('cols = ', event.TPC4_hits_col)
                        #track = Hist2D(80, 1, 80, 336, 1, 336)
                        #for k in range(event.TPC4_npoints[j]):
                        #    track.Fill(event.TPC4_hits_col[k], 
                        #            event.TPC4_hits_row[k], 
                        #            event.TPC4_hits_tot[k])
                        #track.Draw('COLZ')
                        #input('well?')
                        continue
                    
                    neutrons += 1

                    if phi < -360. : phi += 360.
                    elif phi > 360. : phi -= 360.

                    if theta < -360. : theta += 360.
                    elif theta > 360. : theta -= 360.

                    if phi < -90. :
                        phi += 180.
                        theta *= -1.
                        theta += 180.

                    elif phi > 90.:
                        phi -= 180.
                        theta *= -1.
                        theta += 180.

                    if theta < 0. : theta += 180.
                    elif theta > 180. : theta -= 180.


                    tpc4_phis.append(phi)
                    tpc4_thetas.append(theta)

                    tpc4_energies.append(event.TPC4_sumQ[j])
                    tpc4_sumtot.append(event.TPC4_sumTOT[j])
                    tpc4_tlengths.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])

                    #if 80. < theta and theta < 100 : print(theta)
                    ### Select beampipe (+- 20 degrees)
                    if abs(phi) < 20.:
                        tpc4_thetas_beampipe.append(theta)
                        tpc4_energies_bp.append(event.TPC4_sumQ[j])
                        tpc4_sumtot_bp.append(event.TPC4_sumTOT[j])
                        tpc4_tlengths_bp.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])
                    elif abs(phi) > 40.:
                        tpc4_thetas_notbp.append(theta)
                        tpc4_energies_notbp.append(event.TPC4_sumQ[j])
                        tpc4_sumtot_notbp.append(event.TPC4_sumTOT[j])
                        tpc4_tlengths_notbp.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])

    tpc3phi_array = np.array(tpc3_phis)
    tpc3theta_array = np.array(tpc3_thetas)
    tpc3_sumtot_array = np.array(tpc3_sumtot)
    tpc3_sumtot_array_bp = np.array(tpc3_sumtot_bp)
    tpc3_sumtot_array_notbp = np.array(tpc3_sumtot_notbp)
    tpc3_energies_array = np.array(tpc3_energies)
    tpc3_energies_array_bp = np.array(tpc3_energies_bp)
    tpc3_energies_array_notbp = np.array(tpc3_energies_notbp)
    tpc3_tlengths_array = np.array(tpc3_tlengths)
    tpc3_tlengths_array_bp = np.array(tpc3_tlengths_bp)
    tpc3_tlengths_array_notbp = np.array(tpc3_tlengths_notbp)
    tpc3theta_array_beampipe = np.array(tpc3_thetas_beampipe)
    tpc3theta_array_notbp = np.array(tpc3_thetas_notbp)

    tpc4theta_array = np.array(tpc4_thetas)
    tpc4phi_array = np.array(tpc4_phis)
    tpc4_sumtot_array = np.array(tpc4_sumtot)
    tpc4_sumtot_array_bp = np.array(tpc4_sumtot_bp)
    tpc4_sumtot_array_notbp = np.array(tpc4_sumtot_notbp)
    tpc4_energies_array = np.array(tpc4_energies)
    tpc4_energies_array_bp = np.array(tpc4_energies_bp)
    tpc4_energies_array_notbp = np.array(tpc4_energies_notbp)
    tpc4_tlengths_array = np.array(tpc4_tlengths)
    tpc4_tlengths_array_bp = np.array(tpc4_tlengths_bp)
    tpc4_tlengths_array_notbp = np.array(tpc4_tlengths_notbp)
    tpc4theta_array_beampipe = np.array(tpc4_thetas_beampipe)
    tpc4theta_array_notbp = np.array(tpc4_thetas_notbp)

    beamsize = np.array(beamsize)

    #arr3 = np.array([[0.0,0.0]]*len(tpc3_sumtot_array))
    #df3 = pd.DataFrame({'Average Beamsize': avg_beamsize,
    #                   'Average Rate'    : avg_rate, })

    #df = pd.DataFrame(
    #        {'Track Length': tpc3_tlengths_array, 'Sum TOT':tpc3_sumtot_array},
    #        index = ['Event Number'], columns = ['Track Length', 'Sum TOT'])
    #print(df)
    #input('well?')

    phi_bins = 20
    theta_bins = 18

    print('Number of neutrons:')
    print('TPC3:', len(tpc3phi_array))
    print('TPC4:', len(tpc4phi_array))
    print('Total:', neutrons)
    print('Check:', len(tpc3phi_array) + len(tpc4phi_array))
    print('\nBeampipe cut:')
    print('TPC3:', len(tpc3theta_array_beampipe))
    print('TPC4:', len(tpc4theta_array_beampipe))
    print('\nOutside Beampipe:')
    print('TPC3:', len(tpc3theta_array_notbp))
    print('TPC4:', len(tpc4theta_array_notbp))
    input('well?')

    ### Begin plotting
    if root_style == True :
        color = 'black'
        facecolor=None
    elif root_style == False :
        sns.set(color_codes=True)
        color = None

    ### Plot all figures individually
    plt.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_energies_array, color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    plt.savefig('tpc3_phi_weighted.pdf')
    plt.show()

    plt.hist(tpc3phi_array, phi_bins, range=[-100,100],
            color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Neutron Recoil $\phi$')
    plt.savefig('tpc3_phi_unweighted.pdf')
    plt.show()

    plt.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_energies_array, color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    plt.savefig('tpc4_phi_weighted.pdf')
    plt.show()

    plt.hist(tpc4phi_array, phi_bins, range=[-100,100],
            color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Neutron Recoil $\phi$')
    plt.savefig('tpc4_phi_unweighted.pdf')
    plt.show()

    plt.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Energy Weighted Neutron Recoil $\\theta$')
    plt.savefig('tpc3_theta_weighted.pdf')
    plt.show()

    plt.hist(tpc3theta_array, theta_bins, 
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Neutron Recoil $\\theta$')
    plt.savefig('tpc3_theta_unweighted.pdf')
    plt.show()

    plt.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Energy Weighted Neutron Recoil $\\theta$')
    plt.savefig('tpc4_theta_weighted.pdf')
    plt.show()

    plt.hist(tpc4theta_array, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Neutron Recoil $\\theta$')
    plt.savefig('tpc4_theta_unweighted.pdf')
    plt.show()

    plt.hist(tpc3theta_array_beampipe, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc3_theta_unweighted_bp.pdf')
    plt.show()

    plt.hist(tpc4theta_array_beampipe, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc4_theta_unweighted_bp.pdf')
    plt.show()

    plt.hist(tpc3theta_array_notbp, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc3_theta_unweighted_nbp.pdf')
    plt.show()

    plt.hist(tpc4theta_array_notbp, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc4_theta_unweighted_nbp.pdf')
    plt.show()

    plt.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    #plt.set_title('Ch. 3 Track Length vs Sum Q')
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_all.pdf')
    plt.show()

    plt.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    #plt.set_title('Ch. 4 Track Length vs Sum Q')
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_all.pdf')
    plt.show()
    
    plt.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    #plt.set_title('Ch. 3 Track Length vs Sum Q (beampipe)')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.savefig('tpc3_dedx_bp.pdf')
    plt.show()

    plt.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.xlabel('$\mu$m')
    plt.ylabel('Detected Charge (q)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_nbp.pdf')
    plt.show()

    plt.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_bp.pdf')
    plt.show()

    plt.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_nbp.pdf')
    plt.show()

    plt.scatter(tpc3_energies_array, tpc3phi_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.savefig('tpc3_evsphi_scatter.pdf')
    plt.show()

    plt.hist2d(tpc3_energies_array, tpc3phi_array, bins=(25,phi_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.colorbar()
    plt.savefig('tpc3_evsphi_heatmap.pdf')
    plt.show()

    plt.scatter(tpc3_energies_array, tpc3theta_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc3_evstheta_scatter.pdf')
    plt.show()

    plt.hist2d(tpc3_energies_array, tpc3theta_array, bins=(25,theta_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc3_evstheta_heatmap.pdf')
    plt.show()

    plt.scatter(tpc4_energies_array, tpc4phi_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.savefig('tpc4_evsphi_scatter.pdf')
    plt.show()

    plt.hist2d(tpc4_energies_array, tpc4phi_array, bins=(25,phi_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.colorbar()
    plt.savefig('tpc4_evsphi_heatmap.pdf')
    plt.show()

    plt.scatter(tpc4_energies_array, tpc4theta_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc4_evstheta_scatter.pdf')
    plt.show()

    plt.hist2d(tpc4_energies_array, tpc4theta_array, bins=(25,theta_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc4_evstheta_heatmap.pdf')
    plt.show()

    plt.scatter(tpc3phi_array, tpc3theta_array, color = color)
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylim(0.,180.0)
    plt.savefig('tpc3_thetavsphi_scatter.pdf')
    plt.show()

    plt.hist2d(tpc3phi_array, tpc3theta_array, bins=(phi_bins,theta_bins))
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc3_thetavsphi_heatmap.pdf')
    plt.show()

    plt.scatter(tpc4phi_array, tpc4theta_array, color = color)
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc4_thetavsphi_scatter.pdf')
    plt.show()

    plt.hist2d(tpc4phi_array, tpc4theta_array, bins=(phi_bins,theta_bins))
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc4_thetavsphi_heatmap.pdf')
    plt.show()

    gain1 = 30.0
    gain2 = 50.0
    w = 35.075
    tpc3_kev_array = tpc3_energies_array/(gain1 * gain2)*w*1E-3
    tpc4_kev_array = tpc4_energies_array/(gain1 * gain2)*w*1E-3
    tpc3_kev_array_bp = tpc3_energies_array_bp/(gain1 * gain2)*w*1E-3
    tpc3_kev_array_notbp = tpc3_energies_array_notbp/(gain1 * gain2)*w*1E-3
    tpc4_kev_array_bp = tpc4_energies_array_bp/(gain1 * gain2)*w*1E-3
    tpc4_kev_array_notbp = tpc4_energies_array_notbp/(gain1 * gain2)*w*1E-3

    plt.hist(tpc3_kev_array, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc3_recoil_energies.pdf')
    plt.show()

    plt.hist(tpc3_kev_array_bp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc3_recoil_energies_bp.pdf')
    plt.show()

    plt.hist(tpc3_kev_array_notbp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc3_recoil_energies_notbp.pdf')
    plt.show()

    plt.hist(tpc4_kev_array_bp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies_bp.pdf')
    plt.show()

    plt.hist(tpc4_kev_array_notbp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies_notbp.pdf')
    plt.show()

    plt.hist(tpc4_kev_array, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies.pdf')
    plt.show()

    ## Heatmap for E vs theta/phi
    #heatmap, xedges, yedges = np.histogram2d(tpc3phi_array,
    #        tpc3_energies_array, bins = (50, 50))
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #print(heatmap, xedges, yedges)

    #plt.clf()
    #plt.imshow(heatmap, extent=extent, origin ='lower')
    #print('Attempt at TPC3 E vs phi heatmap')
    #plt.show()

    #plt.scatter(tpc3_energies_array, tpc3phi_array)
    #plt.colorbar()
    #plt.show()

    ### Plot figures as subplots in canvases
    ## Plot theta and phi, weighted by energy
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc3_energies_array, color = color, histtype='step')
    #ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc4_energies_array, color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
    #        color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$')
    #ax4.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
    #        color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_weighted.pdf')

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    #        sharey='col')
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc3_energies_array, color = color, histtype='step')
    #ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc4_energies_array, color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    #tpc3_energies_array_bp, color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Beampipe cut)')
    #ax4.hist(tpc4theta_array_beampipe, theta_bins, weights = 
    #        tpc4_energies_array_bp, color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Beampipe cut)')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_weighted-beampipe.pdf')

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    #        sharey='col')
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc3_energies_array, color = color, histtype='step')
    #ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc4_energies_array, color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array_notbp, theta_bins, weights = 
    #        tpc3_energies_array_notbp, color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Outside beampipe)')
    #ax4.hist(tpc4theta_array_notbp, theta_bins, weights = 
    #        tpc4_energies_array_notbp, color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_weighted-outside_beampipe.pdf')

    #### Plot theta and phi, unweighted
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Unweighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array, theta_bins, color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$')
    #ax4.hist(tpc4theta_array, theta_bins, color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_unweighted.pdf')

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    #        sharey='col')
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Unweighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array_beampipe, theta_bins, color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Beampipe cut)')
    #ax4.hist(tpc4theta_array_beampipe, theta_bins, color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Beampipe cut)')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_unweighted-inside_beampipe.pdf')

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    #        sharey='col')
    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('Ch. 4 Uneighted Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array_notbp, theta_bins, color = color, histtype='step')
    #ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Outside beampipe)')
    #ax4.hist(tpc4theta_array_notbp, theta_bins, color = color, histtype='step')
    #ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    #ax4.set_xlabel('Degrees')
    ##f.savefig('theta_phi_unweighted-outside_beampipe.pdf', dpi=200)

    ##m, (dx1, dx2, dx3, dx4) = plt.subplots(1, 4)
    ##dx1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc3_energies_array, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx1.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc4_energies_array, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ##dx1.set_xlabel('$\phi$ ($^{\circ}$)')
    ##dx1.legend(loc='best')
    ##dx2.hist(tpc3theta_array_notbp, theta_bins, weights = 
    ##        tpc3_energies_array_notbp, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx2.hist(tpc4theta_array_notbp, theta_bins, weights = 
    ##        tpc4_energies_array_notbp, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ###ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ##dx2.set_xlabel('$\\theta$ ($^{\circ}$)')
    ##dx2.legend(loc='best')

    #### One big plot of angular information
    #m, ((dx1, dx2, dx3, dx4), (dx5, dx6, dx7, dx8)) = plt.subplots(2, 4,
    #        sharey='row')
    #        #sharex='col', sharey='row')
    #dx1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc3_energies_array, color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx1.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc4_energies_array, color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    #dx1.set_xlabel('$\phi$ ($^{\circ}$)')
    #dx1.legend(loc='best')

    #dx2.hist(tpc3theta_array, theta_bins, weights = 
    #        tpc3_energies_array, color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx2.hist(tpc4theta_array, theta_bins, weights = 
    #        tpc4_energies_array, color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    #dx2.set_xlabel('$\\theta$ ($^{\circ}$)')

    #dx3.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    #        tpc3_energies_array_bp, color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx3.hist(tpc4theta_array_beampipe, theta_bins, weights = 
    #        tpc4_energies_array_bp, color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    ##dx3.legend(loc='best')
    ##dx3.set_xlabel('$\\theta$  ($^{\circ}$)')
    #dx3.set_xlabel('$\\theta$  (Pass beampipe cut)')

    #dx4.hist(tpc3theta_array_notbp, theta_bins, weights = 
    #        tpc3_energies_array_notbp, color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx4.hist(tpc4theta_array_notbp, theta_bins, weights = 
    #        tpc4_energies_array_notbp, color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    ##dx4.legend(loc='best')
    #dx4.set_xlabel('$\\theta$  (Fail beampipe cut)')

    #dx5.hist(tpc3phi_array, phi_bins, range=[-100,100],
    #        color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx5.hist(tpc4phi_array, phi_bins, range=[-100,100],
    #        color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    ##dx5.legend(loc='best')
    #dx5.set_xlabel('$\phi$ ($^{\circ}$)')

    #dx6.hist(tpc3theta_array, theta_bins,
    #        color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx6.hist(tpc4theta_array, theta_bins, 
    #        color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    #dx6.set_xlabel('$\\theta$ ($^{\circ}$)')

    #dx7.hist(tpc3theta_array_beampipe, theta_bins, 
    #        color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx7.hist(tpc4theta_array_beampipe, theta_bins,
    #        color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    ##dx7.legend(loc='best')
    #dx7.set_xlabel('$\\theta$  (Pass beampipe cut)')

    #dx8.hist(tpc3theta_array_notbp, theta_bins, 
    #        color = color, histtype='step',
    #        label = 'Ch. 3')
    #dx8.hist(tpc4theta_array_notbp, theta_bins, 
    #        color = 'blue', histtype='step',
    #        label = 'Ch. 4')
    #dx8.set_xlabel('$\\theta$  (Fail beampipe cut)')
    ##dx8.legend(loc='best')

    ### Plot dE/dx
    g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    bx1.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    bx1.set_title('Ch. 3 Track Length vs Sum Q')
    bx1.set_ylabel('Sum Q')
    bx1.set_xlabel('$\mu$m')
    bx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    bx2.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    bx2.set_title('Ch. 4 Track Length vs Sum Q')
    bx2.set_xlabel('$\mu$m')
    bx2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    g, ((cx1, cx2), (cx3, cx4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    cx1.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    cx1.set_title('Ch. 3 Track Length vs Sum Q (beampipe)')
    #cx1.set_xlim(-5000., 35000.)
    cx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    cx1.set_ylabel('Sum Q')
    cx3.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp,
            color = color)
    cx3.set_title('Ch. 3 Track Length vs Sum Q (not beampipe)')
    cx3.set_xlabel('$\mu$m')
    cx3.set_ylabel('Sum Q')
    cx2.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    cx2.set_title('Ch. 4 Track Length vs Sum Q (beampipe)')
    cx2.set_xlim(-5000., 35000.)
    cx4.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp,
            color = color)
    cx4.set_title('Ch. 4 Track Length vs Sum Q (not beampipe)')
    cx4.set_xlabel('$\mu$m')
    cx4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #bx1 = plt.scatter(tpc3_tlengths_array, tpc3_sumtot_array)
    #cx1=seaborn.heatmap(tpc3_tlengths_sumtot_array, annot=True, fmt='f')
    

    #h, (cx1, cx2) = plt.subplots(1,2, sharex=True, sharey=True)
    #seaborn.set()
    #cx1=seaborn.heatmap(tpc3_tlengths_sumtot_array, annot=True, fmt='f')
    #cx2=seaborn.heatmap(tpc4_tlengths_array, tpc4_sumtot_array, annot=True)
    
    ### Show energy sum vs tlength against sumtot vs tlength
    #g, ((bx1, bx2), (bx3, bx4)) = plt.subplots(2, 2, sharex=True)
    #bx1.scatter(tpc3_tlengths_array, tpc3_energies_array)
    #bx1.set_title("TPC3 'Sum E' vs Track Length")
    #bx1.set_ylim(0, 100000)
    #bx3.scatter(tpc4_tlengths_array, tpc4_energies_array)
    #bx3.set_title("TPC4 'Sum E' vs Track Length")
    #bx3.set_xlabel('\t$\mu$m')
    #bx2.scatter(tpc3_tlengths_array, tpc3_sumtot_array)
    #bx2.set_title('TPC3 Sum TOT vs Track Length')
    #bx4.scatter(tpc4_tlengths_array, tpc4_sumtot_array)
    #bx4.set_title('TPC4 Sum TOT vs Track Length')
    #bx4.set_xlabel('\t$\mu$m')

    print('Number of neutrons:')
    print('TPC3:', len(tpc3phi_array))
    print('TPC4:', len(tpc4phi_array))
    print('Total:', neutrons)
    print('Check:', len(tpc3phi_array) + len(tpc4phi_array))
    print('\nBeampipe cut:')
    print('TPC3:', len(tpc3theta_array_beampipe))
    print('TPC4:', len(tpc4theta_array_beampipe))
    print('\nOutside Beampipe:')
    print('TPC3:', len(tpc3theta_array_notbp))
    print('TPC4:', len(tpc4theta_array_notbp))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()

    ''' Attempts at doing this in ternary operations using numpy built in 
    iterators.  Will hopefully do this later '''
    #for event in data:
    #    neutron_events = (np.where(event.TPC3_PID_neutrons == 1) if 
    #            ('TPC3_N_neutrons' in event.dtype.names) else 0)
    #    tpc3_phis = np.where(event.TPC3_phi for event.TPC3_PID_neutrons == 1)

        #if 'TPC3_N_neutrons' in event.dtype.names:
        #    print(len(event.TPC3_PID_neutrons), 'Event number', event.event)
        #if event.TPC3_N_neutrons[0] > 0:

def neutron_study_raw(datapath):
    tpc3_phis = []
    tpc3_thetas = []
    tpc3_energies = []
    tpc3_energies_bp = []
    tpc3_energies_notbp = []
    tpc3_sumtot = []
    tpc3_sumtot_bp = []
    tpc3_sumtot_notbp = []
    tpc3_tlengths = []
    tpc3_tlengths_bp = []
    tpc3_tlengths_notbp = []
    tpc3_thetas_beampipe = []
    tpc3_thetas_notbp = []

    tpc4_phis = []
    tpc4_thetas = []
    tpc4_energies = []
    tpc4_energies_bp = []
    tpc4_energies_notbp = []
    tpc4_sumtot_bp = []
    tpc4_sumtot_notbp = []
    tpc4_sumtot = []
    tpc4_tlengths = []
    tpc4_tlengths_bp = []
    tpc4_tlengths_notbp = []
    tpc4_thetas_beampipe = []
    tpc4_thetas_notbp = []

    beamsize = []

    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    neutrons = 0

    branches = ['neutron', 'theta', 'phi', 'de_dx', 'e_sum', 'tot_sum',
            'detnb','hitside','t_length', 'npoints', 'min_ret']
    ### Looping over many files
    for subdir, dirs, files in os.walk(datapath):
        for f in files :
            if '.DS' in f: continue
            r_file = str(subdir) + str('/') + str(f)

            #if '.root' not in f: continue

            print(r_file)

            strs = f.split('_')

            if int(strs[-2]) not in good_files : continue

            data = root2rec(r_file, branches=branches)

            if 'tpc3' in f : data.e_sum *= 1.18
            if 'tpc4' in f : data.e_sum *= 1.64

            counter_3 = 0
            counter_4 = 0

            for event in data:

                ### Neutron selections
                dQdx = (event.e_sum/event.t_length if event.hitside == 0
                        and event.t_length > 0 else 0)

                neutron = (1 if event.hitside == 0 
                        and event.min_ret == 0
                        and dQdx > 500.0
                        and event.npoints > 40 
                        and event.theta > 0
                        and event.theta < 180
                        and np.abs(event.phi) < 360
                        and event.t_length > 2000
                        else 0)

                if neutron == 1 :
                    phi = event.phi
                    theta = event.theta

                    ### Check if theta and phi values are absurdly wrong
                    if abs(phi) > 720. or abs(theta) > 360 : continue

                    if theta < 0. : theta += 180.
                    elif theta > 180. : theta -= 180.

                    neutrons += 1

                    if phi < -360. : phi += 360.
                    elif phi > 360. : phi -= 360.

                    if theta < -360. : theta += 360.
                    elif theta > 360. : theta -= 360.

                    if phi < -90. :
                        phi += 180.
                        theta *= -1.
                        theta += 180.

                    elif phi > 90.:
                        phi -= 180.
                        theta *= -1.
                        theta += 180.

                    if theta < 0. : theta = theta + 180.
                    if theta > 180. : theta = theta - 180.

                    #if theta < 0. or theta > 180. :
                        #print('Whoa!', theta, event.theta)
                        #input('well?')

                    if data.detnb[0] == 3 :
                        tpc3_thetas.append(theta)
                        tpc3_phis.append(phi)

                        tpc3_energies.append(event.e_sum)
                        tpc3_sumtot.append(event.tot_sum)
                        tpc3_tlengths.append(event.tot_sum/event.de_dx)

                    ### Select beampipe (+- 20 degrees)
                        if abs(phi) < 20.:
                            tpc3_thetas_beampipe.append(theta)
                            tpc3_energies_bp.append(event.e_sum)
                            tpc3_sumtot_bp.append(event.tot_sum)
                            tpc3_tlengths_bp.append(event.tot_sum/event.de_dx)
                        elif abs(phi) > 40.:
                            tpc3_thetas_notbp.append(theta)
                            tpc3_energies_notbp.append(event.e_sum)
                            tpc3_sumtot_notbp.append(event.tot_sum)
                            tpc3_tlengths_notbp.append(event.tot_sum/event.de_dx)


                    elif data.detnb[0] == 4 :
                        tpc4_phis.append(phi)
                        tpc4_thetas.append(theta)

                        tpc4_energies.append(event.e_sum)
                        tpc4_sumtot.append(event.tot_sum)
                        tpc4_tlengths.append(event.tot_sum/event.de_dx)

                        #if 80. < theta and theta < 100 : print(theta)
                        ### Select beampipe (+- 20 degrees)
                        if abs(phi) < 20.:
                            tpc4_thetas_beampipe.append(theta)
                            tpc4_energies_bp.append(event.e_sum)
                            tpc4_sumtot_bp.append(event.tot_sum)
                            tpc4_tlengths_bp.append(event.tot_sum/event.de_dx)
                        elif abs(phi) > 40.:
                            tpc4_thetas_notbp.append(theta)
                            tpc4_energies_notbp.append(event.e_sum)
                            tpc4_sumtot_notbp.append(event.tot_sum)
                            tpc4_tlengths_notbp.append(event.tot_sum/event.de_dx)

    tpc3phi_array = np.array(tpc3_phis)
    tpc3theta_array = np.array(tpc3_thetas)
    tpc3_sumtot_array = np.array(tpc3_sumtot)
    tpc3_sumtot_array_bp = np.array(tpc3_sumtot_bp)
    tpc3_sumtot_array_notbp = np.array(tpc3_sumtot_notbp)
    tpc3_energies_array = np.array(tpc3_energies)
    tpc3_energies_array_bp = np.array(tpc3_energies_bp)
    tpc3_energies_array_notbp = np.array(tpc3_energies_notbp)
    tpc3_tlengths_array = np.array(tpc3_tlengths)
    tpc3_tlengths_array_bp = np.array(tpc3_tlengths_bp)
    tpc3_tlengths_array_notbp = np.array(tpc3_tlengths_notbp)
    tpc3theta_array_beampipe = np.array(tpc3_thetas_beampipe)
    tpc3theta_array_notbp = np.array(tpc3_thetas_notbp)

    #hist = np.histogram(tpc3phi_array, bins=18)
    #phis=np.arange(-90,90,10)
    #plt.hist(phis, bins=18, weights=hist[0], histtype='step', color='black')
    #plt.bar(phis, hist[0], align='center')
    #plt.show()

    tpc4theta_array = np.array(tpc4_thetas)
    tpc4phi_array = np.array(tpc4_phis)
    tpc4_sumtot_array = np.array(tpc4_sumtot)
    tpc4_sumtot_array_bp = np.array(tpc4_sumtot_bp)
    tpc4_sumtot_array_notbp = np.array(tpc4_sumtot_notbp)

    tpc4_energies_array = np.array(tpc4_energies)
    #tpc4_energies_array *= 1.43

    tpc4_energies_array_bp = np.array(tpc4_energies_bp)
    #tpc4_energies_array_bp *= 1.43

    tpc4_energies_array_notbp = np.array(tpc4_energies_notbp)
    #tpc4_energies_array_notbp *= 1.43

    tpc4_tlengths_array = np.array(tpc4_tlengths)
    tpc4_tlengths_array_bp = np.array(tpc4_tlengths_bp)
    tpc4_tlengths_array_notbp = np.array(tpc4_tlengths_notbp)
    tpc4theta_array_beampipe = np.array(tpc4_thetas_beampipe)
    tpc4theta_array_notbp = np.array(tpc4_thetas_notbp)

    beamsize = np.array(beamsize)

    #arr3 = np.array([[0.0,0.0]]*len(tpc3_sumtot_array))
    #df3 = pd.DataFrame({'Average Beamsize': avg_beamsize,
    #                   'Average Rate'    : avg_rate, })

    #df = pd.DataFrame(
    #        {'Track Length': tpc3_tlengths_array, 'Sum TOT':tpc3_sumtot_array},
    #        index = ['Event Number'], columns = ['Track Length', 'Sum TOT'])
    #print(df)
    #input('well?')

    phi_bins = 18
    theta_bins = 9

    #(n, bins, patches) = plt.hist(tpc3phi_array, bins=phi_bins)
    #print(n, bins, patches)
    #input('well?')

    ### Begin plotting
    if root_style == True :
        color = 'black'
        facecolor=None
    elif root_style == False :
        sns.set(color_codes=True)
        color = None

    ### Plot all figures individually
    #plt.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc3_energies_array, color = color, histtype='step')
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    #plt.xlim(-100,100)
    #plt.savefig('tpc3_phi_weighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc3phi_array, phi_bins, range=[-100,100],
    #        color = color, histtype='step')
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC3 Neutron Recoil $\phi$')
    #plt.xlim(-100, 100)
    ##plt.savefig('tpc3_phi_unweighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    #        tpc4_energies_array, color = color, histtype='step')
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    #plt.xlim(-100, 100)
    #plt.savefig('tpc4_phi_weighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc4phi_array, phi_bins, range=[-100,100],
    #        color = color, histtype='step')
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC4 Neutron Recoil $\phi$')
    #plt.xlim(-100, 100)
    #plt.savefig('tpc4_phi_unweighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
    #        color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC3 Energy Weighted Neutron Recoil $\\theta$')
    ##plt.xlim(0,180)
    #plt.savefig('tpc3_theta_weighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc3theta_array, theta_bins, 
    #        color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC3 Neutron Recoil $\\theta$')
    ##plt.xlim(0,180)
    #plt.savefig('tpc3_theta_unweighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
    #        color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC4 Energy Weighted Neutron Recoil $\\theta$')
    #plt.xlim(0,180)
    #plt.savefig('tpc4_theta_weighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc4theta_array, theta_bins, color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.title('TPC4 Neutron Recoil $\\theta$')
    ##plt.xlim(0,180)
    #plt.savefig('tpc4_theta_unweighted_raw.pdf')
    #plt.show()

    #plt.hist(tpc3theta_array_beampipe, theta_bins, color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.xlim(0,180)
    #plt.savefig('tpc3_theta_unweighted_bp_raw.pdf')
    #plt.show()

    #plt.hist(tpc4theta_array_beampipe, theta_bins, color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0,180)
    #plt.savefig('tpc4_theta_unweighted_bp_raw.pdf')
    #plt.show()

    #plt.hist(tpc3theta_array_notbp, theta_bins, color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0,180)
    #plt.savefig('tpc3_theta_unweighted_nbp_raw.pdf')
    #plt.show()

    #plt.hist(tpc4theta_array_notbp, theta_bins, color = color, histtype='step')
    #plt.xlabel('$\\theta$ ($^{\circ}$)')
    #plt.ylabel('Events per bin')
    ##plt.xlim(0,180)
    #plt.savefig('tpc4_theta_unweighted_nbp_raw.pdf')
    #plt.show()

    #plt.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    ##plt.set_title('Ch. 3 Track Length vs Sum Q')
    #plt.ylabel('Detected Charge (q)')
    #plt.xlabel('$\mu$m')
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.savefig('tpc3_dedx_all_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    ##plt.set_title('Ch. 4 Track Length vs Sum Q')
    #plt.ylabel('Detected Charge (q)')
    #plt.xlabel('$\mu$m')
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.savefig('tpc4_dedx_all_raw.pdf')
    #plt.show()
    #
    #plt.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    ##plt.set_title('Ch. 3 Track Length vs Sum Q (beampipe)')
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ylabel('Detected Charge (q)')
    #plt.xlabel('$\mu$m')
    #plt.savefig('tpc3_dedx_bp_raw.pdf')
    #plt.show()

    #plt.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp, color = color)
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.xlabel('$\mu$m')
    #plt.ylabel('Detected Charge (q)')
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.savefig('tpc3_dedx_nbp_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.ylabel('Detected Charge (q)')
    #plt.xlabel('$\mu$m')
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.savefig('tpc4_dedx_bp_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp, color = color)
    #plt.xlim(-5000., 35000.)
    #plt.ylim(-1E7, 7E7)
    #plt.ylabel('Detected Charge (q)')
    #plt.xlabel('$\mu$m')
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.savefig('tpc4_dedx_nbp_raw.pdf')
    #plt.show()

    #plt.scatter(tpc3_energies_array, tpc3phi_array, color = color)
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\phi$ ($^{\circ}$)')
    #plt.ylim(-100.0,100.0)
    #plt.savefig('tpc3_evsphi_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc3_energies_array, tpc3phi_array, bins=(25,phi_bins))
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\phi$ ($^{\circ}$)')
    #plt.ylim(-100.0,100.0)
    #plt.colorbar()
    #plt.savefig('tpc3_evsphi_heatmap_raw.pdf')
    #plt.show()

    #plt.scatter(tpc3_energies_array, tpc3theta_array, color = color)
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.savefig('tpc3_evstheta_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc3_energies_array, tpc3theta_array, bins=(25,theta_bins))
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.colorbar()
    #plt.savefig('tpc3_evstheta_heatmap_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4_energies_array, tpc4phi_array, color = color)
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\phi$ ($^{\circ}$)')
    #plt.ylim(-100.0,100.0)
    #plt.savefig('tpc4_evsphi_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc4_energies_array, tpc4phi_array, bins=(25,phi_bins))
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\phi$ ($^{\circ}$)')
    #plt.ylim(-100.0,100.0)
    #plt.colorbar()
    #plt.savefig('tpc4_evsphi_heatmap_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4_energies_array, tpc4theta_array, color = color)
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.savefig('tpc4_evstheta_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc4_energies_array, tpc4theta_array, bins=(25,theta_bins))
    #plt.xlabel('Detected Charge (q)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.colorbar()
    #plt.savefig('tpc4_evstheta_heatmap_raw.pdf')
    #plt.show()

    #plt.scatter(tpc3phi_array, tpc3theta_array, color = color)
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.xlim(-100.0,100.0)
    #plt.ylim(0.,180.0)
    #plt.savefig('tpc3_thetavsphi_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc3phi_array, tpc3theta_array, bins=(phi_bins,theta_bins))
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.xlim(-100.0,100.0)
    #plt.ylim(0.,180.0)
    #plt.colorbar()
    #plt.savefig('tpc3_thetavsphi_heatmap_raw.pdf')
    #plt.show()

    #plt.scatter(tpc4phi_array, tpc4theta_array, color = color)
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.xlim(-100.0,100.0)
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.savefig('tpc4_thetavsphi_scatter_raw.pdf')
    #plt.show()

    #plt.hist2d(tpc4phi_array, tpc4theta_array, bins=(phi_bins,theta_bins))
    #plt.xlabel('$\phi$ ($^{\circ}$)')
    #plt.xlim(-100.0,100.0)
    #plt.ylabel('$\\theta$ ($^{\circ}$)')
    #plt.ylim(0.,180.0)
    #plt.colorbar()
    #plt.savefig('tpc4_thetavsphi_heatmap_raw.pdf')
    #plt.show()

    #gain1 = 30.0
    #gain2 = 50.0
    #w = 35.075
    #tpc3_kev_array = tpc3_energies_array/(gain1 * gain2)*w*1E-3
    #tpc4_kev_array = tpc4_energies_array/(gain1 * gain2)*w*1E-3
    #tpc3_kev_array_bp = tpc3_energies_array_bp/(gain1 * gain2)*w*1E-3
    #tpc3_kev_array_notbp = tpc3_energies_array_notbp/(gain1 * gain2)*w*1E-3
    #tpc4_kev_array_bp = tpc4_energies_array_bp/(gain1 * gain2)*w*1E-3
    #tpc4_kev_array_notbp = tpc4_energies_array_notbp/(gain1 * gain2)*w*1E-3

    #plt.hist(tpc3_kev_array, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc3_recoil_energies_raw.pdf')
    #plt.show()

    #plt.hist(tpc3_kev_array_bp, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc3_recoil_energies_bp_raw.pdf')
    #plt.show()

    #plt.hist(tpc3_kev_array_notbp, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc3_recoil_energies_notbp_raw.pdf')
    #plt.show()

    #plt.hist(tpc4_kev_array_bp, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc4_recoil_energies_bp_raw.pdf')
    #plt.show()

    #plt.hist(tpc4_kev_array_notbp, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc4_recoil_energies_notbp_raw.pdf')
    #plt.show()

    #plt.hist(tpc4_kev_array, bins=25, color=color, histtype='step')
    #plt.xlabel('Detected Energy (keV)')
    #plt.ylabel('Events per bin')
    #plt.xlim(0, 1600)
    #plt.yscale('log')
    #plt.ylim(0,4E2)
    #plt.savefig('tpc4_recoil_energies_raw.pdf')
    #plt.show()

    ### Heatmap for E vs theta/phi
    ##heatmap, xedges, yedges = np.histogram2d(tpc3phi_array,
    ##        tpc3_energies_array, bins = (50, 50))
    ##extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ##print(heatmap, xedges, yedges)

    ##plt.clf()
    ##plt.imshow(heatmap, extent=extent, origin ='lower')
    ##print('Attempt at TPC3 E vs phi heatmap')
    ##plt.show()

    ##plt.scatter(tpc3_energies_array, tpc3phi_array)
    ##plt.colorbar()
    ##plt.show()

    #### Plot figures as subplots in canvases
    ### Plot theta and phi, weighted by energy
    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc3_energies_array, color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc4_energies_array, color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
    ##        color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$')
    ##ax4.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
    ##        color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_weighted.pdf')

    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    ##        sharey='col')
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc3_energies_array, color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc4_energies_array, color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    ##tpc3_energies_array_bp, color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Beampipe cut)')
    ##ax4.hist(tpc4theta_array_beampipe, theta_bins, weights = 
    ##        tpc4_energies_array_bp, color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Beampipe cut)')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_weighted-beampipe.pdf')

    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    ##        sharey='col')
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc3_energies_array, color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Energy Weighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc4_energies_array, color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Energy Weighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array_notbp, theta_bins, weights = 
    ##        tpc3_energies_array_notbp, color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Outside beampipe)')
    ##ax4.hist(tpc4theta_array_notbp, theta_bins, weights = 
    ##        tpc4_energies_array_notbp, color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_weighted-outside_beampipe.pdf')

    ##### Plot theta and phi, unweighted
    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Unweighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array, theta_bins, color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$')
    ##ax4.hist(tpc4theta_array, theta_bins, color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_unweighted.pdf')

    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    ##        sharey='col')
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Unweighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array_beampipe, theta_bins, color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Beampipe cut)')
    ##ax4.hist(tpc4theta_array_beampipe, theta_bins, color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Beampipe cut)')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_unweighted-inside_beampipe.pdf')

    ##f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
    ##        sharey='col')
    ##ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax1.set_title('Ch. 3 Unweighted Neutron Recoil $\phi$')
    ##ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], color = color, histtype='step')
    ##ax3.set_xlabel('Degrees')
    ##ax3.set_title('Ch. 4 Uneighted Neutron Recoil $\phi$')
    ##ax2.hist(tpc3theta_array_notbp, theta_bins, color = color, histtype='step')
    ##ax2.set_title('Ch. 3 Neutron Recoil $\\theta$ (Outside beampipe)')
    ##ax4.hist(tpc4theta_array_notbp, theta_bins, color = color, histtype='step')
    ##ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ##ax4.set_xlabel('Degrees')
    ###f.savefig('theta_phi_unweighted-outside_beampipe.pdf', dpi=200)

    ###m, (dx1, dx2, dx3, dx4) = plt.subplots(1, 4)
    ###dx1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ###        tpc3_energies_array, color = color, histtype='step',
    ###        label = 'Ch. 3')
    ###dx1.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ###        tpc4_energies_array, color = 'blue', histtype='step',
    ###        label = 'Ch. 4')
    ###dx1.set_xlabel('$\phi$ ($^{\circ}$)')
    ###dx1.legend(loc='best')
    ###dx2.hist(tpc3theta_array_notbp, theta_bins, weights = 
    ###        tpc3_energies_array_notbp, color = color, histtype='step',
    ###        label = 'Ch. 3')
    ###dx2.hist(tpc4theta_array_notbp, theta_bins, weights = 
    ###        tpc4_energies_array_notbp, color = 'blue', histtype='step',
    ###        label = 'Ch. 4')
    ####ax4.set_title('Ch. 4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ###dx2.set_xlabel('$\\theta$ ($^{\circ}$)')
    ###dx2.legend(loc='best')

    ##### One big plot of angular information
    ##m, ((dx1, dx2, dx3, dx4), (dx5, dx6, dx7, dx8)) = plt.subplots(2, 4,
    ##        sharey='row')
    ##        #sharex='col', sharey='row')
    ##dx1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc3_energies_array, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx1.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
    ##        tpc4_energies_array, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ##dx1.set_xlabel('$\phi$ ($^{\circ}$)')
    ##dx1.legend(loc='best')

    ##dx2.hist(tpc3theta_array, theta_bins, weights = 
    ##        tpc3_energies_array, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx2.hist(tpc4theta_array, theta_bins, weights = 
    ##        tpc4_energies_array, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ##dx2.set_xlabel('$\\theta$ ($^{\circ}$)')

    ##dx3.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    ##        tpc3_energies_array_bp, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx3.hist(tpc4theta_array_beampipe, theta_bins, weights = 
    ##        tpc4_energies_array_bp, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ###dx3.legend(loc='best')
    ###dx3.set_xlabel('$\\theta$  ($^{\circ}$)')
    ##dx3.set_xlabel('$\\theta$  (Pass beampipe cut)')

    ##dx4.hist(tpc3theta_array_notbp, theta_bins, weights = 
    ##        tpc3_energies_array_notbp, color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx4.hist(tpc4theta_array_notbp, theta_bins, weights = 
    ##        tpc4_energies_array_notbp, color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ###dx4.legend(loc='best')
    ##dx4.set_xlabel('$\\theta$  (Fail beampipe cut)')

    ##dx5.hist(tpc3phi_array, phi_bins, range=[-100,100],
    ##        color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx5.hist(tpc4phi_array, phi_bins, range=[-100,100],
    ##        color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ###dx5.legend(loc='best')
    ##dx5.set_xlabel('$\phi$ ($^{\circ}$)')

    ##dx6.hist(tpc3theta_array, theta_bins,
    ##        color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx6.hist(tpc4theta_array, theta_bins, 
    ##        color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ##dx6.set_xlabel('$\\theta$ ($^{\circ}$)')

    ##dx7.hist(tpc3theta_array_beampipe, theta_bins, 
    ##        color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx7.hist(tpc4theta_array_beampipe, theta_bins,
    ##        color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ###dx7.legend(loc='best')
    ##dx7.set_xlabel('$\\theta$  (Pass beampipe cut)')

    ##dx8.hist(tpc3theta_array_notbp, theta_bins, 
    ##        color = color, histtype='step',
    ##        label = 'Ch. 3')
    ##dx8.hist(tpc4theta_array_notbp, theta_bins, 
    ##        color = 'blue', histtype='step',
    ##        label = 'Ch. 4')
    ##dx8.set_xlabel('$\\theta$  (Fail beampipe cut)')
    ###dx8.legend(loc='best')

    #### Plot dE/dx
    #g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    #bx1.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    #bx1.set_title('Ch. 3 Track Length vs Sum Q')
    #bx1.set_ylabel('Sum Q')
    #bx1.set_xlabel('$\mu$m')
    #bx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #bx2.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    #bx2.set_title('Ch. 4 Track Length vs Sum Q')
    #bx2.set_xlabel('$\mu$m')
    #bx2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #
    #g, ((cx1, cx2), (cx3, cx4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    #cx1.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    #cx1.set_title('Ch. 3 Track Length vs Sum Q (beampipe)')
    ##cx1.set_xlim(-5000., 35000.)
    #cx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #cx1.set_ylabel('Sum Q')
    #cx3.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp,
    #        color = color)
    #cx3.set_title('Ch. 3 Track Length vs Sum Q (not beampipe)')
    #cx3.set_xlabel('$\mu$m')
    #cx3.set_ylabel('Sum Q')
    #cx2.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    #cx2.set_title('Ch. 4 Track Length vs Sum Q (beampipe)')
    #cx2.set_xlim(-5000., 35000.)
    #cx4.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp,
    #        color = color)
    #cx4.set_title('Ch. 4 Track Length vs Sum Q (not beampipe)')
    #cx4.set_xlabel('$\mu$m')
    #cx4.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ##bx1 = plt.scatter(tpc3_tlengths_array, tpc3_sumtot_array)
    ##cx1=seaborn.heatmap(tpc3_tlengths_sumtot_array, annot=True, fmt='f')
    #

    ##h, (cx1, cx2) = plt.subplots(1,2, sharex=True, sharey=True)
    ##seaborn.set()
    ##cx1=seaborn.heatmap(tpc3_tlengths_sumtot_array, annot=True, fmt='f')
    ##cx2=seaborn.heatmap(tpc4_tlengths_array, tpc4_sumtot_array, annot=True)
    #
    #### Show energy sum vs tlength against sumtot vs tlength
    ##g, ((bx1, bx2), (bx3, bx4)) = plt.subplots(2, 2, sharex=True)
    ##bx1.scatter(tpc3_tlengths_array, tpc3_energies_array)
    ##bx1.set_title("TPC3 'Sum E' vs Track Length")
    ##bx1.set_ylim(0, 100000)
    ##bx3.scatter(tpc4_tlengths_array, tpc4_energies_array)
    ##bx3.set_title("TPC4 'Sum E' vs Track Length")
    ##bx3.set_xlabel('\t$\mu$m')
    ##bx2.scatter(tpc3_tlengths_array, tpc3_sumtot_array)
    ##bx2.set_title('TPC3 Sum TOT vs Track Length')
    ##bx4.scatter(tpc4_tlengths_array, tpc4_sumtot_array)
    ##bx4.set_title('TPC4 Sum TOT vs Track Length')
    ##bx4.set_xlabel('\t$\mu$m')

    #print('Number of neutrons:')
    #print('TPC3:', len(tpc3phi_array))
    #print('TPC4:', len(tpc4phi_array))
    #print('Total:', neutrons)
    #print('Check:', len(tpc3phi_array) + len(tpc4phi_array))
    #print('\nBeampipe cut:')
    #print('TPC3:', len(tpc3theta_array_beampipe))
    #print('TPC4:', len(tpc4theta_array_beampipe))
    #print('\nOutside Beampipe:')
    #print('TPC3:', len(tpc3theta_array_notbp))
    #print('TPC4:', len(tpc4theta_array_notbp))
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.show()

    #''' Attempts at doing this in ternary operations using numpy built in 
    #iterators.  Will hopefully do this later '''
    ##for event in data:
    ##    neutron_events = (np.where(event.TPC3_PID_neutrons == 1) if 
    ##            ('TPC3_N_neutrons' in event.dtype.names) else 0)
    ##    tpc3_phis = np.where(event.TPC3_phi for event.TPC3_PID_neutrons == 1)

    #    #if 'TPC3_N_neutrons' in event.dtype.names:
    #    #    print(len(event.TPC3_PID_neutrons), 'Event number', event.event)
    #    #if event.TPC3_N_neutrons[0] > 0:

    return (tpc3theta_array, tpc4theta_array, tpc3phi_array, tpc4phi_array,
            tpc3theta_array_beampipe, tpc3theta_array_notbp,
            tpc4theta_array_beampipe, tpc4theta_array_notbp)

def neutron_study_sim(simpath):
    sumQ = []
    tlengths = []
    thetas = []
    phis = []
    pdg = []
    detnbs = []
    bcids = []
    tots = []
    hitsides = []
    min_rets = []
    npoints = []
    touschek = []
    beam_gas = []

    truth_thetas = []
    truth_phis = []

    truth_file = '/Users/BEASTzilla/BEAST/sim/v5.2/mc_beast_run_2016-02-09.root'

    for f in os.listdir(simpath) :
        if '.root' not in f or 'HER' in f : continue
        infile = simpath + f
        print(infile)
        data = root2rec(infile)

        sumQ = np.concatenate([
                               sumQ,
                               data.e_sum
                               ])
        tlengths = np.concatenate([
                                   tlengths,
                                   data.t_length
                                   ])
        thetas = np.concatenate([
                                 thetas,
                                 data.theta
                                 ])
        phis = np.concatenate([
                               phis, 
                               data.phi
                               ])
        pdg = np.concatenate([
                              pdg, 
                              data.pdg
                              ])
        detnbs = np.concatenate([
                                 detnbs,
                                 data.detnb
                                 ])
        bcids = np.concatenate([
                                bcids,
                                data.bcid
                                ])
        tots = np.concatenate([
                                tots,
                                data.tot
                                ])
        hitsides = np.concatenate([
                                   hitsides,
                                   data.hitside
                                   ])
        min_rets = np.concatenate([
                                   min_rets,
                                   data.min_ret
                                   ])
        npoints = np.concatenate([
                                   npoints,
                                   data.npoints
                                   ])
        if 'Touschek' in f :
            touschek = np.concatenate([touschek, np.ones(len(data)) ])
            beam_gas = np.concatenate([beam_gas, np.zeros(len(data)) ])
        else :
            beam_gas = np.concatenate([beam_gas, np.ones(len(data)) ])
            touschek = np.concatenate([touschek, np.zeros(len(data)) ])


    pulse_widths = []

    for i in range(len(bcids)) :
        lowest_bcid = np.min(bcids[i])
        pulses = bcids[i] + tots[i]
        largest_pulse = np.max(pulses)
        largest_pulse_element = np.where(pulses==largest_pulse)
        
        evt_pulse_width = largest_pulse - lowest_bcid
        pulse_widths.append(evt_pulse_width)
        
    pulse_widths = np.array(pulse_widths)

    dQdx = sumQ/tlengths
    dQdx[tlengths == 0] = 0 

    sels = (
               (pulse_widths > 3)
              & (hitsides == 0)
              & (min_rets == 0)
              & (dQdx > 500)
              & (npoints > 40)
              & (thetas > 0)
              & (thetas < 180)
              & (np.abs(phis) < 360)
              & (tlengths > 2000)
            )

    ch3_sels = (
               (pulse_widths > 3)
              & (detnbs == 2)
              & (hitsides == 0)
              & (min_rets == 0)
              & (dQdx > 500)
              & (npoints > 40)
              & (thetas > 0)
              & (thetas < 180)
              & (np.abs(phis) < 360)
              #& (tlengths > 2000)
            )

    ch4_sels = (
               (pulse_widths > 3)
              & (detnbs == 3)
              & (hitsides == 0)
              & (min_rets == 0)
              & (dQdx > 500)
              & (npoints > 40)
              & (thetas > 0)
              & (thetas < 180)
              & (np.abs(phis) < 360)
              #& (tlengths > 2000)
            )

    #### Correct for theta and (phi) outside of 180 (360) degrees
    folded_phis = phis
    folded_thetas = thetas

    folded_thetas[( (folded_phis < -90) )] *= -1
    folded_thetas[( (folded_phis < -90) )] += 180

    folded_thetas[( (folded_phis > 90) )] *= -1
    folded_thetas[( (folded_phis > 90) )] += 180

    folded_phis[( (folded_phis < -90) )] += 180
    folded_phis[( (folded_phis > 90) )] -= 180


    ch3_folded_thetas = folded_thetas[ch3_sels]    
    ch4_folded_thetas = folded_thetas[ch4_sels]    

    ch3_folded_phis = folded_phis[ch3_sels]    
    ch4_folded_phis = folded_phis[ch4_sels]    

    ch3_bp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis < 20)]
    ch4_bp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis < 20)]

    ch3_nbp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis > 40)]
    ch4_nbp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis > 40)]

    ch3_nbp_thetas = ch3_folded_thetas[np.abs(ch3_folded_phis > 40)]
    ch4_nbp_thetas = ch4_folded_thetas[np.abs(ch4_folded_phis > 40)]

    ch3_bp_thetas = np.histogram(
            ch3_bp_thetas,
            bins=9,
            range=[0,180] )
    ch3_nbp_thetas = np.histogram(
            ch3_nbp_thetas,
            bins=9,
            range=[0,180] )

    ch4_bp_thetas = np.histogram(
            ch4_bp_thetas,
            bins=9,
            range=[0,180] )
    ch4_nbp_thetas = np.histogram(
            ch4_nbp_thetas,
            bins=9,
            range=[0,180] )

    ch3_thetas = np.histogram(
            ch3_folded_thetas,
            bins=18,
            range=[0,180] )
    ch4_thetas = np.histogram(
            ch4_folded_thetas,
            bins=18,
            range=[0,180] )

    ch3_phis = np.histogram(
            ch3_folded_phis,
            bins=18,
            range=[-90,90] )
    ch4_phis = np.histogram(
            ch4_folded_phis,
            bins=18,
            range=[-90,90] )

    '''
    print(ch3_phis[0])
    print(ch4_phis[0])
    print(ch3_thetas[0])
    print(ch4_thetas[0])
    print(ch3_bp_thetas[0])
    print(ch4_bp_thetas[0])
    print(ch3_nbp_thetas[0])
    print(ch4_nbp_thetas[0])
    '''
    
    ch3_phis_touschek = folded_phis[( (ch3_sels) & (touschek==1) )]
    ch3_phis_beamgas = folded_phis[( (ch3_sels) & (beam_gas==1) )]
    ch3_phis_bks = [ch3_phis_touschek, ch3_phis_beamgas]

    ch3_thetas_touschek = folded_thetas[( (ch3_sels) & (touschek==1) )]
    ch3_thetas_beamgas = folded_thetas[( (ch3_sels) & (beam_gas==1) )]
    ch3_thetas_bks = [ch3_thetas_touschek, ch3_thetas_beamgas]

    ch3_thetas_bpdirect_touschek = folded_thetas[( (ch3_sels)
                                                   & (touschek==1)
                                                   & (np.abs(folded_phis) <
                                                       20)
                                                   )]
    ch3_thetas_bpdirect_beamgas = folded_thetas[( (ch3_sels)
                                                   & (beam_gas==1)
                                                   & (np.abs(folded_phis) <
                                                       20)
                                                   )]

    ch3_thetas_bpdirect = [ch3_thetas_bpdirect_touschek,
            ch3_thetas_bpdirect_beamgas]

    ch3_thetas_nbpdirect_touschek = folded_thetas[( (ch3_sels)
                                                   & (touschek==1)
                                                   & (np.abs(folded_phis) >
                                                       40)
                                                   )]
    ch3_thetas_nbpdirect_beamgas = folded_thetas[( (ch3_sels)
                                                   & (beam_gas==1)
                                                   & (np.abs(folded_phis) >
                                                       40)
                                                   )]
    ch3_thetas_nbpdirect = [ch3_thetas_nbpdirect_touschek,
            ch3_thetas_nbpdirect_beamgas]
        

    ch4_phis_touschek = folded_phis[( (ch4_sels) & (touschek==1) )]
    ch4_phis_beamgas = folded_phis[( (ch4_sels) & (beam_gas==1) )]
    ch4_phis_bks = [ch4_phis_touschek, ch4_phis_beamgas]

    ch4_thetas_touschek = folded_thetas[( (ch4_sels) & (touschek==1) )]
    ch4_thetas_beamgas = folded_thetas[( (ch4_sels) & (beam_gas==1) )]
    ch4_thetas_bks = [ch4_thetas_touschek, ch4_thetas_beamgas]

    ch4_thetas_bpdirect_touschek = folded_thetas[( (ch4_sels)
                                                   & (touschek==1)
                                                   & (np.abs(folded_phis) <
                                                       20)
                                                   )]
    ch4_thetas_bpdirect_beamgas = folded_thetas[( (ch4_sels)
                                                   & (beam_gas==1)
                                                   & (np.abs(folded_phis) <
                                                       20)
                                                   )]

    ch4_thetas_bpdirect = [ch4_thetas_bpdirect_touschek,
            ch4_thetas_bpdirect_beamgas]

    ch4_thetas_nbpdirect_touschek = folded_thetas[( (ch4_sels)
                                                   & (touschek==1)
                                                   & (np.abs(folded_phis) >
                                                       40)
                                                   )]
    ch4_thetas_nbpdirect_beamgas = folded_thetas[( (ch4_sels)
                                                   & (beam_gas==1)
                                                   & (np.abs(folded_phis) >
                                                       40)
                                                   )]
    ch4_thetas_nbpdirect = [ch4_thetas_nbpdirect_touschek,
            ch4_thetas_nbpdirect_beamgas]



    return (ch3_thetas[0], ch4_thetas[0], ch3_phis[0], ch4_phis[0],
            ch3_bp_thetas[0], ch3_nbp_thetas[0], ch4_bp_thetas[0],
            ch4_nbp_thetas[0], ch3_thetas_bks, ch4_thetas_bks, 
            ch3_phis_bks, ch4_phis_bks, ch3_thetas_bpdirect,
            ch3_thetas_nbpdirect, ch4_thetas_bpdirect, ch4_thetas_nbpdirect)

def energy_eff_study(gain_path):
    #runs = run_names('LER_ToushekTPC')

    all_e = []
    n_e = []
    topa_e = []
    bota_e = []
    p_e = []
    x_e = []
    others_e = []
    n_3 = []
    n_4 = []
    #for f in os.listdir(datapath):
    #    if f not in runs: continue

    branches = ['hitside',
                'de_dx',
                'neutron',
                'npoints',
                ]


    for subdir, dirs, files in os.walk(gain_path):
        for f in files:
            r_file = str(subdir) + str('/') + str(f)

            data = root2rec(r_file)

            #ifile = datapath
            #ifile += f

            #rfile = TFile(ifile)
            #tree = rfile.Get('tout')
            #test = str(tree)
            #if (test == '<ROOT.TObject object at 0x(nil)>' or tree.GetEntries() == 
            #        0): continue

            print(r_file)

            #all_e = []
            #n_e = []

            for event in data:
                
                if event.hitside == 0 : all_e.append(event.e_sum)
                #if event.neutron == 1 : n_e.append(event.e_sum)
                if (event.neutron == 1 and event.detnb == 3
                        and event.e_sum * 1.18 > 500.0 and
                        event.npoints > 40) :
                    n_3.append(event.e_sum * 1.18)
                    n_e.append(event.e_sum * 1.18)
                if (event.neutron == 1 and event.detnb == 4
                        and event.e_sum * 1.18 > 500.0 and
                        event.npoints > 40) :
                    n_4.append(event.e_sum * 1.64)
                    n_e.append(event.e_sum * 1.64)
                #if event.top_alpha == 1 : topa_e.append(event.e_sum)
                #if event.bottom_alpha == 1 : bota_e.append(event.e_sum)
                #if event.proton == 1 : p_e.append(event.e_sum)






    all_e = np.array(all_e)
    n_e = np.array(n_e)

    n_3 = np.array(n_3)
    n_4 = np.array(n_4)
    
    h, (cx1) = plt.subplots(1, 1)
    cx1.hist(n_3, bins = 100, color='black', histtype='step')
        
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    #h.savefig('tpc3_neutron_energies.eps') 
    plt.show()

    h, (cx1) = plt.subplots(1, 1)
    cx1.hist(n_4, bins = 100, color='black', histtype='step')
        
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    #h.savefig('tpc4_neutron_energies.eps') 
    plt.show()

    input('done plotting')
    max_e = np.max(all_e)
    hist_all = Hist(100, 0., max_e)
    hist_all.fill_array(all_e)
    np_hist_all = np.histogram(all_e, bins=500, range=[0,max_e])

    max_ne = np.max(n_e)

    np_hist_n3 = np.histogram(n_3, bins=500, range=[0,max_ne])
    np_hist_n4 = np.histogram(n_4, bins=500, range=[0,max_ne])

    np_hist_n = np.histogram(n_e, bins=500, range=[0,max_e])

    all_e = np.array(all_e)
    divided_e = np.array([0.]*500)
    divided_e3 = np.array([0.]*500)
    divided_e4 = np.array([0.]*500)
    divided_bins = np.array([0.]*500)
    div_errs = np.array([0.]*500)

    n3_bins = np.array([0.]*500)
    n4_bins = np.array([0.]*500)

    t_errs = np.array([0.]*500)
    b_errs = np.array([0.]*500)
    n_errs = np.array([0.]*500)
    p_errs = np.array([0.]*500)
    x_errs = np.array([0.]*500)
    o_errs = np.array([0.]*500)

    n3_errs = np.array([0.]*500)
    n4_errs = np.array([0.]*500)
    for i in range(500):
        e_a = np_hist_all[0][i]
        n_a = np_hist_n[0][i]
        divided_e[i] = n_a/e_a if e_a != 0 else 0

        e_3 = np_hist_all[0][i]
        divided_e3[i] = n_3/e_a if e_a != 0 else 0
        divided_bins[i] = (np_hist_n[1][i]+np_hist_n[1][i+1])/2.0
        #div_errs[i] = (np.sqrt(n_a)/n_a)*divided_e[i] if n_a != 0 else 0
        n3_bins[i] = (np_hist_n3[1][i]+np_hist_n3[1][i+1])/2.0
        n4_bins[i] = (np_hist_n4[1][i]+np_hist_n4[1][i+1])/2.0

        #t_err = np_hist_t[0][i]
        #t_errs[i] = (np.sqrt(t_err)/t_err)

        n_err = np_hist_n[0][i]
        n_errs[i] = (np.sqrt(n_err)/n_err)

        #b_err = np_hist_b[0][i]
        #b_errs[i] = (np.sqrt(b_err)/b_err)

        #p_err = np_hist_p[0][i]
        #p_errs[i] = (np.sqrt(p_err)/p_err)

        #x_err = np_hist_x[0][i]
        #x_errs[i] = (np.sqrt(x_err)/x_err)

        #o_err = np_hist_o[0][i]
        #o_errs[i] = (np.sqrt(o_err)/o_err)

        n3_err = np_hist_n3[0][i]
        n3_errs[i] = (np.sqrt(n3_err)/n3_err)

        n4_err = np_hist_n4[0][i]
        n4_errs[i] = (np.sqrt(n4_err)/n4_err)

    gain1 = 30.0
    gain2 = 50.0
    w = 35.075
    divided_bins_kev = divided_bins/(gain1 * gain2)*w*1E-3
    #print(divided_bins_kev)
    #print(divided_e)
    #input('well?')
    n3_kev = np_hist_n3[0]/(gain1 * gain2)*w*1E-3
    n4_kev = np_hist_n4[0]/(gain1 * gain2)*w*1E-3

    divided_e_prompt = divided_e/(gain1 * gain2)*w*1E-3

    ### Begin plotting
    if root_style == True :
        color = 'black'
        facecolor=None
    elif root_style == False :
        sns.set(color_codes=True)
        color = None

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #ax1.hist(n_e, bins = 500, color=color, histtype='step')
    #ax1.set_xlabel('Neutron Sum Q')
    #ax3.hist(all_e, bins = 500, color=color, histtype='step')
    #ax3.set_xlabel('All Sum Q')
    ##print(len(divided_e), len(np_hist_n[1]))
    ##ax2.scatter(divided_bins, divided_e)
    #ax2.errorbar(divided_bins, divided_e, yerr=div_errs, fmt='o', capsize=0,
    #        color=color)
    #ax2.set_xlabel('Neutron Sum Q')
    #ax2.set_ylabel('Efficiency')

    # 'Efficiency' plot individually

    h, (ax1) = plt.subplots(1, 1)
    ax1.errorbar(divided_bins_kev, divided_e, yerr=div_errs, fmt='o', capsize=0,
            color=color)
    #ax1.errorbar(divided_bins, divided_e, yerr=div_errs, fmt='o', capsize=0,
    #        color=color)
    #ax1.set_xlabel('Neutron Recoil Sum Q')
    ax1.set_xlabel('Detected Energy (keV)')
    ax1.set_ylabel('Efficiency')
    #ax1.set_xlim(0,1E7)
    ax1.set_xlim(0,200)
    #h.savefig('neutron_efficiency_energy.eps')
    plt.show()

    #h, (ax1) = plt.subplots(1, 1)
    #ax1.errorbar(divided_bins_kev, np_hist_n3[0], yerr=div_errs, fmt='o', capsize=0,
    #        color=color)
    ##ax1.errorbar(divided_bins, divided_e, yerr=div_errs, fmt='o', capsize=0,
    ##        color=color)
    ##ax1.set_xlabel('Neutron Recoil Sum Q')
    #ax1.set_xlabel('Detected Energy (keV)')
    #ax1.set_ylabel('Efficiency')
    ##ax1.set_xlim(0,1E7)
    #ax1.set_xlim(0,200)
    #h.savefig('tpc3_neutron_efficiency_energy.eps')
    #plt.show()

    #h, (ax1) = plt.subplots(1, 1)
    #ax1.errorbar(divided_bins_kev, np_hist_n4[0], yerr=div_errs, fmt='o', capsize=0,
    #        color=color)
    ##ax1.errorbar(divided_bins, divided_e, yerr=div_errs, fmt='o', capsize=0,
    ##        color=color)
    ##ax1.set_xlabel('Neutron Recoil Sum Q')
    #ax1.set_xlabel('Detected Energy (keV)')
    #ax1.set_ylabel('Efficiency')
    ##ax1.set_xlim(0,1E7)
    #ax1.set_xlim(0,200)
    #h.savefig('tpc4_neutron_efficiency_energy.eps')
    #plt.show()

    # "Dot" histogram of all PID
    #ax4.errorbar(divided_bins, np_hist_t[0], yerr=t_errs, fmt='o', capsize=0,
    #        color='orange', label='Top alphas')
    #ax4.errorbar(divided_bins, np_hist_b[0], yerr=b_errs, fmt='o', capsize=0,
    #        color='yellow', label='Bottom alphas')
    #ax4.errorbar(divided_bins, np_hist_n[0], yerr=n_errs, fmt='o', capsize=0,
    #        label='Neutrons')

    #ax4.errorbar(divided_bins, np_hist_p[0], yerr=p_errs, fmt='o', capsize=0,
    #        color='purple', label='Protons')
    #ax4.errorbar(divided_bins, np_hist_x[0], yerr=x_errs, fmt='o', capsize=0,
    #        color='black', label='x rays')
    #ax4.errorbar(divided_bins, np_hist_o[0], yerr=o_errs, fmt='o', capsize=0,
    #        color='red', label='Others')
    #ax4.set_xlabel('Sum Q')
    #ax4.legend(loc='upper right')

    # Bar histogram of all PID
    #ax4.hist(topa_e, bins = 100, color='orange')
    #ax4.hist(bota_e, bins = 100, color='yellow')
    #ax4.hist(p_e, bins = 100, color='purple')
    #ax4.hist(x_e, bins = 100, color='black')
    #ax4.hist(others_e, bins = 100, color='red')
    #ax4.hist(n_e, bins = 100)

    # Neutron energy spectrum in each TPC using sumQ
    #g, (bx1, bx2) = plt.subplots(1, 2, sharey=True)
    #bx1.errorbar(n3_bins, np_hist_n3[0], yerr=n3_errs, fmt='o', capsize=0,
    #        color=color)
    #bx1.set_title('Sum Q for Neutron Candidates (Ch. 3)')
    #bx1.set_xlabel('Sum Q')
    #bx1.set_yscale('log')

    #bx2.errorbar(n4_bins, np_hist_n4[0], yerr=n4_errs, fmt='o', capsize=0,
    #        color=color)
    #bx2.set_xlabel('Sum Q')
    #bx2.set_title('Sum Q for Neutron Candidates (Ch. 4)')

    ## Neutron energy spectrum in each TPC using sumQ
    #h, (cx1, cx2) = plt.subplots(1, 2, sharey=True)
    #cx1.errorbar(n3_bins_kev, np_hist_n3[0], yerr=n3_errs, fmt='o', capsize=0,
    #       color=color)
    #cx1.set_title('Energy of Neutron Candidates (Ch. 3)')
    #cx1.set_xlabel('Sum E (KeV)')
    #cx1.set_yscale('log')

    #cx2.errorbar(n4_bins_kev, np_hist_n4[0], yerr=n4_errs, fmt='o', capsize=0,
    #       color=color)
    #cx2.set_xlabel('Sum E (KeV)')
    #cx2.set_title('Energy of Neutron Candidates (Ch. 4)')
    #cx2.set_yscale('log')

    #plt.show()
    #hist_n.Draw()
    #input('well?')

    ### Plot figures individually
    # Neutron energy spectrum in each TPC using sumQ
    #plt.errorbar(n3_bins, np_hist_n3[0], yerr=n3_errs, fmt='o', capsize=0,
    #        color=color)
    #plt.set_title('Sum Q for Neutron Candidates (Ch. 3)')
    #plt.set_xlabel('Sum Q')
    #plt.set_yscale('log')
    #plt.show()

    #plt.errorbar(n4_bins, np_hist_n4[0], yerr=n4_errs, fmt='o', capsize=0,
    #        color=color)
    #plt.set_xlabel('Sum Q')
    #plt.set_title('Sum Q for Neutron Candidates (Ch. 4)')
    #plt.show()

    h, (cx1) = plt.subplots(1, 1)
    cx1.errorbar(n3_bins_kev, np_hist_n3[0], yerr=n3_errs, fmt='o', capsize=0,
           color=color)
    #plt.title('Energy of Neutron Candidates (Ch. 3)')
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    h.savefig('ch3_neutron_energies.eps') 
    plt.show()

    h, (cx1) = plt.subplots(1, 1)
    cx1.errorbar(n4_bins_kev, np_hist_n4[0], yerr=n4_errs, fmt='o', capsize=0,
           color=color)
    #plt.title('Energy of Neutron Candidates (Ch. 4)')
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    h.savefig('ch4_neutron_energies.eps') 
    plt.show()

    #plt.errorbar(n4_bins_kev, np_hist_n4[0], yerr=n4_errs, fmt='o', capsize=0,
    #       color=color)
    #plt.xlabel('Sum E (KeV)')
    ##plt.title('Energy of Neutron Candidates (Ch. 4)')
    #plt.yscale('log')
    #plt.show()

def energy_study(datapath, simpath):
    ### Define good data runs
    good_files = [1464483600,
                  1464487200,
                  1464490800,
                  1464494400,
                  1464498000,
                  1464501600,
                  1464505200,
                  1464483600,
                  1464487200,
                  1464490800,
                  1464494400,
                  1464498000,
                  1464501600,
                  1464505200]
    ### Populate data arrays
    print('Populating data arrays ...')
    data_hitsides = []
    data_Q = []
    data_tlengths = []
    data_detnbs = []
    data_npoints = []
    data_min_rets = []
    

    branches = ['de_dx', 
                'e_sum', 
                'detnb',
                'hitside', 
                'npoints',
                'min_ret', 
                't_length']

    for f in os.listdir(datapath):
        strs = f.split('_')
        if int(strs[-2]) not in good_files : continue

        r_file = str(datapath) + str(f)
        print(r_file)

        data = root2rec(r_file, branches=branches)

        if data.detnb[0] == 3 : data.e_sum *= 1.18
        if data.detnb[0] == 4 : data.e_sum *= 1.64

        data_hitsides = np.concatenate([data_hitsides, data.hitside])
        data_Q = np.concatenate([data_Q, data.e_sum])
        data_tlengths = np.concatenate([data_tlengths, data.t_length])
        data_detnbs = np.concatenate([data_detnbs, data.detnb])
        data_npoints = np.concatenate([data_npoints, data.npoints])
        data_min_rets = np.concatenate([data_min_rets, data.min_ret])

    gain1 = 30.0
    gain2 = 50.0
    W = 35.075

    data_E = data_Q/(gain1 * gain2) * W * 1E-3

    data_dQdx = data_Q/data_tlengths
    data_dQdx[data_tlengths==0] = 0

    # Selections in data
    ch3_data_sels = (
                (data_detnbs == 3)
                & (data_hitsides == 0)
                & (data_min_rets == 0)
                & (data_dQdx > 500.0)
                & (data_npoints > 40)
                )

    ch4_data_sels = (
                (data_detnbs == 4)
                & (data_hitsides == 0)
                & (data_min_rets == 0)
                & (data_dQdx > 500.0)
                & (data_npoints > 40)
                )

    ### Populate simulation arrays
    print('Data arrays populated.  Moving to simulation ...')
    sim_hitsides = []
    sim_pdgs = []
    sim_Q = []
    sim_tlengths = []
    sim_detnbs = []
    sim_bcids = []
    sim_tots = []
    sim_npoints = []
    sim_min_rets = []

    touschek = []
    beam_gas = []

    truth_KE = []
    truth_file = '/Users/BEASTzilla/BEAST/sim/v5.2/FTFP_BERT_HP/mc_beast_run_2016-02-09.root'

    branches = ['de_dx', 
                'pdg',
                'e_sum', 
                'detnb',
                'hitside', 
                'min_ret', 
                't_length',
                'bcid',
                'npoints',
                'tot',
                'truth_KineticEnergy',
                ]

    for f in os.listdir(simpath):
        if '.root' not in f or 'HER' in f : continue
        r_file = str(simpath) + str(f)
        print(r_file)

        sim = root2rec(r_file, branches=branches)

        sim_hitsides = np.concatenate([sim_hitsides, sim.hitside])
        sim_pdgs = np.concatenate([sim_pdgs, sim.pdg])
        sim_Q = np.concatenate([sim_Q, sim.e_sum])
        sim_tlengths = np.concatenate([sim_tlengths, sim.t_length])
        sim_bcids = np.concatenate([sim_bcids, sim.bcid])
        sim_tots = np.concatenate([sim_tots, sim.tot])
        sim_npoints = np.concatenate([sim_npoints, sim.npoints])
        sim_detnbs = np.concatenate([sim_detnbs, sim.detnb+1])
        sim_min_rets = np.concatenate([sim_min_rets, sim.min_ret])

        tree_name = r_file.split('.')[-2].split('/')[-1]
        truth = root2rec(truth_file, tree_name)
        truth_KE = np.concatenate([truth_KE, truth.truth_KineticEnergy*1E3])

        if 'Touschek' in f :
            touschek = np.concatenate([touschek, np.ones(len(sim)) ])
            beam_gas = np.concatenate([beam_gas, np.zeros(len(sim)) ])
        else :
            beam_gas = np.concatenate([beam_gas, np.ones(len(sim)) ])
            touschek = np.concatenate([touschek, np.zeros(len(sim)) ])
    

    sim_dQdx = sim_Q/sim_tlengths
    sim_dQdx[sim_tlengths == 0] = 0

    sim_E = sim_Q/(gain1 * gain2) * W * 1E-3

    pulse_widths = []

    # Objects for using TGraph method of TOT->charge conversion
    sim_Q_v2 =[]
    grPlsrDACvTOT = TGraph("TOTcalibration.txt")
    PlsrDACtoQ = 52.0
    sim_tots_v2 = []

    for i in range(len(sim_bcids)) :
        # Calculate pulse widths
        lowest_bcid = np.min(sim_bcids[i])
        pulses = sim_bcids[i] + sim_tots[i]
        largest_pulse = np.max(pulses)
        largest_pulse_element = np.where(pulses==largest_pulse)
        
        evt_pulse_width = largest_pulse - lowest_bcid
        pulse_widths.append(evt_pulse_width)
        
        # Convert TOT -> charge via new method (TGraph)
        event_tot_v2 = []
        for k in range(int(sim_npoints[i])) :
            #tot_v2 = PlsrDACtoQ * grPlsrDACvTOT.Eval(sim_tots[i][k])
            #print(tot_v2)
            tot_v2 = PlsrDACtoQ * grPlsrDACvTOT.Eval(sim_tots[i][k]+0.5)
            #print(tot_v2)
            #input('well?')
            event_tot_v2.append(tot_v2)
        event_tot_v2 = np.array(event_tot_v2)
        sim_tots_v2.append(event_tot_v2)
        sim_Q_v2.append(np.sum(event_tot_v2))

    sim_Q_v2 = np.array(sim_Q_v2)

    sim_E_v2 = sim_Q_v2/(gain1 * gain2) * W * 1E-3
        
    pulse_widths = np.array(pulse_widths)


    # Selections in sim
    ch3_touschek_sels = (
                (sim_detnbs == 3)
                & (touschek==1)
                & (pulse_widths > 3)
                & (sim_pdgs > 10000)
                & (sim_hitsides == 0)
                & (sim_min_rets == 0)
                & (sim_dQdx > 500)
                & (sim_npoints > 40)
                )

    ch4_touschek_sels = (
                (sim_detnbs == 4)
                & (touschek==1)
                & (pulse_widths > 3)
                & (sim_pdgs > 10000)
                & (sim_hitsides == 0)
                & (sim_min_rets == 0)
                & (sim_dQdx > 500)
                & (sim_npoints > 40)
                )

    ch3_beamgas_sels = (
                (sim_detnbs == 3)
                & (beam_gas==1)
                & (pulse_widths > 3)
                & (sim_pdgs > 10000)
                & (sim_hitsides == 0)
                & (sim_min_rets == 0)
                & (sim_dQdx > 500)
                & (sim_npoints > 40)
                )

    ch4_beamgas_sels = (
                (sim_detnbs == 4)
                & (beam_gas==1)
                & (pulse_widths > 3)
                & (sim_pdgs > 10000)
                & (sim_hitsides == 0)
                & (sim_min_rets == 0)
                & (sim_dQdx > 500)
                & (sim_npoints > 40)
                )

    ### Debug
    ch3_v1_ratio = sim_E[ch3_touschek_sels]/truth_KE[ch3_touschek_sels]
    ch3_v1_ratio = np.concatenate([ch3_v1_ratio,
        sim_E[ch3_beamgas_sels]/truth_KE[ch3_beamgas_sels]]) 
    ch4_v1_ratio = sim_E[ch4_touschek_sels]/truth_KE[ch4_touschek_sels]
    ch4_v1_ratio = np.concatenate([ch4_v1_ratio,
        sim_E[ch4_beamgas_sels]/truth_KE[ch4_beamgas_sels]]) 
    #print(ch3_v1_ratio[ch3_v1_ratio > 1.0])
    #print(ch4_v1_ratio[ch4_v1_ratio > 1.0])

    ch3_v2_ratio = sim_E_v2[ch3_touschek_sels]/truth_KE[ch3_touschek_sels]
    ch4_v2_ratio = sim_E_v2[ch4_touschek_sels]/truth_KE[ch4_touschek_sels]
    print(ch3_v2_ratio[ch3_v2_ratio > 1.0])
    print(ch4_v2_ratio[ch4_v2_ratio > 1.0])
    
    print('Investigating events with  reco/truth KE greater than 1')
    print('Npoints:', sim_npoints[ch4_v2_ratio > 1.0][0])
    print(np.min(sim_tots[ch4_v2_ratio > 1.0][0]))


    ### Quick plots for other purposes
    d = plt.figure()
    ax0 = d.add_subplot(111)
    ax0.scatter(sim_E[ch3_touschek_sels]/truth_KE[ch3_touschek_sels],
            sim_E[ch3_touschek_sels], color='C0')
    ax0.scatter(sim_E[ch4_touschek_sels]/truth_KE[ch4_touschek_sels],
            sim_E[ch4_touschek_sels], color='C0')
    ax0.scatter(sim_E[ch3_beamgas_sels]/truth_KE[ch3_beamgas_sels],
            sim_E[ch3_beamgas_sels], color='C0')
    ax0.scatter(sim_E[ch4_beamgas_sels]/truth_KE[ch4_beamgas_sels],
            sim_E[ch4_beamgas_sels], color='C0')
    ax0.set_xlabel('Reco/truth recoil energy', ha='right', x=1.0) 
    ax0.set_ylabel('Reconstructed recoil energy [keV]', ha='right', y=1.0)
    ax0.set_xlim(plt.xlim()[0], 1.50)
    ax0.set_ylim(plt.ylim()[0], 1750.0)

    e = plt.figure()
    ax1 = e.add_subplot(111)
    ax1.scatter(sim_E_v2[ch3_touschek_sels]/truth_KE[ch3_touschek_sels],
            sim_E_v2[ch3_touschek_sels], color='C0')
    ax1.scatter(sim_E_v2[ch4_touschek_sels]/truth_KE[ch4_touschek_sels],
            sim_E_v2[ch4_touschek_sels], color='C0')
    ax1.scatter(sim_E_v2[ch3_beamgas_sels]/truth_KE[ch3_beamgas_sels],
            sim_E_v2[ch3_beamgas_sels], color='C0')
    ax1.scatter(sim_E_v2[ch4_beamgas_sels]/truth_KE[ch4_beamgas_sels],
            sim_E_v2[ch4_beamgas_sels], color='C0')
    ax1.set_xlabel('Reco/truth recoil energy', ha='right', x=1.0) 
    ax1.set_ylabel('Reconstructed recoil energy [keV]', ha='right', y=1.0)
    ax1.set_xlim(plt.xlim()[0], 1.50)
    ax1.set_ylim(plt.ylim()[0], 1750.0)

    plt.show()

    ### Debug
    print('Printing number of events in Data, Touschek, and Beamgas in ch3, ch4')
    print(len(data_E[ch3_data_sels]) )
    print(len(sim_E[ch3_touschek_sels]) )
    print(len(sim_E[ch3_beamgas_sels]) )

    print(len(data_E[ch4_data_sels]) )
    print(len(sim_E[ch4_touschek_sels]) )
    print(len(sim_E[ch4_beamgas_sels]) )

    ### Define exponential function for fitting recoil energy spectra

    def exp_pdf(x, a, b):
        return a*np.exp(-b*x)

    ### Histogram the arrays
    (ch3_data_n, ch3_data_bins, ch3_data_patches) = plt.hist(data_E[ch3_data_sels], bins=25, range=[0,
        np.max(data_E[ch3_data_sels])] )

    ch3_data_bin_centers = 0.5 * (ch3_data_bins[:-1] + ch3_data_bins[1:])

    ch3_data_errs = np.sqrt(ch3_data_n)
    ch3_data_errs[ch3_data_errs==0] = 1
    ch3_data_chi2 = probfit.Chi2Regression(exp_pdf, ch3_data_bin_centers, ch3_data_n,
            ch3_data_errs)
 
    ch3_data_minu = iminuit.Minuit(ch3_data_chi2, a=2000.0, error_a = 1.0, b=0.0001, error_b=0.00001)
    ch3_data_minu.migrad()
    ch3_data_pars = ch3_data_minu.values
    ch3_data_p_errs = ch3_data_minu.errors
    print('\nPrinting Ch3 data fit parameters')
    print(ch3_data_pars, ch3_data_p_errs)

    (ch4_data_n, ch4_data_bins, ch4_data_patches) = plt.hist(data_E[ch4_data_sels], bins=25, range=[0,
        np.max(data_E[ch4_data_sels])] )

    ch4_data_bin_centers = 0.5 * (ch4_data_bins[:-1] + ch4_data_bins[1:])

    ch4_data_errs = np.sqrt(ch4_data_n)
    ch4_data_errs[ch4_data_errs==0] = 1
    ch4_data_chi2 = probfit.Chi2Regression(exp_pdf, ch4_data_bin_centers, ch4_data_n,
            ch4_data_errs)
 
    ch4_data_minu = iminuit.Minuit(ch4_data_chi2, a=2000.0, error_a = 1.0, b=0.0001, error_b=0.00001)
    ch4_data_minu.migrad()
    ch4_data_pars = ch4_data_minu.values
    ch4_data_p_errs = ch4_data_minu.errors
    print('\nPrinting Ch4 data fit parameters')
    print(ch4_data_pars, ch4_data_p_errs)

    beast_datapath = '/Users/BEASTzilla/BEAST/data/v3.1/'
    subrun_times, subrun_BeamGas, subrun_Touschek, _ = calc_sim_weights(beast_datapath, simpath)

    # Normalizing [Touschek, Beamgas] weights by time and sim beam conditions
    ch3_weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_E[ch3_touschek_sels])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_E[ch3_beamgas_sels])),
            ]

    ch3_weights[0] *= np.sum(subrun_Touschek*subrun_times)
    ch3_weights[1] *= np.sum(subrun_BeamGas*subrun_times)

    ch4_weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_E[ch4_touschek_sels])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_E[ch4_beamgas_sels])),
            ]

    ch4_weights[0] *= np.sum(subrun_Touschek*subrun_times)
    ch4_weights[1] *= np.sum(subrun_BeamGas*subrun_times)

    (ch3_touschek_n, ch3_touschek_bins, ch3_touschek_patches)=plt.hist(sim_E[ch3_touschek_sels], 
        bins=ch3_data_bins, range=[0, np.max(sim_E[ch3_touschek_sels])],
        weights=ch3_weights[0])

    ch3_touschek_bin_centers = 0.5 * (ch3_touschek_bins[:-1] + ch3_touschek_bins[1:])

    ch3_touschek_errs = np.sqrt(ch3_touschek_n)
    ch3_touschek_errs[ch3_touschek_errs==0] = 1
    ch3_touschek_chi2 = probfit.Chi2Regression(exp_pdf, ch3_touschek_bin_centers,
            ch3_touschek_n, ch3_touschek_errs)
 
    ch3_touschek_minu = iminuit.Minuit(ch3_touschek_chi2, a=2000.0, error_a = 1.0, b=0.0001,
            error_b=0.00001)
    ch3_touschek_minu.migrad()
    ch3_touschek_pars = ch3_touschek_minu.values
    ch3_touschek_p_errs = ch3_touschek_minu.errors
    print('\nPrinting Ch3 touschek fit parameters')
    print(ch3_touschek_pars, ch3_touschek_p_errs)

    (ch4_touschek_n, ch4_touschek_bins, ch4_touschek_patches)=plt.hist(sim_E[ch4_touschek_sels], 
        bins=ch4_data_bins, range=[0, np.max(sim_E[ch4_touschek_sels])],
        weights=ch4_weights[0])

    ch4_touschek_bin_centers = 0.5 * (ch4_touschek_bins[:-1] + ch4_touschek_bins[1:])

    ch4_touschek_errs = np.sqrt(ch4_touschek_n)
    ch4_touschek_errs[ch4_touschek_errs==0] = 1
    ch4_touschek_chi2 = probfit.Chi2Regression(exp_pdf, ch4_touschek_bin_centers,
            ch4_touschek_n, ch4_touschek_errs)
 
    ch4_touschek_minu = iminuit.Minuit(ch4_touschek_chi2, a=2000.0, error_a = 1.0, b=0.0001,
            error_b=0.00001)
    ch4_touschek_minu.migrad()
    ch4_touschek_pars = ch4_touschek_minu.values
    ch4_touschek_p_errs = ch4_touschek_minu.errors
    print('\nPrinting Ch4 touschek fit parameters')
    print(ch4_touschek_pars, ch4_touschek_p_errs)

    (ch3_beamgas_n, ch3_beamgas_bins, ch3_beamgas_patches) = plt.hist(sim_E[ch3_beamgas_sels],
        bins=ch3_data_bins, range=[0, np.max(sim_E[ch3_beamgas_sels])] ,
        weights=ch3_weights[1])

    ch3_beamgas_errs = np.sqrt(ch3_beamgas_n)
    ch3_beamgas_errs[ch3_beamgas_errs==0] = 1
    ch3_beamgas_bin_centers = 0.5 * (ch3_beamgas_bins[:-1] + ch3_beamgas_bins[1:])

    ch3_beamgas_chi2 = probfit.Chi2Regression(exp_pdf, ch3_beamgas_bin_centers,
            ch3_beamgas_n, ch3_beamgas_errs)
 
    ch3_beamgas_minu = iminuit.Minuit(ch3_beamgas_chi2, a=2000.0, error_a = 1.0, b=0.0001,
            error_b=0.00001)
    ch3_beamgas_minu.migrad()
    ch3_beamgas_pars = ch3_beamgas_minu.values
    ch3_beamgas_p_errs = ch3_beamgas_minu.errors
    print('\nPrinting Ch3 beamgas fit parameters')
    print(ch3_beamgas_pars, ch3_beamgas_p_errs)

    (ch4_beamgas_n, ch4_beamgas_bins, ch4_beamgas_patches) = plt.hist(sim_E[ch4_beamgas_sels],
        bins=ch4_data_bins, range=[0, np.max(sim_E[ch4_beamgas_sels])] ,
        weights=ch4_weights[1])

    ch4_beamgas_errs = np.sqrt(ch4_beamgas_n)
    ch4_beamgas_errs[ch4_beamgas_errs==0] = 1
    ch4_beamgas_bin_centers = 0.5 * (ch4_beamgas_bins[:-1] + ch4_beamgas_bins[1:])

    ch4_beamgas_chi2 = probfit.Chi2Regression(exp_pdf, ch4_beamgas_bin_centers,
            ch4_beamgas_n, ch4_beamgas_errs)
 
    ch4_beamgas_minu = iminuit.Minuit(ch4_beamgas_chi2, a=2000.0, error_a = 1.0, b=0.0001,
            error_b=0.00001)
    ch4_beamgas_minu.migrad()
    ch4_beamgas_pars = ch4_beamgas_minu.values
    ch4_beamgas_p_errs = ch4_beamgas_minu.errors
    print('\nPrinting Ch4 beamgas fit parameters')
    print(ch4_beamgas_pars, ch4_beamgas_p_errs)

    ch3_bkg_bin_centers = [ch3_touschek_bin_centers, ch3_beamgas_bin_centers]
    ch3_bkg_bins = [ch3_touschek_bins, ch3_beamgas_bins]
    ch3_bkg_weights = [ch3_touschek_n, ch3_beamgas_n]

    ch3_bkg_list=[sim_E[ch3_touschek_sels], sim_E[ch3_beamgas_sels]]

    ((ch3_touschek_edges, ch3_touscheky), _, (ch3_touschek_pdf_x,
        ch3_touschek_pdf_y), ch3_parts) = ch3_touschek_chi2.draw(ch3_touschek_minu,
                                                    parts=True)

    ((ch3_beamgas_edges, ch3_beamgasy), _, (ch3_beamgas_pdf_x,
        ch3_beamgas_pdf_y), ch3_parts ) = ch3_beamgas_chi2.draw(ch3_beamgas_minu,
                                                    parts=True)

    ((ch3_data_edges, ch3_datay), _, (ch3_data_pdf_x,
        ch3_data_pdf_y), ch3_parts) = ch3_data_chi2.draw(ch3_data_minu,
                                                    parts=True)

    ch4_bkg_bin_centers = [ch4_touschek_bin_centers, ch4_beamgas_bin_centers]
    ch4_bkg_bins = [ch4_touschek_bins, ch4_beamgas_bins]
    ch4_bkg_weights = [ch4_touschek_n, ch4_beamgas_n]

    ch4_bkg_list=[sim_E[ch4_touschek_sels], sim_E[ch4_beamgas_sels]]

    ((ch4_touschek_edges, ch4_touscheky), _, (ch4_touschek_pdf_x,
        ch4_touschek_pdf_y), ch4_parts) = ch4_touschek_chi2.draw(ch4_touschek_minu,
                                                    parts=True)

    ((ch4_beamgas_edges, ch4_beamgasy), _, (ch4_beamgas_pdf_x,
        ch4_beamgas_pdf_y), ch4_parts ) = ch4_beamgas_chi2.draw(ch4_beamgas_minu,
                                                    parts=True)

    ((ch4_data_edges, ch4_datay), _, (ch4_data_pdf_x,
        ch4_data_pdf_y), ch4_parts) = ch4_data_chi2.draw(ch4_data_minu,
                                                    parts=True)

    ### Plots
    f = plt.figure()
    ax1 = f.add_subplot(111)

    ax1.hist([ch3_data_bin_centers,ch3_data_bin_centers], bins=ch3_data_bins,
            weights=ch3_bkg_weights,
            range=[0,np.max(data_E[ch3_data_sels])],
            label=['Touschek MC','Beam Gas MC'],
            stacked=True, color=['C0','C1'])

    ax1.errorbar(ch3_data_bin_centers, ch3_data_n, yerr=np.sqrt(ch3_data_n), fmt='o', color='black',
            label='Experiment')
    ax1.plot(ch3_data_pdf_x, ch3_data_pdf_y, color='r', lw=2)
    ax1.plot(ch3_touschek_pdf_x, ch3_touschek_pdf_y, color='C2', lw=2)
    ax1.plot(ch3_beamgas_pdf_x, ch3_beamgas_pdf_y, color='C2', lw=2)
    ax1.set_xlabel('Detected Energy [keV]', ha='right', x=1.0)
    ax1.set_ylabel('Events per bin', ha='right', y=1.0)
    ax1.set_ylim(1E-1,1E4)
    ax1.set_yscale('log')
    ax1.grid(b=False)
    ax1.legend(loc='best')
    f.savefig('ch3_recoilE_datavsMC.pdf')


    g = plt.figure()
    ax2 = g.add_subplot(111)

    ax2.hist([ch4_data_bin_centers,ch4_data_bin_centers], bins=ch4_data_bins,
            weights=ch4_bkg_weights,
            range=[0,np.max(data_E[ch4_data_sels])],
            label=['Touschek MC','Beam Gas MC'],
            stacked=True, color=['C0','C1'])

    ax2.errorbar(ch4_data_bin_centers, ch4_data_n, yerr=np.sqrt(ch4_data_n), fmt='o', color='black',
            label='Experiment')
    ax2.plot(ch4_data_pdf_x, ch4_data_pdf_y, color='r', lw=2)
    ax2.plot(ch4_touschek_pdf_x, ch4_touschek_pdf_y, color='C2', lw=2)
    ax2.plot(ch4_beamgas_pdf_x, ch4_beamgas_pdf_y, color='C2', lw=2)
    ax2.set_xlabel('Detected Energy [keV]', ha='right', x=1.0)
    ax2.set_ylabel('Events per bin', ha='right', y=1.0)
    ax2.set_ylim(1E-1,1E4)
    ax2.set_yscale('log')
    ax2.grid(b=False)
    ax2.legend(loc='best')
    g.savefig('ch4_recoilE_datavsMC.pdf')

    #h = plt.figure()
    #ax3 = h.add_subplot(111)
    #ax3.hist([ch4_data_bin_centers,ch4_data_bin_centers], bins=ch4_data_bins,
    #        weights=(ch4_bkg_weights + ch3_bkg_weights),
    #        range=[0,np.max(data_E[ch4_data_sels])],
    #        label=['Touschek MC','Beam Gas MC'],
    #        stacked=True, color=['C0','C1'])

    #ax3.errorbar(ch4_data_bin_centers, ch4_data_n + ch3_data_n,
    #        yerr=np.sqrt(ch4_data_n + ch3_data_n),
    #        fmt='o', color='black',
    #        label='Experiment')
    #ax3.plot(ch4_data_pdf_x, ch4_data_pdf_y, color='r', lw=2)
    #ax3.plot(ch4_touschek_pdf_x, ch4_touschek_pdf_y, color='C2', lw=2)
    #ax3.plot(ch4_beamgas_pdf_x, ch4_beamgas_pdf_y, color='C2', lw=2)
    #ax3.set_xlabel('Detected Energy [keV]', ha='right', x=1.0)
    #ax3.set_ylabel('Events per bin', ha='right', y=1.0)
    #ax3.set_ylim(1E-1,1E4)
    #ax3.set_yscale('log')
    #ax3.grid(b=False)
    #ax3.legend(loc='best')

    plt.show()

def gain_study(gain_path):
    t3_etop = []
    t3_ebottom = []
    t3t_ts = []
    t3b_ts = []
    t4_etop = []
    t4_ebottom = []
    t4t_ts = []
    t4b_ts = []

    tmax = 1464505369.0
    tmin = 1464485170.0
    t3_start = tmin
    t4_start = tmin

    print(gain_path)

    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    for subdir, dirs, files in os.walk(gain_path):
        for f in files:
            strs = f.split('_')
            if int(strs[-2]) not in good_files : continue
            r_file = str(subdir) + str('/') + str(f)
            data = root2rec(r_file)
            for event in data:
                if event.tstamp > tmax or event.tstamp < tmin: continue
                if (data.detnb[0] == 3) :
                    if (event.top_alpha == 1 and event.theta > 85.0 and 
                            event.theta < 95.0 and event.phi > -5.0 and 
                            event.phi < 5.0 ):
                        t3_etop.append(event.e_sum)
                        ts = event.tstamp-t3_start
                        t3t_ts.append(ts)

                    if (event.bottom_alpha == 1 and event.theta > 85.0 and
                            event.theta < 95.0 and event.phi > -5.0 and 
                            event.phi < 5.0 ):
                        t3_ebottom.append(event.e_sum)
                        ts = event.tstamp-t3_start
                        t3b_ts.append(ts)

                elif (data.detnb[0] == 4) :
                    #if event.top_alpha == 1 and event.theta > 85.0 and event.theta < 95.0 :
                    if (event.top_alpha == 1 and event.theta > 85.0 and event.theta < 95.0
                            and event.phi > -5.0 and event.phi < 5.0 ):
                        t4_etop.append(event.e_sum)
                        ts = event.tstamp-t4_start
                        t4t_ts.append(ts)
                    #if event.bottom_alpha == 1 and event.theta > 85.0 and event.theta < 95.0 :
                    if (event.bottom_alpha == 1 and event.theta > 85.0 and event.theta < 95.0
                            and event.phi > -5.0 and event.phi < 5.0 ):
                        t4_ebottom.append(event.e_sum)
                        ts = event.tstamp-t4_start
                        t4b_ts.append(ts)


            print('Finished file:', r_file)

    t3_etop = np.array(t3_etop)
    t3_ebottom = np.array(t3_ebottom)
    t4_etop = np.array(t4_etop)
    t4_ebottom = np.array(t4_ebottom)
    t3t_ts = np.array(t3t_ts)
    t3b_ts = np.array(t3b_ts)
    t4t_ts = np.array(t4t_ts)
    t4b_ts = np.array(t4b_ts)

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.scatter(t3t_ts, t3_etop, color='black', label='Top')
    ax1.scatter(t3b_ts, t3_ebottom, color='blue', label='Bottom')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Sum Q')
    ax1.set_title('Alpha Sum Q vs Time in TPC 3')
    ax1.legend(loc='lower left')
    ax2.scatter(t4t_ts, t4_etop, color='black', label='Top')
    ax2.scatter(t4b_ts, t4_ebottom, color='blue', label='Bottom')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Sum Q')
    ax2.set_title('Alpha Sum Q vs Time in TPC 4')
    ax2.legend(loc='lower left')

    ### Use pandas dataframe to make "profile" histogram
    #pd.set_option('display.max_rows', 1000)

    ### Using np.digitize
    bins_t3t = np.linspace(0, max(t3t_ts), 201)
    bin_centers_t3t = 0.5 * (bins_t3t[:-1] + bins_t3t[1:])
    bin_width_t3t = bins_t3t[1] - bins_t3t[0]

    bins_t3b = np.linspace(0, max(t3b_ts), 201)
    bin_centers_t3b = 0.5 * (bins_t3b[:-1] + bins_t3b[1:])
    bin_width_t3b = bins_t3b[1] - bins_t3b[0]

    bins_t4t = np.linspace(0, max(t4t_ts), 201)
    bin_centers_t4t = 0.5 * (bins_t4t[:-1] + bins_t4t[1:])
    bin_width_t4t = bins_t4t[1] - bins_t4t[0]

    bins_t4b = np.linspace(0, max(t4b_ts), 201)
    bin_centers_t4b = 0.5 * (bins_t4b[:-1] + bins_t4b[1:])
    bin_width_t4b = bins_t4b[1] - bins_t4b[0]

    df3t = pd.DataFrame({'t3t_ts': t3t_ts, 't3_etop': t3_etop})
    df3b = pd.DataFrame({'t3b_ts': t3b_ts, 't3_ebottom': t3_ebottom})
    df4t = pd.DataFrame({'t4t_ts': t4t_ts, 't4_etop': t4_etop})
    df4b = pd.DataFrame({'t4b_ts': t4b_ts, 't4_ebottom': t4_ebottom})

    df3t['bin'] = np.digitize(t3t_ts, bins=bins_t3t)
    df3b['bin'] = np.digitize(t3b_ts, bins=bins_t3b)
    df4t['bin'] = np.digitize(t4t_ts, bins=bins_t4t)
    df4b['bin'] = np.digitize(t4b_ts, bins=bins_t4b)

    binned_t3 = df3t.groupby('bin')
    result_t3 = binned_t3['t3_etop'].agg(['mean', 'sem'])
    result_t3.fillna(0)
    bins = np.array(result_t3.index.values)
    x_binned = np.array([0.]*len(bins))
    for i in range(len(bins)):
        if bins[i] < 3 : 
            x_binned[i] = bin_centers_t3t[0]
        else : x_binned[i] = bin_centers_t3t[bins[i]-2]
    result_t3['x'] = x_binned
    result_t3['xerr'] = x_binned/ 2.0

    binned_b3 = df3b.groupby('bin')
    result_b3 = binned_b3['t3_ebottom'].agg(['mean', 'sem'])
    result_b3.fillna(0)
    bins = np.array(result_b3.index.values)
    x_binned = np.array([0.]*len(bins))
    for i in range(len(bins)):
        if bins[i] < 3 : 
            x_binned[i] = bin_centers_t3b[0]
        else : x_binned[i] = bin_centers_t3b[bins[i]-2]
    result_b3['x'] = x_binned
    result_b3['xerr'] = x_binned/ 2.0

    binned_t4 = df4t.groupby('bin')
    result_t4 = binned_t4['t4_etop'].agg(['mean', 'sem'])
    result_t4.fillna(0)
    bins = np.array(result_t4.index.values)
    x_binned = np.array([0.]*len(bins))
    for i in range(len(bins)):
        if bins[i] < 3 : 
            x_binned[i] = bin_centers_t4t[0]
        else : x_binned[i] = bin_centers_t4t[bins[i]-2]
    print(binned_t4)
    result_t4['x'] = x_binned
    result_t4['xerr'] = x_binned/ 2.0

    binned_b4 = df4b.groupby('bin')
    result_b4 = binned_b4['t4_ebottom'].agg(['mean', 'sem'])
    result_b4.fillna(0)
    bins = np.array(result_b4.index.values)
    x_binned = np.array([0.]*len(bins))
    for i in range(len(bins)):
        if bins[i] < 3 : 
            x_binned[i] = bin_centers_t4b[0]
        else : x_binned[i] = bin_centers_t4b[bins[i]-2]
    result_b4['x'] = x_binned
    result_b4['xerr'] = x_binned/ 2.0

    # Print mean and rms of gain
    #print(len(t3_etop), len(t3_ebottom), len(t4_etop), len(t4_ebottom))
    print(len(result_t3['mean'].values))
    print(len(result_b3['mean'].values))
    print(len(result_t4['mean'].values))
    print(len(result_b4['mean'].values))

    s = (''' 
    Ch. 3 top source: Mean = %f, RMS = %f
    Ch. 3 bottom source: Mean = %f, RMS = %f

    Ch. 4 top source: Mean = %f, RMS = %f
    Ch. 4 bottom source: Mean = %f, RMS = %f
        ''')

    t3_tm = np.mean(t3_etop)
    t3_bm = np.mean(t3_ebottom)
    t4_tm = np.mean(t4_etop)
    t4_bm = np.mean(t4_ebottom)

    t3_tr = np.std(t3_etop)
    t3_br = np.std(t3_ebottom)
    t4_tr = np.std(t4_etop)
    t4_br = np.std(t4_ebottom)

    print(s % (t3_tm, t3_tr, t3_bm, t3_br, t4_tm, t4_tr, t4_bm, t4_br))
    print(len(result_t3['mean'].values[:-10]))
    print(len(result_b3['mean'].values[:-10]))
    print(len(result_t4['mean'].values[:-10]))
    print(len(result_b4['mean'].values[:-10]))
    t3_tm = np.mean(result_t3['mean'].values[:-10])
    t3_bm = np.mean(result_b3['mean'].values[:-10])
    t4_tm = np.mean(result_t4['mean'].values[:-10])
    t4_bm = np.mean(result_b4['mean'].values[:-10])

    t3_tr = np.std(result_t3['mean'].values[:-10])
    t3_br = np.std(result_b3['mean'].values[:-10])
    t4_tr = np.std(result_t4['mean'].values[:-10])
    t4_br = np.std(result_b4['mean'].values[:-10])

    h, (cx1) = plt.subplots(1, 1)
    result_t3.plot.scatter(x='x', y='mean', yerr='sem', ax=cx1,
        color='black')
    result_b3.plot.scatter(x='x', y='mean', yerr='sem', ax=cx1)
        
    #plt.title('Alpha Sum Q vs Time in Ch. 3 (profile)')
    #plt.legend(loc='best')
    cx1.set_ylim(0,5E7)
    cx1.set_xlabel('Time (s)')
    cx1.set_ylabel('Detected Charge (q)')
    #cx1.legend_.remove()
    h.savefig('tpc3_gainstability.pdf')
    plt.show()

    h, (cx1) = plt.subplots(1, 1)
    result_t4.plot.scatter(x='x', y='mean', yerr='sem', ax=cx1,
       color='black')
    result_b4.plot.scatter(x='x', y='mean', yerr='sem', ax=cx1)
    #bx2.set_title('Alpha Sum Q vs Time in TPC4 (profile)')
    #bx2.legend(loc='lower left')
    cx1.set_ylim(0,5E7)
    cx1.set_xlabel('Time (s)')
    cx1.set_ylabel('Detected Charge (q)')
    #cx1.legend_.remove()
    h.savefig('tpc4_gainstability.pdf')
    plt.show()

def pid_study(datapath, simpath):
    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    branches = ['neutron', 'theta', 'phi', 'de_dx', 'e_sum', 'tot_sum',
            'detnb','hitside', 't_length', 'npoints']

    ### Populate data arrays
    data_hitsides = []
    data_tlengths = []
    data_sumQ = []
    data_detnbs = []
    data_npoints = []

    for subdir, dirs, files in os.walk(datapath):
        for f in files:
            strs = f.split('_')
            if int(strs[-2]) not in good_files : continue
            r_file = str(subdir) + str('/') + str(f)
            print(r_file)
            data = root2rec(r_file, branches=branches)

            data_hitsides = np.concatenate([data_hitsides, data.hitside])
            data_tlengths = np.concatenate([data_tlengths, data.t_length])
            data_sumQ = np.concatenate([data_sumQ, data.e_sum])
            data_detnbs = np.concatenate([data_detnbs, data.detnb])
            data_npoints = np.concatenate([data_npoints, data.npoints])

    gain1 = 30.0
    gain2 = 50.0
    W = 35.075

    data_sumQ[data_detnbs == 4] *= 1.43
    data_sumE = (data_sumQ / (gain1 * gain2)) * W * 1E-3
    data_dQdx = data_sumQ/data_tlengths
    data_dQdx[data_tlengths==0] = 0

    ### Populate simulation arrays
    names = ['TPC3_Touschek_LER',
             'TPC4_Touschek_LER',
             'TPC3_Touschek_HER',
             'TPC4_Touschek_HER',
             'TPC3_Coulomb_LER',
             'TPC4_Coulomb_LER',
             'TPC3_Coulomb_HER',
             'TPC4_Coulomb_HER',
             'TPC3_Brems_LER',
             'TPC4_Brems_LER',
             'TPC3_Brems_HER',
             'TPC4_Brems_HER']

    sig_sim_hitsides = []
    sig_sim_tlengths = []
    sig_sim_sumQ = []
    sig_sim_detnbs = []
    sig_sim_npoints = []

    bak_sim_hitsides = []
    bak_sim_tlengths = []
    bak_sim_sumQ = []
    bak_sim_detnbs = []
    bak_sim_npoints = []

    for name in names:
        sigsim_file = str(simpath) + str(name) + str('.root')
        sig_sim = root2rec(sigsim_file, branches=branches)

        sig_sim_hitsides = np.concatenate([sig_sim_hitsides, sig_sim.hitside])
        sig_sim_tlengths = np.concatenate([sig_sim_tlengths, sig_sim.t_length])
        sig_sim_sumQ = np.concatenate([sig_sim_sumQ, sig_sim.e_sum])
        sig_sim_detnbs = np.concatenate([sig_sim_detnbs, sig_sim.detnb+1])
        sig_sim_npoints = np.concatenate([sig_sim_npoints, sig_sim.npoints])

        baksim_file = str(simpath) + str(name) + str('_bak.root')
        bak_sim = root2rec(baksim_file, branches=branches)

        bak_sim_hitsides = np.concatenate([bak_sim_hitsides, bak_sim.hitside])
        bak_sim_tlengths = np.concatenate([bak_sim_tlengths, bak_sim.t_length])
        bak_sim_sumQ = np.concatenate([bak_sim_sumQ, bak_sim.e_sum])
        bak_sim_detnbs = np.concatenate([bak_sim_detnbs, bak_sim.detnb+1])
        bak_sim_npoints = np.concatenate([bak_sim_npoints, bak_sim.npoints])

    sig_sim_sumQ[sig_sim_detnbs==4] *= (4./3.)
    sig_sim_dQdx = sig_sim_sumQ/sig_sim_tlengths
    sig_sim_dQdx[sig_sim_tlengths==0] = 0

    bak_sim_sumQ[bak_sim_detnbs==4] *= (4./3.)
    bak_sim_dQdx = bak_sim_sumQ/bak_sim_tlengths
    bak_sim_dQdx[bak_sim_tlengths==0] = 0

    f = plt.figure()
    ax1=f.add_subplot(111)

    labels=['MC Bkg','MC Signal']
    ax1.hist([bak_sim_npoints, sig_sim_npoints], bins=100,
            stacked=True, label=labels)
    ax1.hist(data_npoints, bins=100, stacked=True,
             histtype='step', label='Data')
    ax1.legend(loc='best')
    ax1.set_xlabel('Npoints (all)')
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1E0)
    #f.savefig('pid_sels_all.pdf')

    g = plt.figure()
    ax2=g.add_subplot(111)

    ax2.hist([
              bak_sim_hitsides[( (bak_sim_hitsides==0)  )],
              sig_sim_hitsides[( (sig_sim_hitsides==0)  )]
              ],
              bins=100, stacked=True, label=labels)
    ax2.hist(data_hitsides[( (data_hitsides==0) )], bins=100, stacked=True,
             histtype='step', label='Data')
    ax2.legend(loc='best')
    ax2.set_xlabel('Edge Cut')
    ax2.set_yscale('log')
    ax2.set_ylim(ymin=1E0)
    #g.savefig('pid_sels_edge.pdf')

    h = plt.figure()
    ax3=h.add_subplot(111)
    ax3.hist([
              bak_sim_dQdx[( (bak_sim_hitsides==0)
                            #& (bak_sim_dQdx>500.0) 
                            )],
              sig_sim_dQdx[( (sig_sim_hitsides==0) 
                            #& (sig_sim_dQdx>500.0) 
                            )]
              ],
              bins=100, stacked=True, label=labels)
    ax3.hist(data_dQdx[( (data_hitsides==0) 
                        #& (data_dQdx>500.0)
                        )],
             bins=100, stacked=True, histtype='step', label='Data')
    ax3.legend(loc='best')
    ax3.set_xlabel('dQ/dx (edge cut)')
    ax3.set_yscale('log')
    ax3.set_ylim(ymin=1E0)
    h.savefig('pid_sels_edge_dQdx.pdf')

    l = plt.figure()
    ax4=l.add_subplot(111)
    ax4.hist([
              bak_sim_npoints[( (bak_sim_hitsides==0) 
                                & (bak_sim_dQdx>500.0)
                                #& (bak_sim_npoints>40)
                                )],
              sig_sim_npoints[( (sig_sim_hitsides==0) 
                                & (sig_sim_dQdx>500.0)
                                #& (sig_sim_npoints>40) 
                                )]
              ],
              bins=100, stacked=True, label=labels)
    ax4.hist(data_npoints[( (data_hitsides==0) 
                            & (data_dQdx>500.0)
                            #& (data_npoints>40) 
                            )], bins=100, stacked=True,
                        histtype='step', label='Data')

    ax4.legend(loc='best')
    ax4.set_xlabel('Npoints (edge + dQdx cuts)')
    ax4.set_yscale('log')
    ax4.set_ylim(ymin=1E0)
    l.savefig('pid_sels_edge_dQdx_npoints.pdf')

    print('Printing background event info.......')
    print(len(bak_sim_npoints ) )

    print(len(bak_sim_npoints[( (bak_sim_hitsides==0) )] ) )

    print(len(bak_sim_npoints[( (bak_sim_hitsides==0) & (bak_sim_dQdx>500.0)
            )] ) )

    print(len(bak_sim_npoints[( (bak_sim_hitsides==0) & (bak_sim_dQdx>500.0)
            & (bak_sim_npoints>40) )] ) )

    print('Number of signal with maximum bak dQdx cut:', len(sig_sim_dQdx[(
        (sig_sim_hitsides==0) & (sig_sim_dQdx>612.0))] ) )

    print('Eff of tight dQdx cut vs only edge cut:', len(sig_sim_dQdx[(
        (sig_sim_hitsides==0) & (sig_sim_dQdx>612.0) )] ) / len(sig_sim_dQdx[sig_sim_hitsides == 0]))

    print(len(bak_sim_dQdx[( (bak_sim_hitsides==0) & (bak_sim_npoints < 40)
        )]))



    print('Printing signal event info.......')
    print(len(sig_sim_npoints))

    print(len(sig_sim_npoints[( (sig_sim_hitsides==0)  )]))

    print(len(sig_sim_npoints[( (sig_sim_hitsides==0) & (sig_sim_dQdx>500.0)
        )] ))

    print(len(sig_sim_npoints[( (sig_sim_hitsides==0) & (sig_sim_dQdx>500.0)
        & (sig_sim_npoints>40) )] ))





    plt.show()

    '''
    plt.scatter(tlengths_all, energies_all, facecolors='none', edgecolors='red')
    plt.scatter(tlengths_n, energies_n,color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc_dedx_cuts_combined.pdf')
    plt.show()

    plt.scatter(ch3_tlengths_all, ch3_energies_all, facecolors='none', edgecolors='red')
    plt.scatter(ch3_tlengths_n, ch3_energies_n, color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_cuts.pdf')
    plt.show()

    plt.scatter(ch4_tlengths_all, ch4_energies_all, facecolors='none', edgecolors='red')
    plt.scatter(ch4_tlengths_n, ch4_energies_n, color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_cuts_corrected.pdf')
    plt.show()
    '''

def event_inspection(datapath):
    ### Data
    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    n_neutrons = 0

    phis = []
    x_bins = 80
    y_bins = 336
    branches = [
                'hitside', 
                'de_dx', 
                 'col', 
                 'row', 
                 'tot', 
                 'neutron', 
                 'phi',
                 'npoints',
                 't_length',
                 'min_ret',
                 #'pdg'
                 ]
    for subdir, dirs, files in os.walk(datapath):
        for f in files:
            strs = f.split('_')
            
            ### Data
            if int(strs[-2]) not in good_files : continue #For data files
            #if '.root' not in f or 'bak' in f: continue
            #if '10hr' not in f: continue
            r_file = str(subdir) + str('/') + str(f)
            print(r_file)
            data = root2rec(r_file, branches=branches)
            #n_neutrons = sum(data.neutron)
            #grid = int(np.sqrt(n_neutrons)) + 1
            #n_events = (
            #        (data.hitside == 0) 
            #        & (data.de_dx > 500.0) 
            #        & (data.phi > 85.0)
            #        & (data.phi < 95.0)
            #        & (data.npoints > 40)
            #        ).sum()

            n_events = (
                    (data.hitside == 0) 
                    & (data.min_ret == 0)
                    & (data.de_dx > 500.0) 
                    & (data.npoints > 40)
                    #& (data.pdg > 10000)
                    ).sum()
            grid = int(np.sqrt(n_events)) + 1
            print(n_events, grid)
            #input('well?')
            if n_events == 0 : continue
            fig, axs = plt.subplots(grid, grid)
            counter = 0
            for event in data:
                if (event.hitside == 0
                        and event.min_ret == 0
                        and event.de_dx > 500.0
                        and event.npoints > 40) :
                        #and event.pdg > 10000) :
                #if (event.neutron == 1 and event.de_dx > 500.0 
                #        and event.phi >85.0 and event.phi < 95.0
                #        and event.npoints>40):
                #if event.hitside == 0 and event.de_dx > 0.35 :
                    #plt.hist2d(event.col, event.row, 
                    #plt.hist2d(event.col, event.row, bins = (
                    #    range(0, x_bins, 1) , range(0, y_bins, 1) ), weights = event.tot + 1)
                    axs.flat[counter].hist2d( event.col, event.row, bins = (
                            range(0, x_bins, 1) , range(0, y_bins, 1) ),
                            weights = event.tot, cmin = -1, cmax = 15,
                            cmap=('YlOrBr'))
                    axs.flat[counter].set_frame_on(False)
                    axs.flat[counter].get_yaxis().set_visible(False)
                    axs.flat[counter].get_xaxis().set_visible(False)
                    axs.flat[counter].set_xlim(0, x_bins)
                    axs.flat[counter].set_ylim(0, y_bins)
                    #axs.flat[counter].pcolorfast('YlOrBr')
                    cbar = mpl.colorbar.ColorbarBase(axs.flat[counter],
                            cmap='YlOrBr',
                            norm=mpl.colors.Normalize(vmin=-1, vmax = 15),
                            )
                    cbar.set_clim(0,15)
                    counter += 1
            #plt.set_frame_on(False)
            #plt.get_yaxis().set_visible(False)
            #plt.get_xaxis().set_visible(False)
            #plt.set_xlim(0, x_bins)
            #plt.set_ylim(0, y_bins)

                    #print(event.event)
                    #plt.xlim(0, x_bins)
                    #plt.ylim(0, y_bins)
                    #plt.colorbar()
                    #plt.show()
            #fig.colorbar(axs)
            diff = grid**2 - n_events
            for i in range(1,diff+1):
                axs.flat[-i].set_frame_on(False)
                axs.flat[-i].get_yaxis().set_visible(False)
                axs.flat[-i].get_xaxis().set_visible(False)
                
            #fig.colorbar(_, cax=axs)
            
            #plt.colorbar()
            fname = f.split('.')[0]
            pname = fname + str('.pdf')
            print(pname)
            fig.savefig(pname)
    #print(n_neutrons)
    #input('well?')

def compare_toushek(datapath, simpath):
    # Get rate vs beamsize from BEAST data ntuples
    data_toushek = neutron_rate_data(datapath)
    print('Printing rates and subrun durations from data ... ')
    print(data_toushek)

    # Get rate vs beamsize from BEAST sim ntuples
    sim_toushek = neutron_rate_sim(simpath)

    # Reweight raw sim for third rate vs beamsize measurement
    rawSimPath = '/Users/BEASTzilla/BEAST/sim/sim_refitter/v5.2/FTFP_BERT_HP/'
    sim_angles = neutron_study_sim(rawSimPath)
    #sim_angles = neutron_rate_raw_sim(rawSimPath)

    ch3_touschek_unweighted_rate = len(sim_angles[8][0])
    ch3_beamgas_unweighted_rate = len(sim_angles[8][1])

    ch4_touschek_unweighted_rate = len(sim_angles[9][0])
    ch4_beamgas_unweighted_rate = len(sim_angles[9][1])

    print('Printing number of neutrons of each type and channel from raw sim ... ')
    print('Ch3 Touschek is:', ch3_touschek_unweighted_rate)
    print('Ch3 Beam Gas is:', ch3_beamgas_unweighted_rate)
    print('Ch4 Touschek is:', ch4_touschek_unweighted_rate)
    print('Ch4 Beam Gas is:', ch4_beamgas_unweighted_rate)

    #print('Printing raw touschek rates for chs 3 and 4')
    #print(ch3_touschek_unweighted_rate)
    #print(ch4_touschek_unweighted_rate)
    #input('well?')

    subrun_times, subrun_BeamGas, subrun_Touschek, TouschekPlot_vals = calc_sim_weights(datapath, simpath)
    #exp_IPZ2 = TouschekPlot_vals[1]
    exp_IPZ2 = subrun_BeamGas

    ch3_weights= [0,0]

    ch3_weights[0] = subrun_Touschek *(
            (ch3_touschek_unweighted_rate)/
                    (36000.0*9090.91) )

    ch3_weights[1] = subrun_BeamGas *  (
            (ch3_beamgas_unweighted_rate)/
                    (36000.0*0.0097) ) 

    ch3_weights = np.array(ch3_weights)

    ch3_weighted_rates = np.array(ch3_weights[0]+ch3_weights[1])

    ch3_weighted_xvals = TouschekPlot_vals[0]

    #ch3_weighted_xvals = np.array(ch3_weighted_xvals)

    ch4_weights=[0,0]

    ch4_weights[0] = subrun_Touschek * (
            (ch4_touschek_unweighted_rate)/
                    (36000.0*9090.91) )

    ch4_weights[1] = subrun_BeamGas * (
            (ch4_beamgas_unweighted_rate)/
                    (36000.0*0.0097) ) 

    ch4_weighted_xvals = TouschekPlot_vals[0]

    #data_toushek = np.array(data_toushek)
    #sim_toushek = np.array(sim_toushek)

    ch3_data_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch3_weighted_xvals,
            y=(data_toushek[0]/exp_IPZ2),
            #y=(data_toushek[0]+data_toushek[2]),
            #error=(data_toushek[0]/(exp_IPZ2*np.sqrt(data_toushek[1])))
            error=np.sqrt((data_toushek[0]/exp_IPZ2)),
            #error=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1]))
            )
    ch3_data_minu = iminuit.Minuit(ch3_data_chi2)
    ch3_data_minu.migrad()

    ch4_data_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch4_weighted_xvals,
            y=(data_toushek[2]/exp_IPZ2),
            #y=(data_toushek[0]+data_toushek[2]),
            #error=(data_toushek[2]/(exp_IPZ2*np.sqrt(data_toushek[1])))
            error=(data_toushek[2]/exp_IPZ2)/np.sqrt(subrun_times)
            #error=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1]))
            )
    ch4_data_minu = iminuit.Minuit(ch4_data_chi2)
    ch4_data_minu.migrad()

    print('Calculating sensitivies for ch. 3 in weighted sim ...\n')
    ch3_weighted_sim_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch3_weighted_xvals,
            #y=(weighted_rates),
            y=(ch3_weighted_rates)/exp_IPZ2,
            #error=(weighted_rates)/(np.sqrt(sim_toushek[1]))
            #error=(ch3_weighted_rates)/(exp_IPZ2*np.sqrt(ch3_weighted_rates))
            error=(ch3_weighted_rates)/exp_IPZ2/np.sqrt(subrun_times)
            )
    ch3_weighted_sim_minu = iminuit.Minuit(ch3_weighted_sim_chi2)
    ch3_weighted_sim_minu.migrad()


    ch4_weighted_rates = np.array(ch4_weights[0]+ch4_weights[1])
    ch4_weighted_rates = (ch3_weighted_rates *
                    (ch4_touschek_unweighted_rate/ch3_touschek_unweighted_rate) )
    print('Calculating sensitivies for ch. 4 in weighted sim ...\n')
    ch4_weighted_sim_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch4_weighted_xvals,
            #y=(weighted_rates),
            y=(ch4_weighted_rates)/exp_IPZ2,
            #error=(weighted_rates)/(np.sqrt(sim_toushek[1]))
            #error=(ch4_weighted_rates)/(exp_IPZ2*np.sqrt(ch4_weighted_rates))
            error=(ch4_weighted_rates)/exp_IPZ2/np.sqrt(subrun_times)
            )
    ch4_weighted_sim_minu = iminuit.Minuit(ch4_weighted_sim_chi2)
    ch4_weighted_sim_minu.migrad()

    print('Calculating sensitivies for ch. 3 in data ...\n')
    ch3_data_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch3_weighted_xvals,
            y=(data_toushek[0]/exp_IPZ2),
            #y=(data_toushek[0]+data_toushek[2]),
            #error=data_toushek[0]/(exp_IPZ2*np.sqrt(data_toushek[1]))
            error=(data_toushek[0]/exp_IPZ2)/np.sqrt(subrun_times)
            #error=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1]))
            )
    ch3_data_minu = iminuit.Minuit(ch3_data_chi2)
    ch3_data_minu.migrad()

    print('Calculating sensitivies for ch. 4 in data ...\n')
    ch4_data_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch4_weighted_xvals,
            y=(data_toushek[2]/exp_IPZ2),
            #y=(data_toushek[0]+data_toushek[2]),
            #error=data_toushek[2]/(exp_IPZ2*np.sqrt(data_toushek[3]))
            error=(data_toushek[2]/exp_IPZ2)/np.sqrt(subrun_times)
            #error=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1]))
            )
    ch4_data_minu = iminuit.Minuit(ch4_data_chi2)
    ch4_data_minu.migrad()

    total_data_y=(data_toushek[0]+data_toushek[2])/exp_IPZ2
    print('Calculating combined sensitivites in data ...\n')
    data_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch3_weighted_xvals,
            y=(data_toushek[0]+data_toushek[2])/exp_IPZ2,
            #y=total_data_y,
            #y=(data_toushek[0]+data_toushek[2]),
            #error=(data_toushek[0]+data_toushek[2])/(exp_IPZ2*np.sqrt(data_toushek[1]+data_toushek[3]))
            error=(data_toushek[0]+data_toushek[2])/exp_IPZ2/np.sqrt(subrun_times)
            #error=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1]))
            )
    data_minu = iminuit.Minuit(data_chi2)
    data_minu.migrad()

    total_weighted_rate = ch3_weighted_rates + ch4_weighted_rates
    print('Calculating combined senstivities in weighted sim ... \n')
    print('Calculating error for total weighted sim rate ... ')

    print(ch3_weighted_rates/exp_IPZ2)
    print(np.sqrt(ch3_weighted_rates))
    print(ch3_weighted_rates/(exp_IPZ2*np.sqrt(ch3_weighted_rates)))

    print(total_weighted_rate/exp_IPZ2)
    print(np.sqrt(total_weighted_rate))
    print(total_weighted_rate/(exp_IPZ2*np.sqrt(total_weighted_rate)))

    sim_chi2 = probfit.Chi2Regression(probfit.linear,
            x=ch3_weighted_xvals,
            y=(total_weighted_rate)/exp_IPZ2,
            #y=(sim_toushek[0]+sim_toushek[2]),
            #error=(sim_toushek[0]+sim_toushek[2])/(np.sqrt(sim_toushek[1]))
            #error=(total_weighted_rate)/(exp_IPZ2*np.sqrt(total_weighted_rate))
            error=total_weighted_rate/exp_IPZ2/np.sqrt(subrun_times)
            )
    sim_minu = iminuit.Minuit(sim_chi2)
    sim_minu.migrad()

    #weighted_rates = ch3_weighted_rates + ch4_weighted_rates

    #weighted_sim_chi2 = probfit.Chi2Regression(probfit.linear,
    #        x=ch3_weighted_xvals,
    #        #y=(weighted_rates),
    #        y=(weighted_rates)/exp_IPZ2,
    #        #error=(weighted_rates)/(np.sqrt(sim_toushek[1]))
    #        error=(weighted_rates)/(exp_IPZ2*np.sqrt(sim_toushek[1]))
    #        )
    #weighted_sim_minu = iminuit.Minuit(weighted_sim_chi2)
    #weighted_sim_minu.migrad()

    g = plt.figure()
    ax0 = g.add_subplot(111)
    data_chi2.draw(data_minu, print_par=False)
    sim_chi2.draw(sim_minu, print_par=False)
    ch3_weighted_sim_chi2.draw(ch3_weighted_sim_minu, print_par=False)
    ch4_weighted_sim_chi2.draw(ch4_weighted_sim_minu, print_par=False)
    ch3_data_chi2.draw(ch3_data_minu, print_par=False)
    ch4_data_chi2.draw(ch4_data_minu, print_par=False)
    ax0.set_xlim(0,plt.xlim()[1])
    ax0.set_ylim(0,plt.ylim()[1])
    ax0.set_xlabel('$\\frac{I}{P\sigma_yZ_{eff}^{2}}\ [mA^{-1}\ Pa^{-1}\ \mu m^{-1}]$\t',
           ha='right', x=1.0)
    ax0.set_ylabel('$\\frac{Rate}{IPZ_{eff}^{2}}\ [Hz\ mA^{-1}\ Pa^{-1}]$', ha='right', y=1.0)
    from matplotlib.ticker import ScalarFormatter
    ax0.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax0.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax0.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    f = plt.figure()
    ax1 = f.add_subplot(111)


    #ax1.errorbar(
    #            ch3_weighted_xvals,
    #            (data_toushek[0]+data_toushek[2])/exp_IPZ2,
    #            yerr=np.sqrt((data_toushek[0]+data_toushek[2])/exp_IPZ2/np.sqrt(subrun_times)),
    #            fmt='o',
    #            color='C0',
    #            label='Total Exp.')

    #ax1.errorbar(
    #            ch3_weighted_xvals,
    #            (total_weighted_rate)/exp_IPZ2,
    #            yerr=total_weighted_rate/exp_IPZ2/np.sqrt(subrun_times),
    #            fmt='o',
    #            color='C1',
    #            label='Total MC')

    ax1.errorbar(
                ch3_weighted_xvals,
                (data_toushek[0]/exp_IPZ2),
                yerr=(data_toushek[0]/exp_IPZ2)/np.sqrt(subrun_times),
                fmt='o',
                ms=5.8,
                color='C0',
                label='Ch. 3 Exp')

    ax1.errorbar(
                ch3_weighted_xvals,
                (ch3_weighted_rates)/exp_IPZ2,
                yerr=ch3_weighted_rates/exp_IPZ2/np.sqrt(subrun_times),
                fmt='^',
                ms=5.8,
                #color='C0',
                mec='C0',
                mfc='none',
                label='Ch. 3 MC')

    ax1.errorbar(
                ch4_weighted_xvals,
                (data_toushek[2]/exp_IPZ2),
                yerr=(data_toushek[2]/exp_IPZ2)/np.sqrt(subrun_times),
                fmt='o',
                ms=5.8,
                color='C1',
                label='Ch. 4 Exp')

    ax1.errorbar(
                ch4_weighted_xvals,
                (ch4_weighted_rates)/exp_IPZ2,
                yerr=ch4_weighted_rates/exp_IPZ2/np.sqrt(subrun_times),
                fmt='^',
                ms=5.8,
                #color='C1',
                mec='C1',
                mfc='none',
                label='Ch. 4 MC')

    #((data_x, data_y), _, (data_total_pdf_x, data_total_pdf_y), _) = (
    #        data_chi2.draw(data_minu, print_par=False, no_plot=True) )
    #parameters = data_minu.values
    #data_total_pdf_y[-1] = parameters['c']
    #data_total_pdf_x[-1] = 0.0
    #ax1.plot(data_total_pdf_x, data_total_pdf_y, lw=2)#, label='Total Exp. Data')

    #((sim_x, sim_y), _, (sim_total_pdf_x, sim_total_pdf_y), _) = (
    #        sim_chi2.draw(sim_minu, print_par=False, no_plot=True))
    #parameters = sim_minu.values
    #sim_total_pdf_y[-1] = parameters['c']
    #sim_total_pdf_x[-1] = 0.0
    #ax1.plot(sim_total_pdf_x, sim_total_pdf_y, lw=2)#, label='Total Wtd. MC')

    ((ch3_weighted_sim_x, ch3_weighted_sim_y), _, (ch3_sim_pdf_x, ch3_sim_pdf_y), _) = (
            ch3_weighted_sim_chi2.draw(ch3_weighted_sim_minu, print_par=False,
                no_plot=True))
    parameters = ch3_weighted_sim_minu.values
    ch3_sim_pdf_y[-1] = parameters['c']
    ch3_sim_pdf_x[-1] = 0.0
    ax1.plot(ch3_sim_pdf_x, ch3_sim_pdf_y, lw=2, color='C0')#, label='Ch. 3 Wtd. MC')

    ((ch4_weighted_sim_x, ch4_weighted_sim_y), _, (ch4_sim_pdf_x, ch4_sim_pdf_y),
            _) = (ch4_weighted_sim_chi2.draw(ch4_weighted_sim_minu,
                        print_par=False, no_plot=True) )
    parameters = ch4_weighted_sim_minu.values
    ch4_sim_pdf_y[-1] = parameters['c']
    ch4_sim_pdf_x[-1] = 0.0
    ax1.plot(ch4_sim_pdf_x, ch4_sim_pdf_y, lw=2, color='C1')#, label='Ch. 4 Wtd. MC')

    ((ch3_data_x, ch3_data_y), _, (ch3_data_pdf_x, ch3_data_pdf_y), _) = (
            ch3_data_chi2.draw(ch3_data_minu, print_par=False,
                no_plot=True))
    parameters = ch3_data_minu.values
    ch3_data_pdf_y[-1] = parameters['c']
    ch3_data_pdf_x[-1] = 0.0
    ax1.plot(ch3_data_pdf_x, ch3_data_pdf_y, lw=2, color='C0')#, label='Ch. 3 Exp. Data')

    ((ch4_data_x, ch4_data_y), _, (ch4_data_pdf_x, ch4_data_pdf_y), _) = (
            ch4_data_chi2.draw(ch4_data_minu, print_par=False, no_plot=True) )
    parameters = ch4_data_minu.values
    ch4_data_pdf_y[-1] = parameters['c']
    ch4_data_pdf_x[-1] = 0.0
    ax1.plot(ch4_data_pdf_x, ch4_data_pdf_y, lw=2, color='C1')#, label='Ch. 4 Exp. Data')

    ax1.set_xlim(0,plt.xlim()[1])
    ax1.set_ylim(0,plt.ylim()[1])
    ax1.set_xlabel('$\\frac{I}{P\sigma_yZ_{eff}^{2}}\ [mA^{-1}\ Pa^{-1}\ \mu m^{-1}]$\t',
           ha='right', x=1.0)
    ax1.set_ylabel('$\\frac{Rate}{IPZ_{eff}^{2}}\ [Hz\ mA^{-1}\ Pa^{-1}]$', ha='right', y=1.0)
    from matplotlib.ticker import ScalarFormatter
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.legend(loc='best')
    #ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    f.savefig('tpc_heuristic_bg_vs_t.pdf')

    plt.show()

    '''
    f = plt.figure()
    ax1 = f.add_subplot(111)
    #data_toushek[4].draw(data_toushek[5], print_par=False)
    #sim_toushek[4].draw(sim_toushek[5], print_par=False)
    data_chi2.draw(data_minu, print_par=False)
    sim_chi2.draw(sim_minu, print_par=False)
    weighted_sim_chi2.draw(weighted_sim_minu, print_par=False)

    ax1.errorbar(weighted_xvals, 
            #(data_toushek[0]+data_toushek[2]),
            (data_toushek[0]+data_toushek[2])/exp_IPZ2,
            #xerr=data_toushek[2],
            #yerr=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1])),
            yerr=(data_toushek[0]+data_toushek[2])/(exp_IPZ2*np.sqrt(data_toushek[1])),
            fmt='o', color='black', label='Experiment')
    ax1.errorbar(weighted_xvals, 
            #(sim_toushek[0]+sim_toushek[2]),
            (sim_toushek[0]+sim_toushek[2])/exp_IPZ2,
            #xerr=data_toushek[2],
            yerr=(sim_toushek[0]+sim_toushek[2])/(exp_IPZ2*np.sqrt(sim_toushek[1])),
            #yerr=((sim_toushek[0]+sim_toushek[2])/(np.sqrt(sim_toushek[1]))),
            fmt='o', color='C0', label='Simulation')
    #ax1.errorbar(TouschekPlot_vals[0], 
    #        weighted_rates,
    #        #xerr=data_toushek[2],
    #        #yerr=weighted_rates, 
    #        fmt='o', color='C2', label='Simulation')
    ax1.errorbar(weighted_xvals,
            weighted_rates/exp_IPZ2,
            yerr=weighted_rates/(exp_IPZ2*np.sqrt(data_toushek[1])),
            #yerr=weighted_rates/(np.sqrt(data_toushek[1])),
            color='C2', fmt='o', label='Self Weighted')

    ax1.set_xlabel('$\\frac{I}{P\sigma_y}$ [$mA$ Pa$^{-1}$ $\mu$m$^{-1}$]\t\t',
                   ha='right', x=1.0)
    ax1.set_ylabel('$\\frac{Rate}{IPZ_{eff}^{2}}$', ha='right', y=1.0)
    #ax1.set_xlim([0.0,1.0E7])
    #ax1.set_ylim([0.0,80.0])
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    xfmt = mpl.ticker.ScalarFormatter(useMathText=True)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.legend(loc='best')
    #f.savefig('TPC_peter_toushek_measurement_simcuts_sim_and_data.pdf')
    #f.savefig('TPC_peter_touschek_dataVSmc_igal_weighting.pdf')

    print('Printing y vals and ratios for data and Igal-weighted sim ... ')
    print((data_toushek[0]+data_toushek[2])/exp_IPZ2)
    print((sim_toushek[0]+sim_toushek[2])/exp_IPZ2)

    print(((data_toushek[0]+data_toushek[2])) /
             ((sim_toushek[0]+sim_toushek[2])) )
    
    g = plt.figure()
    ax2 = g.add_subplot(111)
    ax2.errorbar(weighted_xvals,
            data_toushek[0]+data_toushek[2],
            yerr=(data_toushek[0]+data_toushek[2])/(np.sqrt(data_toushek[1])),
            color='k', fmt='o', label='Experiment')
    ax2.scatter(weighted_xvals,
            sim_toushek[0]+sim_toushek[2],
            color='C0', label='Simulation')
    ax2.scatter(weighted_xvals,
            weighted_rates,
            color='C2', label='Self Weighted')
    '''


    plt.show()




def compare_angles(datapath, simpath):
    #data_angles = neutron_study_raw(datapath)
    data_angles = neutron_angles_data(datapath)
    sim_angles = neutron_study_sim(simpath)
    print('\nPassed angle arrays to plotting function:')
    #print(sim_angles[0])
    #print(sim_angles[1])
    #print(sim_angles[2])
    #print(sim_angles[3])
    #print(sim_angles[4])
    #print(sim_angles[5])
    #print(sim_angles[6])
    #print(sim_angles[7])
    print('Sim angles:', np.sum(sim_angles[0]), np.sum(sim_angles[1]),
            np.sum(sim_angles[2]), np.sum(sim_angles[3]))
    print('Data angles:', len(data_angles[0]), len(data_angles[1]),
            len(data_angles[2]), len(data_angles[3]))

    theta_bins = 9
    phi_bins = 9

    phis = np.arange(-90,90,20)
    thetas = np.arange(0,180,20)


    ### Attempt to fit for Touschek/Beamgas components from sim, and scale to
    ### experimental data

    print('\nNumber of entries in each Ch3 Theta Histogram ..')
    print('SIM: Touschek: %i  BeamGas: %i' % (len(sim_angles[8][0]),
        len(sim_angles[8][1]) ) )
    print('DATA:', len(data_angles[0]))
    ch3Theta_Touschek_hist = np.histogram(sim_angles[8][0], bins=9,
            range=[0,180])

    ch3Theta_TouschekPDF = probfit.pdf.HistogramPdf(
                                ch3Theta_Touschek_hist[0],
                                binedges=ch3Theta_Touschek_hist[1]   
                                )
    ch3Theta_TouschekPDF = probfit.Extended(ch3Theta_TouschekPDF)

    ch3Theta_BeamGas_hist = np.histogram(sim_angles[8][1], bins=9,
            range=[0,180])

    # Check histogram bins
    ch3Theta_BeamGasPDF = probfit.pdf.HistogramPdf(
                                ch3Theta_BeamGas_hist[0],
                                binedges=ch3Theta_BeamGas_hist[1]   
                                )
    ch3Theta_BeamGasPDF = probfit.Extended(ch3Theta_BeamGasPDF)

    ch3Theta_bkgPDF = probfit.functor.AddPdf(ch3Theta_TouschekPDF,
                                             ch3Theta_BeamGasPDF,
                                             prefix=['Touschek', 'BeamGas']
                                             )

    ch3Theta_chi2 = probfit.BinnedChi2(ch3Theta_bkgPDF,
                                       #np.concatenate([sim_angles[8][0],
                                       #    sim_angles[8][1]]),
                                       data_angles[0],
                                       #bins=ch3Theta_Touschek_hist[1],
                                       bins=9,bound=(0,180),
                                       )

    ch3Theta_minu = iminuit.Minuit(ch3Theta_chi2)
    ch3Theta_minu.migrad()
    parameters = ch3Theta_minu.values
    ch3Theta_chi2.draw(parts=True, print_par=False)
    ((data_edges, data_y), (errorp,errorm), (total_pdf_x, total_pdf_y), parts) = (
            ch3Theta_chi2.draw(parts=True, print_par=False, no_plot=True))

    data_x = 0.5 * (data_edges[:-1] + data_edges[1:])

    Touschek_norm = np.sum(ch3Theta_Touschek_hist[0])
    BeamGas_norm = np.sum(ch3Theta_BeamGas_hist[0])

    plt.show()

    # Plot sum of Touschek and beam gas obtained from histogram PDF
    sum_hists = (ch3Theta_Touschek_hist[0] + ch3Theta_BeamGas_hist[0])
    sum_norm = np.sum(sum_hists)
    sum_fitted_norm = parameters['TouschekN'] + parameters['BeamGasN']

    plt.hist(thetas,
            weights=sum_hists*sum_fitted_norm/sum_norm,
            color='C2', bins=data_edges, label='MC Fitted Sum')

    # Plot beam gas distribution obtained from histogram PDF
    plt.hist(thetas,
            weights=ch3Theta_BeamGas_hist[0]*parameters['BeamGasN']/BeamGas_norm,
            color='C1', bins=data_edges, label='MC Beam Gas')

    # Plot Touschek distribution obtained from histogram PDF
    plt.hist(thetas,
            weights=ch3Theta_Touschek_hist[0]*parameters['TouschekN']/Touschek_norm,
            color='C0', bins=data_edges, label='MC Touschek')


    plt.errorbar(data_x, data_y, fmt='o', yerr=np.sqrt(data_y), color='k',
            label='Exp.')
    #plt.plot(total_pdf_x, total_pdf_y, color='blue', lw=2)
    #colors = ['orange', 'purple', 'DarkGreen']
    #labels = ['Background', 'Signal 1', 'Signal 2']
    #for color, label, part in zip(colors, parts):
    #    x, y = part
    #    plt.plot(x, y, ls='--', color=color)
    #plt.grid(True)

    plt.xlabel('Ch. 3 $\\theta$ [$^\circ$]', ha='right', x=1.0)
    plt.ylabel('Events per bin', ha='right', y=1.0)
    plt.legend(loc='best')
    plt.savefig('ch3_theta_histoPDF_fit.pdf')
    plt.show()


    print('\nNumber of entries in each Ch4 Theta Histogram ..')
    print('Touschek: %i  BeamGas: %i' % (len(sim_angles[9][0]),
        len(sim_angles[9][1]) ) )
    print('DATA:', len(data_angles[1]))
    ch4Theta_Touschek_hist = np.histogram(sim_angles[9][0], bins=9,
            range=[0,180])

    ch4Theta_TouschekPDF = probfit.pdf.HistogramPdf(
                                ch4Theta_Touschek_hist[0],
                                binedges=ch4Theta_Touschek_hist[1]   
                                )
    ch4Theta_TouschekPDF = probfit.Extended(ch4Theta_TouschekPDF)

    ch4Theta_BeamGas_hist = np.histogram(sim_angles[9][1], bins=9,
            range=[0,180])

    ch4Theta_BeamGasPDF = probfit.pdf.HistogramPdf(
                                ch4Theta_BeamGas_hist[0],
                                binedges=ch4Theta_BeamGas_hist[1]   
                                )
    ch4Theta_BeamGasPDF = probfit.Extended(ch4Theta_BeamGasPDF)

    ch4Theta_bkgPDF = probfit.functor.AddPdf(ch4Theta_TouschekPDF,
                                             ch4Theta_BeamGasPDF,
                                             prefix=['Touschek', 'BeamGas']
                                             )

    ch4Theta_chi2 = probfit.BinnedChi2(ch4Theta_bkgPDF,
                                       #np.concatenate([sim_angles[9][0],
                                       #    sim_angles[9][1]]),
                                       data_angles[1],
                                       bins=9,bound=(0,180),
                                       )

    ch4Theta_minu = iminuit.Minuit(ch4Theta_chi2)
    ch4Theta_minu.migrad()

    parameters = ch4Theta_minu.values
    ch4Theta_chi2.draw(parts=True, print_par=False)
    ((data_edges, data_y), (errorp,errorm), (total_pdf_x, total_pdf_y), parts) = (
            ch4Theta_chi2.draw(parts=True, print_par=False, no_plot=True))

    data_x = 0.5 * (data_edges[:-1] + data_edges[1:])

    Touschek_norm = np.sum(ch4Theta_Touschek_hist[0])
    BeamGas_norm = np.sum(ch4Theta_BeamGas_hist[0])

    plt.show()

    # Plot sum of Touschek and beam gas obtained from histogram PDF
    sum_hists = (ch4Theta_Touschek_hist[0] + ch4Theta_BeamGas_hist[0])
    sum_norm = np.sum(sum_hists)
    sum_fitted_norm = parameters['TouschekN'] + parameters['BeamGasN']

    plt.hist(thetas,
            weights=sum_hists*sum_fitted_norm/sum_norm,
            color='C2', bins=data_edges, label='MC Fitted Sum')

    # Plot beam gas distribution obtained from histogram PDF
    plt.hist(thetas,
            weights=ch4Theta_BeamGas_hist[0]*parameters['BeamGasN']/BeamGas_norm,
            color='C1', bins=data_edges, label='MC Beam Gas')

    # Plot Touschek distribution obtained from histogram PDF
    plt.hist(thetas,
            weights=ch4Theta_Touschek_hist[0]*parameters['TouschekN']/Touschek_norm,
            color='C0', bins=data_edges, label='MC Touschek')


    plt.errorbar(data_x, data_y, fmt='o', yerr=np.sqrt(data_y), color='k',
            label='Exp.')
    #plt.plot(total_pdf_x, total_pdf_y, color='blue', lw=2)
    #colors = ['orange', 'purple', 'DarkGreen']
    #labels = ['Background', 'Signal 1', 'Signal 2']
    #for color, label, part in zip(colors, parts):
    #    x, y = part
    #    plt.plot(x, y, ls='--', color=color)
    #plt.grid(True)

    plt.xlabel('Ch. 4 $\\theta$ [$^\circ$]', ha='right', x=1.0)
    plt.ylabel('Events per bin', ha='right', y=1.0)
    plt.legend(loc='best')
    plt.savefig('ch4_theta_histoPDF_fit.pdf')
    plt.show()


    print('\nNumber of entries in each Ch3 Phi Histogram ..')
    print('Touschek: %i  BeamGas: %i' % (len(sim_angles[10][0]),
        len(sim_angles[10][1]) ) )
    print('DATA:', len(data_angles[2]))
    ch3Phi_Touschek_hist = np.histogram(sim_angles[10][0], bins=9,
            range=[-90,90])

    ch3Phi_TouschekPDF = probfit.pdf.HistogramPdf(
                                ch3Phi_Touschek_hist[0],
                                binedges=ch3Phi_Touschek_hist[1]   
                                )
    ch3Phi_TouschekPDF = probfit.Extended(ch3Phi_TouschekPDF)

    ch3Phi_BeamGas_hist = np.histogram(sim_angles[10][1], bins=9,
            range=[-90,90])

    ch3Phi_BeamGasPDF = probfit.pdf.HistogramPdf(
                                ch3Phi_BeamGas_hist[0],
                                binedges=ch3Phi_BeamGas_hist[1]   
                                )
    ch3Phi_BeamGasPDF = probfit.Extended(ch3Phi_BeamGasPDF)

    ch3Phi_bkgPDF = probfit.functor.AddPdf(ch3Phi_TouschekPDF,
                                             ch3Phi_BeamGasPDF,
                                             prefix=['Touschek', 'BeamGas']
                                             )
    print(ch3Phi_Touschek_hist)
    print(ch3Phi_BeamGas_hist)
    data_hist = np.histogram(data_angles[2], bins=9, range=[-90,90])
    print(data_hist)
    ch3phis = data_angles[2]
    print(ch3phis[ch3phis>90])
    print(ch3phis[ch3phis<-90])
    #input('well?')

    ch3Phi_chi2 = probfit.BinnedChi2(ch3Phi_bkgPDF,
                                       #np.concatenate([sim_angles[10][0],
                                       #    sim_angles[10][1]]),
                                       data_angles[2],
                                       bins=9,bound=(-90,90),
                                       )

    ch3Phi_minu = iminuit.Minuit(ch3Phi_chi2)
    ch3Phi_minu.migrad()
    parameters = ch3Phi_minu.values
    ch3Phi_chi2.draw(parts=True, print_par=False)
    ((data_edges, data_y), (errorp,errorm), (total_pdf_x, total_pdf_y), parts) = (
            ch3Phi_chi2.draw(parts=True, print_par=False, no_plot=True))

    data_x = 0.5 * (data_edges[:-1] + data_edges[1:])

    Touschek_norm = np.sum(ch3Phi_Touschek_hist[0])
    BeamGas_norm = np.sum(ch3Phi_BeamGas_hist[0])

    plt.show()

    # Plot sum of Touschek and beam gas obtained from histogram PDF
    sum_hists = (ch3Phi_Touschek_hist[0] + ch3Phi_BeamGas_hist[0])
    sum_norm = np.sum(sum_hists)
    sum_fitted_norm = parameters['TouschekN'] + parameters['BeamGasN']

    plt.hist(phis,
            weights=sum_hists*sum_fitted_norm/sum_norm,
            color='C2', bins=data_edges, label='MC Fitted Sum')

    # Plot Touschek distribution obtained from histogram PDF
    plt.hist(phis,
            weights=ch3Phi_Touschek_hist[0]*parameters['TouschekN']/Touschek_norm,
            color='C0', bins=data_edges, label='MC Touschek')

    # Plot beam gas distribution obtained from histogram PDF
    plt.hist(phis,
            weights=ch3Phi_BeamGas_hist[0]*parameters['BeamGasN']/BeamGas_norm,
            color='C1', bins=data_edges, label='MC Beam Gas')


    plt.errorbar(data_x, data_y, fmt='o', yerr=np.sqrt(data_y), color='k')

    plt.xlabel('Ch. 3 $\\phi$ [$^\circ$]', ha='right', x=1.0)
    plt.ylabel('Events per bin', ha='right', y=1.0)
    plt.legend(loc='best')
    plt.savefig('ch3_phi_histoPDF_fit.pdf')
    plt.show()

    print('\nNumber of entries in each Ch4 Phi Histogram ..')
    print('Touschek: %i  BeamGas: %i' % (len(sim_angles[11][0]),
        len(sim_angles[11][1]) ) )
    print('DATA:', len(data_angles[3]))
    ch4Phi_Touschek_hist = np.histogram(sim_angles[11][0], bins=9,
            range=[-90,90])

    ch4Phi_TouschekPDF = probfit.pdf.HistogramPdf(
                                ch4Phi_Touschek_hist[0],
                                binedges=ch4Phi_Touschek_hist[1]   
                                )
    ch4Phi_TouschekPDF = probfit.Extended(ch4Phi_TouschekPDF)

    ch4Phi_BeamGas_hist = np.histogram(sim_angles[11][1], bins=9,
            range=[-90,90])

    ch4Phi_BeamGasPDF = probfit.pdf.HistogramPdf(
                                ch4Phi_BeamGas_hist[0],
                                binedges=ch4Phi_BeamGas_hist[1]   
                                )
    ch4Phi_BeamGasPDF = probfit.Extended(ch4Phi_BeamGasPDF)

    ch4Phi_bkgPDF = probfit.functor.AddPdf(ch4Phi_TouschekPDF,
                                             ch4Phi_BeamGasPDF,
                                             prefix=['Touschek', 'BeamGas']
                                             )

    ch4Phi_chi2 = probfit.BinnedChi2(ch4Phi_bkgPDF,
                                       #np.concatenate([sim_angles[11][0],
                                       #    sim_angles[11][1]]),
                                       data_angles[3],
                                       bins=9,bound=(-90,90),
                                       )

    ch4Phi_minu = iminuit.Minuit(ch4Phi_chi2)
    ch4Phi_minu.migrad()


    parameters = ch4Phi_minu.values
    ch4Phi_chi2.draw(parts=True, print_par=False)
    ((data_edges, data_y), (errorp,errorm), (total_pdf_x, total_pdf_y), parts) = (
            ch4Phi_chi2.draw(parts=True, print_par=False, no_plot=True))

    data_x = 0.5 * (data_edges[:-1] + data_edges[1:])

    Touschek_norm = np.sum(ch4Phi_Touschek_hist[0])
    BeamGas_norm = np.sum(ch4Phi_BeamGas_hist[0])

    plt.show()

    # Plot sum of Touschek and beam gas obtained from histogram PDF
    sum_hists = (ch4Phi_Touschek_hist[0] + ch4Phi_BeamGas_hist[0])
    sum_norm = np.sum(sum_hists)
    sum_fitted_norm = parameters['TouschekN'] + parameters['BeamGasN']

    plt.hist(phis,
            weights=sum_hists*sum_fitted_norm/sum_norm,
            color='C2', bins=data_edges, label='MC Fitted Sum')

    # Plot beam gas distribution obtained from histogram PDF
    plt.hist(phis,
            weights=ch4Phi_BeamGas_hist[0]*parameters['BeamGasN']/BeamGas_norm,
            color='C1', bins=data_edges, label='MC Beam Gas')

    # Plot Touschek distribution obtained from histogram PDF
    plt.hist(phis,
            weights=ch4Phi_Touschek_hist[0]*parameters['TouschekN']/Touschek_norm,
            color='C0', bins=data_edges, label='MC Touschek')

    plt.errorbar(data_x, data_y, fmt='o', yerr=np.sqrt(data_y), color='k')

    plt.xlabel('Ch. 4 $\\phi$ [$^\circ$]', ha='right', x=1.0)
    plt.ylabel('Events per bin', ha='right', y=1.0)
    plt.legend(loc='best')
    plt.savefig('ch4_phi_histoPDF_fit.pdf')
    plt.show()



    weights=[ 
            np.array([1/(len(sim_angles[8][0])+len(sim_angles[8][1]))]*len(sim_angles[8][0])),
            np.array([1/(len(sim_angles[8][0])+len(sim_angles[8][1]))]*len(sim_angles[8][1])),
            ]

    (n, bins, patches) = plt.hist(data_angles[0], bins=theta_bins,
            range=[0,180])

    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.hist(sim_angles[8], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    ax1.errorbar(thetas+10, n/np.sum(n), yerr=np.sqrt(n)/np.sum(n),
           color='black', fmt='o',label='Experiment')
    ax1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    ax1.set_ylabel('Events per bin',ha='right',y=1.0)
    ax1.set_xlim(-10,190)
    ax1.legend(loc='best')
    f.savefig('TPC3_theta_datavsmc.pdf')

    (n, bins, patches) = plt.hist(data_angles[1], bins=theta_bins,
            range=[0,180])

    weights=[ 
            np.array([1/(len(sim_angles[9][0])+len(sim_angles[9][1]))]*len(sim_angles[9][0])),
            np.array([1/(len(sim_angles[9][0])+len(sim_angles[9][1]))]*len(sim_angles[9][1])),
            ]
    g = plt.figure()
    bx1 = g.add_subplot(111)
    bx1.hist(sim_angles[9], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    bx1.errorbar(thetas+10, n/np.sum(n), yerr=np.sqrt(n)/np.sum(n), color='black',
           fmt='o', label='Experiment')
    bx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    bx1.set_ylabel('Events per bin',ha='right',y=1.0)
    bx1.set_xlim(-10,190)
    bx1.legend(loc='best')
    g.savefig('TPC4_theta_datavsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[10][0])+len(sim_angles[10][1]))]*len(sim_angles[10][0])),
            np.array([1/(len(sim_angles[10][0])+len(sim_angles[10][1]))]*len(sim_angles[10][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[2], bins=phi_bins, range=[-90,90])
    h = plt.figure()
    cx1 = h.add_subplot(111)
    cx1.hist(sim_angles[10], bins=phi_bins, range=[-90,90], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    cx1.errorbar(phis+10, n/np.sum(n), yerr=np.sqrt(n)/np.sum(n), color='black',
           fmt='o', label='Experiment')
    cx1.set_xlabel('Ch. 3 $\phi$ [$^{\circ}$]',ha='right',x=1.0)
    cx1.set_ylabel('Events per bin',ha='right',y=1.0)
    cx1.set_xlim(-100,100)
    cx1.legend(loc='best')
    h.savefig('TPC3_phi_datavsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[11][0])+len(sim_angles[11][1]))]*len(sim_angles[11][0])),
            np.array([1/(len(sim_angles[11][0])+len(sim_angles[11][1]))]*len(sim_angles[11][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[3], bins=phi_bins, range=[-90,90])
    k = plt.figure()
    dx1 = k.add_subplot(111)
    dx1.hist(sim_angles[11], bins=phi_bins, range=[-90,90], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    dx1.errorbar(phis+10, n/np.sum(n), yerr=np.sqrt(n)/np.sum(n), color='black',
            fmt='o', label='Experiment')
    dx1.set_xlabel('Ch. 4 $\phi$ [$^{\circ}$]',ha='right',x=1.0)
    dx1.set_ylabel('Events per bin',ha='right',y=1.0)
    dx1.set_xlim(-100,100)
    dx1.legend(loc='best')
    k.savefig('TPC4_phi_datavsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[12][0])+len(sim_angles[12][1]))]*len(sim_angles[12][0])),
            np.array([1/(len(sim_angles[12][0])+len(sim_angles[12][1]))]*len(sim_angles[12][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[4], bins=theta_bins,
            range=[0,180])
    l = plt.figure()
    ex1 = l.add_subplot(111)
    ex1.hist(sim_angles[12], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    ex1.errorbar(thetas+10,n/np.sum(n),yerr=np.sqrt(n)/np.sum(n),color='black',
            fmt='o',label='Experiment')
    ex1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    ex1.set_ylabel('Events per bin',ha='right',y=1.0)
    ex1.set_xlim(-10,190)
    ex1.legend(loc='best')
    l.savefig('TPC3_theta_bpdirectvsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[13][0])+len(sim_angles[13][1]))]*len(sim_angles[13][0])),
            np.array([1/(len(sim_angles[13][0])+len(sim_angles[13][1]))]*len(sim_angles[13][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[5], bins=theta_bins,
            range=[0,180])
    m = plt.figure()
    fx1 = m.add_subplot(111)
    fx1.hist(sim_angles[13], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    fx1.errorbar(thetas+10,n/np.sum(n),yerr=np.sqrt(n)/np.sum(n),color='black',
           fmt='o',label='Experiment')
    fx1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    fx1.set_ylabel('Events per bin',ha='right',y=1.0)
    fx1.set_xlim(-10,190)
    fx1.legend(loc='best')
    m.savefig('TPC3_theta_not_bpdirectvsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[14][0])+len(sim_angles[14][1]))]*len(sim_angles[14][0])),
            np.array([1/(len(sim_angles[14][0])+len(sim_angles[14][1]))]*len(sim_angles[14][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[6], bins=theta_bins,
           range=[0,180])
    o = plt.figure()
    gx1 = o.add_subplot(111)
    gx1.hist(sim_angles[14], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    gx1.errorbar(thetas+10,n/np.sum(n),yerr=np.sqrt(n)/np.sum(n),color='black',
           fmt='o',label='Experiment')
    gx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    gx1.set_ylabel('Events per bin',ha='right',y=1.0)
    gx1.set_xlim(-10,190)
    gx1.legend(loc='best')
    o.savefig('TPC4_theta_bpdirectvsmc.pdf')

    weights=[ 
            np.array([1/(len(sim_angles[15][0])+len(sim_angles[15][1]))]*len(sim_angles[15][0])),
            np.array([1/(len(sim_angles[15][0])+len(sim_angles[15][1]))]*len(sim_angles[15][1])),
            ]
    (n, bins, patches) = plt.hist(data_angles[7], bins=theta_bins,
            range=[0,180])
    p = plt.figure()
    hx1 = p.add_subplot(111)
    hx1.hist(sim_angles[15], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    hx1.errorbar(thetas+10,n/np.sum(n),yerr=np.sqrt(n)/np.sum(n),color='black',
            fmt='o',label='Experiment')
    hx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    hx1.set_ylabel('Events per bin',ha='right',y=1.0)
    hx1.set_xlim(-10,190)
    hx1.legend(loc='best')
    p.savefig('TPC4_theta_not_bpdirectvsmc.pdf')


    # Normalizing [Touschek, Beamgas] weights by time and sim beam conditions

    beast_datapath = '/Users/BEASTzilla/BEAST/data/v3.1/'
    subrun_times, subrun_BeamGas, subrun_Touschek, _ = calc_sim_weights(beast_datapath, simpath)

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[8][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[8][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)

    print('Printing integrals of reweighted angular histograms ...')
    print('Ch 3 Theta Touschek:', len(sim_angles[8][0]) * weights[0][0])
    print('Ch 3 Theta Beam Gas:', len(sim_angles[8][1]) * weights[1][0])
    print('Ch 4 Theta Touschek:', len(sim_angles[9][0]) * weights[0][0])
    print('Ch 4 Theta Beam Gas:', len(sim_angles[9][1]) * weights[1][0])
    print('Ch 3 Phi Touschek:', len(sim_angles[10][0]) * weights[0][0])
    print('Ch 3 Phi Beam Gas:', len(sim_angles[10][1]) * weights[1][0])
    print('Ch 4 Phi Touschek:', len(sim_angles[11][0]) * weights[0][0])
    print('Ch 4 Phi Beam Gas:', len(sim_angles[11][1]) * weights[1][0])

    print(subrun_Touschek * subrun_times)
    print(subrun_BeamGas * subrun_times)
    input('well?')


    (n, bins, patches) = plt.hist(data_angles[0], bins=theta_bins,
            range=[0,180])
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.hist(sim_angles[8], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights, )
    ax1.errorbar(thetas+10, n, yerr=np.sqrt(n),
            color='black', fmt='o',label='Experiment')
    ax1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    ax1.set_ylabel('Events per bin',ha='right',y=1.0)
    ax1.set_xlim(-10,190)
    ax1.legend(loc='best')

    f.savefig('TPC3_theta_datavsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[9][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[9][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[1], bins=theta_bins,
            range=[0,180])
            
    g = plt.figure()
    bx1 = g.add_subplot(111)

    #bx1.hist(thetas, bins=theta_bins,
    #        weights=sim_angles[1]*(5.5/10), label='Sim',
    #        range=[0,180])
    bx1.hist(sim_angles[9], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    bx1.errorbar(thetas+10, n, yerr=np.sqrt(n), color='black',
            fmt='o', label='Experiment')

    bx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    bx1.set_ylabel('Events per bin',ha='right',y=1.0)
    bx1.set_xlim(-10,190)
    bx1.legend(loc='best')

    g.savefig('TPC4_theta_datavsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[10][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[10][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[2], bins=phi_bins, range=[-90,90])
    h = plt.figure()
    cx1 = h.add_subplot(111)

    #cx1.hist(phis, bins=phi_bins, weights=sim_angles[2]*(5.5/10),
    #        label='Sim', range=[-90,90])
    cx1.hist(sim_angles[10], bins=phi_bins, range=[-90,90], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    cx1.errorbar(phis+10, n, yerr=np.sqrt(n), color='black',
            fmt='o', label='Experiment')

    cx1.set_xlabel('Ch. 3 $\phi$ [$^{\circ}$]',ha='right',x=1.0)
    cx1.set_ylabel('Events per bin',ha='right',y=1.0)
    cx1.set_xlim(-100,100)
    cx1.legend(loc='best')

    h.savefig('TPC3_phi_datavsmc_sim_weighted.pdf')


    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[11][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[11][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[3], bins=phi_bins, range=[-90,90])
    k = plt.figure()
    dx1 = k.add_subplot(111)

    #dx1.hist(phis, bins=phi_bins, weights=sim_angles[3]*(5.5/10),
    #        label='Sim', range=[-90,90])
    dx1.hist(sim_angles[11], bins=phi_bins, range=[-90,90], stacked=True,
            label=['Touschek MC','Beam Gas MC'],
            weights=weights)
    dx1.errorbar(phis+10, n, yerr=np.sqrt(n), color='black',
            fmt='o', label='Experiment')

    dx1.set_xlabel('Ch. 4 $\phi$ [$^{\circ}$]',ha='right',x=1.0)
    dx1.set_ylabel('Events per bin',ha='right',y=1.0)
    dx1.set_xlim(-100,100)
    dx1.legend(loc='best')

    k.savefig('TPC4_phi_datavsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[12][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[12][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[4], bins=theta_bins,
            range=[0,180])
    l = plt.figure()
    ex1 = l.add_subplot(111)
    ex1.hist(sim_angles[12], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC', 'Beam Gas MC'],
            weights=weights)
    ex1.errorbar(thetas+10,n,yerr=np.sqrt(n),color='black',
            fmt='o',label='Experiment')
    ex1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    ex1.set_ylabel('Events per bin',ha='right',y=1.0)
    ex1.set_xlim(-10,190)
    ex1.legend(loc='best')
    l.savefig('TPC3_theta_bpdirectvsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[13][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[13][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[5], bins=theta_bins,
            range=[0,180])
    m = plt.figure()
    fx1 = m.add_subplot(111)
    fx1.hist(sim_angles[13], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC', 'Beam Gas MC'],
            weights=weights)
    fx1.errorbar(thetas+10,n,yerr=np.sqrt(n),color='black',
           fmt='o',label='Experiment')
    fx1.set_xlabel('Ch. 3 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    fx1.set_ylabel('Events per bin',ha='right',y=1.0)
    fx1.set_xlim(-10,190)
    fx1.legend(loc='best')
    m.savefig('TPC3_theta_not_bpdirectvsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[14][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[14][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[6], bins=theta_bins,
           range=[0,180])
    o = plt.figure()
    gx1 = o.add_subplot(111)
    gx1.hist(sim_angles[14], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC', 'Beam Gas MC'],
            weights=weights)
    gx1.errorbar(thetas+10,n,yerr=np.sqrt(n),color='black',
           fmt='o',label='Experiment')
    gx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    gx1.set_ylabel('Events per bin',ha='right',y=1.0)
    gx1.set_xlim(-10,190)
    gx1.legend(loc='best')
    o.savefig('TPC4_theta_bpdirectvsmc_sim_weighted.pdf')

    weights=[ 
            np.array([1.0/(36000.0*9090.91)]*len(sim_angles[15][0])),
            np.array([1.0/(36000.0*0.0097)]*len(sim_angles[15][1])),
            ]

    weights[0] *= np.sum(subrun_Touschek*subrun_times)
    weights[1] *= np.sum(subrun_BeamGas*subrun_times)
    (n, bins, patches) = plt.hist(data_angles[7], bins=theta_bins,
            range=[0,180])
    p = plt.figure()
    hx1 = p.add_subplot(111)
    hx1.hist(sim_angles[15], bins=theta_bins, range=[0,180], stacked=True,
            label=['Touschek MC', 'Beam Gas MC'],
            weights=weights)
    hx1.errorbar(thetas+10,n,yerr=np.sqrt(n),color='black',
            fmt='o',label='Experiment')
    hx1.set_xlabel('Ch. 4 $\\theta$ [$^{\circ}$]',ha='right',x=1.0)
    hx1.set_ylabel('Events per bin',ha='right',y=1.0)
    hx1.set_xlim(-10,190)
    hx1.legend(loc='best')
    p.savefig('TPC4_theta_notbpdirectvsmc_sim_weighted.pdf')

    plt.show()


def cut_study_data(datapath):
    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    branches = ['hitside',
                'de_dx',
                #'col', 
                #'row', 
                #'tot',
                'tot_sum',
                'e_sum',
                't_length',
                'neutron',
                'detnb',
                'par_fit_err',
                'phi',
                'theta',
                'npoints',
                'bcid']

    t_lengths = []
    sumQ = []
    theta_errs= []
    phi_errs = []

    phis = []
    thetas = []

    npoints = []
    tots = []
    bcids = []

    total_events = 0
    hitside_pass = 0
    dQdx_pass = 0
    npoints_pass = 0

    for subdir, dirs, files in os.walk(datapath):
        for f in files:
            if '.DS' in f : continue
            strs = f.split('_')
            if int(strs[-2]) not in good_files : continue
            r_file = str(subdir) + str('/') + str(f)
            print(r_file)
            data = root2rec(r_file, branches=branches)

            if 'tpc4' in r_file: data.e_sum *= 1.43

            cuts = (
                    (data.hitside == 0)
                    & (data.t_length > 0)
                    & (data.e_sum /data.t_length > 500.0)
                    & (data.npoints > 40)
                    )

            total_events += len(data)
            hitside_pass += len(data[data.hitside==0])
            dQdx_pass += len(data[( 
                                  (data.hitside==0)
                                  & (data.t_length > 0)
                                  & ( (data.e_sum/data.t_length) > 500.0)
                                  )] )
            npoints_pass += len(data[cuts])

            t_lengths = np.concatenate([t_lengths, data.t_length[cuts]])
            sumQ = np.concatenate([sumQ, data.e_sum[cuts]]) 
            theta_errs = np.concatenate([theta_errs, data.par_fit_err[cuts, 3]])
            phi_errs = np.concatenate([phi_errs, data.par_fit_err[cuts, 4]])
            phis = np.concatenate([phis, data.phi[cuts]])
            thetas = np.concatenate([thetas, data.theta[cuts]])
            npoints = np.concatenate([npoints, data.npoints[cuts]])
            bcids.append(data.bcid[cuts].tolist())

    # Convert from radians -> degrees
    theta_errs *= 180./np.pi
    phi_errs *= 180./np.pi

    dQdx = sumQ / t_lengths

    # Convert from Q -> keV
    gain1 = 30.0
    gain2 = 50.0
    w = 35.075
    sumE = w * 1E-3 * sumQ/(gain1 * gain2)

    print('Total number of events:', total_events)
    print('Pass edge cut:', hitside_pass)
    print('Pass dQdx cut:', dQdx_pass)
    print('Pass npoints cut (all cuts):',npoints_pass)
    print(len(theta_errs), len(np.nonzero(theta_errs)[0]))
    print(len(phi_errs), len(np.nonzero(phi_errs)[0]))

    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(t_lengths, theta_errs)
    #plt.xlim([-100,2000])
    ax1.set_xlabel('Track length ($\mu$m)')
    #plt.ylim([-0.1,1.])
    ax1.set_ylabel('$\\theta$ uncertainty')

    g = plt.figure()
    ax2 = g.add_subplot(111)
    ax2.scatter(t_lengths, phi_errs)
    #plt.xlim([-100,5000])
    ax2.set_xlabel('Track length ($\mu$m)')
    #plt.ylim([-0.1,2.])
    ax2.set_ylabel('$\phi$ uncertainty')

    h = plt.figure()
    ax3 = h.add_subplot(111)
    ax3.scatter(t_lengths, sumQ)
    ax3.set_xlabel('Track Length ($\mu$m)')
    ax3.set_ylabel('Detected charge')

    k = plt.figure()
    ax4 = k.add_subplot(111)
    ax4.scatter(sumE, theta_errs)
    #plt.xlim([-1E5,2E6])
    ax4.set_xlabel('Sum E (keV)')
    #plt.ylim([-0.1,1.])
    ax4.set_ylabel('$\\theta$ uncertainty')

    l = plt.figure()
    ax5 = l.add_subplot(111)
    ax5.scatter(sumE, phi_errs)
    #plt.xlim([-1E5,2E6])
    ax5.set_xlabel('Sum E (keV)')
    #plt.ylim([-0.1,2.])
    ax5.set_ylabel('$\phi$ uncertainty')

    m = plt.figure()
    ax6 = m.add_subplot(111)
    ax6.hist(phis, bins = 100, range=[-360,360])
    ax6.set_xlabel('Phi')

    n = plt.figure()
    ax7 = n.add_subplot(111)
    ax7.hist(thetas, bins = 100, range=[-360,360])
    ax7.set_xlabel('Theta')

    p = plt.figure()
    ax8 = p.add_subplot(111)
    ax8.scatter(npoints, phi_errs)

    #import seaborn as sns
    #sns.jointplot(x=npoints, y=phi_errs, stat_func=None, xlim=[-100,1000], ylim=[-10,100]).set_axis_labels("Npoints", "Phi err (degrees)")

    q = plt.figure()
    ax9 = q.add_subplot(111)
    ax9.hist(npoints, bins=100)
    ax9.set_xlabel('Npoints')

    r = plt.figure()
    bx1 = r.add_subplot(111)
    bx1.hist(npoints, bins=100, range=[-10,100])
    bx1.set_xlabel('Npoints')

    #min_bcid = []
    #bcid_range = []

    ##bcids = np.array(bcids)
    #for i in range(len(bcids)):
    #    for event in bcids[i]:
    #        min_bcid.append(np.min(event))
    #        bcid_range.append(np.max(event) - np.min(event) )

    #min_bcid = np.array(min_bcid)
    #bcid_range = np.array(bcid_range)

    #print(len(bcid_range[bcid_range > 15]))
    #print(len(bcid_range[bcid_range <= 15]))
    #print(len(bcid_range))

    #s = plt.figure()
    #bx2 = s.add_subplot(111)
    #bx2.hist(min_bcid, bins=100)
    #bx2.set_xlabel('Min bcid')

    #t = plt.figure()
    #bx3 = t.add_subplot(111)
    #bx3.hist(bcid_range, bins = 100)
    #bx3.set_xlabel('bcid range')

    plt.show()

def fit_study(datapath):
    names = ['TPC3_Touschek_LER',
             'TPC4_Touschek_LER',
             'TPC3_Touschek_HER',
             'TPC4_Touschek_HER',
             'TPC3_Coulomb_LER',
             'TPC4_Coulomb_LER',
             'TPC3_Coulomb_HER',
             'TPC4_Coulomb_HER',
             'TPC3_Brems_LER',
             'TPC4_Brems_LER',
             'TPC3_Brems_HER',
             'TPC4_Brems_HER']

    truth_file = '/Users/BEASTzilla/BEAST/sim/v5.2/FTFP_BERT_HP/mc_beast_run_2016-02-09.root'

    sumQ = []
    tlengths = []
    thetas = []
    phis = []
    pdg = []
    detnbs = []
    truthKE = []
    phi_errs = []
    theta_errs = []
    npoints = []
    hitside = []
    neutrons = []
    truth_theta = []
    truth_phi = []
    min_rets = []
    chi2s = []

    truth_detnbs = []

    for file in os.listdir(datapath):
        if 'TPC' not in file : continue
        infile = datapath + file 
    #for name in names:
    #    infile = datapath + name + str('.root')
        if '10hr' not in file : continue
        print(infile)
        try : 
            data = root2rec(infile)
        except :
            print('File is empty. Continuing ...')
            continue
        sumQ = np.concatenate([sumQ, (data.e_sum)])
        tlengths = np.concatenate([tlengths, (data.t_length)])
        thetas = np.concatenate([thetas, (data.theta)])
        phis = np.concatenate([phis, (data.phi)])
        pdg = np.concatenate([pdg, (data.pdg)])
        detnbs = np.concatenate([detnbs, (data.detnb + 1)])
        truthKE = np.concatenate([truthKE, data.truth_KineticEnergy])
        phi_errs = np.concatenate([phi_errs, data.par_fit_err[:,4]])
        theta_errs = np.concatenate([theta_errs, data.par_fit_err[:,3]])
        npoints = np.concatenate([npoints, data.npoints])
        neutrons = np.concatenate([neutrons, data.neutron])
        hitside = np.concatenate([hitside,data.hitside])
        min_rets = np.concatenate([min_rets, data.min_ret])
        chi2s = np.concatenate([chi2s, data.chi2])

    #for name in names:
        truth_tree = file.split('.')[0]
        data = root2rec(truth_file, truth_tree)
        truth_theta = np.concatenate([truth_theta, data.truth_Theta])
        truth_phi = np.concatenate([truth_phi, data.truth_phi])
        truth_detnbs = np.concatenate([truth_detnbs, (data.detnb + 1) ])
    
    phi_errs *= (180.0/np.pi)
    theta_errs *= (180.0/np.pi)
    sumQ[detnbs==4] *= (4.0/3.0)

    dQdx = sumQ/tlengths
    dQdx[tlengths == 0] = 0 

    #truth_phi[detnbs==4] += 90.

    angle_sels = ( 
                 (detnbs == 3)
                 & (hitside == 0)
                 & (dQdx > 500.0)
                 & (npoints > 40)
                 #& ( ((npoints < 400) & (dQdx < 1500)) | 
                 #    ((npoints < 1500) & (dQdx > 1500) ))
                 & (min_rets == 0)
                 & (thetas > 0)
                 & (thetas < 180)
                 & (np.abs(phis) < 360 )
                 #& (tlengths > 2000)
                 )

    folded_phis = phis
    folded_thetas = thetas

    folded_truth_phis = truth_phi
    folded_truth_thetas = truth_theta

    folded_thetas[ (folded_phis < -90) ] *= -1
    folded_thetas[ (folded_phis < -90) ] += 180

    folded_thetas[ (folded_phis > 90) ] *= -1
    folded_thetas[ (folded_phis > 90) ] += 180

    folded_phis[ (folded_phis < -90) ] += 180
    folded_phis[ (folded_phis > 90) ] -= 180

    folded_truth_thetas[ (folded_truth_phis < -90) ] *= -1
    folded_truth_thetas[ (folded_truth_phis < -90) ] += 180

    folded_truth_thetas[ (folded_truth_phis > 90) ] *= -1
    folded_truth_thetas[ (folded_truth_phis > 90) ] += 180

    folded_truth_phis[ (folded_truth_phis < -90) ] += 180
    folded_truth_phis[ (folded_truth_phis > 90) ] -= 180

    print(len(folded_phis[angle_sels]), len(folded_truth_phis[angle_sels]), len(phi_errs[angle_sels]))
    print(len(folded_thetas[angle_sels]), len(folded_truth_thetas[angle_sels]),
            len(theta_errs[angle_sels]))

    folded_phi_errs = np.abs(folded_phis - folded_truth_phis)
    folded_phi_errs[folded_phi_errs > 90] =  (180.0 - 
            folded_phi_errs[folded_phi_errs > 90] )

    folded_theta_errs = np.abs(folded_thetas - folded_truth_thetas)
    folded_theta_errs[folded_theta_errs > 90] =  (180.0 - 
            folded_theta_errs[folded_theta_errs > 90] )

    from collections import Counter
    c = Counter(pdg[angle_sels])
    print(c)

    ### Study of true fit error

    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.scatter(np.abs(folded_phis[angle_sels] - folded_truth_phis[angle_sels]),
            phi_errs[angle_sels])

    ax1.set_xlabel('$\phi$ - $\phi$$_{truth}$')
    ax1.set_ylabel('$\phi$ error')

    g = plt.figure()
    ax2 = g.add_subplot(111)
    ax2.scatter(folded_theta_errs[angle_sels],
            theta_errs[angle_sels])
    ax2.set_xlabel('$\\theta$ - $\\theta$$_{truth}$')
    ax2.set_ylabel('$\\theta$ error')

    h = plt.figure()
    ax3 = h.add_subplot(111)
    ax3.hist(folded_truth_phis[angle_sels], bins = 18)
    ax3.set_xlabel('$\phi$$_{reco}$')

    l = plt.figure()
    ax4 = l.add_subplot(111)
    ax4.hist(folded_truth_thetas[angle_sels], bins = 9)
    ax4.set_xlabel('$\\theta$$_{reco}$')

    m = plt.figure()
    ax5= m.add_subplot(111)
    ax5.hist(truth_theta[angle_sels], bins=9)
    ax5.set_xlabel('$\\theta$$_{truth}$')

    n = plt.figure()
    ax6= n.add_subplot(111)
    ax6.hist(truth_phi[angle_sels], bins=18)
    ax6.set_xlabel('$\phi$$_{truth}$')

    p = plt.figure()
    ax7 = p.add_subplot(111)
    ax7.scatter(tlengths[angle_sels], folded_phi_errs[angle_sels] )
    ax7.set_xlabel('Track length [$\mu$m]',
                    ha='right', x=1.0)
    ax7.set_ylabel('Truth $\phi$ error [$^{\circ}$]',
                    ha='right', y=1.0)
    #p.savefig('tlength_vs_phiError_withoutcut.pdf')

    q = plt.figure()
    ax8 = q.add_subplot(111)
    ax8.scatter(chi2s[angle_sels], folded_phi_errs[angle_sels] )
    ax8.set_xlabel('Chi2')
    ax8.set_ylabel('True phi error (degrees)')
    ax8.set_xlim(-0.01, 0.01)
    #q.savefig('chi2_vs_phiError_withcut.pdf')

    r = plt.figure()
    ax9 = r.add_subplot(111)
    ax9.scatter(tlengths[angle_sels],
            folded_theta_errs[angle_sels]) 
    ax9.set_xlabel('Track length [$\mu$m]',
                    ha='right', x=1.0)
    ax9.set_xlim(plt.xlim()[0], 5000)
    ax9.set_ylabel('Truth $\\theta$ error [$^{\circ}$]',
                    ha='right', y=1.0)
    r.savefig('tlength_vs_thetaError_withoutcut.pdf')

    r = plt.figure()
    ax9 = r.add_subplot(111)
    ax9.scatter(tlengths[angle_sels],
            folded_phi_errs[angle_sels]) 
    ax9.set_xlabel('Track length [$\mu$m]',
                    ha='right', x=1.0)
    ax9.set_xlim(plt.xlim()[0], 5000)
    ax9.set_ylabel('Truth $\\phi$ error [$^{\circ}$]',
                    ha='right', y=1.0)
    r.savefig('tlength_vs_phiError_withoutcut.pdf')

    s = plt.figure()
    ax10 = s.add_subplot(111)
    ax10.hist(folded_phi_errs[angle_sels],bins=18)
    ax10.set_xlabel('Truth $\phi$ error (degrees)',
                    ha='right', x=1.0),
    ax10.set_ylabel('Events per bin',
                    ha='right', y=1.0)
    #s.savefig('true_phi_error_withoutcut.pdf')

    t = plt.figure()
    ax11 = t.add_subplot(111)
    ax11.hist(folded_theta_errs[angle_sels],bins=18)
    ax11.set_xlabel('Truth $\\theta$ error (degrees)',
                    ha='right', x=1.0)
    ax11.set_ylabel('Events per bin',
                    ha='right', y=1.0)
    #t.savefig('true_theta_error_withoutcut.pdf')




    u = plt.figure()
    ax12 = u.add_subplot(111)

    weights_phi_all=np.array([1/len(folded_truth_phis)] * len(folded_truth_phis) )

    (nPhi_all, binsPhi_all, _) = ax12.hist(folded_truth_phis,
                                            bins=9,
                                            histtype='step',
                                            range=[-90,90],
                                            label='All truth $\phi$',
                                            #weights=weights_phi_all,
                                            )

    weights_phi_sels=np.array([1/len(folded_truth_phis[angle_sels])] *
            len(folded_truth_phis[angle_sels]) )
    (nPhi_sels, binsPhi_sels, _) = ax12.hist(folded_truth_phis[angle_sels],
                                            bins=9,
                                            histtype='step',
                                            range=[-90,90], 
                                            label='Selected truth $\phi$',
                                            #weights=weights_phi_sels,
                                            )
    ax12.set_xlabel('$\phi$ [$^\circ$]', ha='right', x=1.0)
    plt.legend(loc='best')
    ax12.set_ylabel('Events per bin', ha='right', y=1.0)
    u.savefig('truthPhi_all_vs_selected_recoils.pdf')

    v = plt.figure()
    ax13 = v.add_subplot(111)

    weights_theta_all=np.array([1/len(folded_truth_thetas)] *
            len(folded_truth_thetas) )

    (nTheta_all, binsTheta_all, _) = ax13.hist(folded_truth_thetas,
                                            bins=9,
                                            histtype='step',
                                            range=[0,180], 
                                            label='All truth $\\theta$',
                                            #weights=np.sqrt(nTheta_all),
                                            )

    weights_theta_sels=np.array([1/len(folded_truth_thetas[angle_sels])] *
            len(folded_truth_thetas[angle_sels]) )
    (nTheta_sels, binsTheta_sels, _) = ax13.hist(folded_truth_thetas[angle_sels],
                                            bins=9,
                                            histtype='step',
                                            range=[0,180],
                                            label='Selected truth $\\theta$',
                                            #weights=np.sqrt(nTheta_sels)
                                            )
    ax13.set_xlabel('$\\theta$ [$^\circ$]', ha='right', x=1.0)
    ax13.set_ylabel('Events per bin', ha='right', y=1.0)
    ax13.legend(loc='best')
    v.savefig('truthTheta_all_vs_selected_recoils.pdf')

    phi_bin_centers = 0.5 * (binsPhi_sels[:-1] + binsPhi_sels[1:])
    aa = plt.figure()
    ax13 = aa.add_subplot(111)
    ax13.errorbar(phi_bin_centers, nPhi_sels/nPhi_all, fmt='o',
            yerr=nPhi_sels/nPhi_all*np.sqrt(1/nPhi_all+1/nPhi_sels), color='k')
    ax13.set_xlabel('$\phi$ [$^\circ$]', ha='right',
            x=1.0)
    ax13.set_ylabel('Ratio [selected/total]', ha='right', y=1.0)
    ax13.set_ylim(0.0, 0.2)
    aa.savefig('truthPhi_all_vs_selected_ratio.pdf')

    theta_bin_centers = 0.5 * (binsTheta_sels[:-1] + binsTheta_sels[1:])
    bb = plt.figure()
    ax14 = bb.add_subplot(111)
    #ax14.hist(theta_bin_centers, bins=binsTheta_all, weights=nTheta_sels/nTheta_all,
    #        range=[0,180])
    ax14.errorbar(theta_bin_centers, nTheta_sels/nTheta_all, fmt='o',
            yerr=nTheta_sels/nTheta_all*np.sqrt(1/nTheta_all+1/nTheta_sels), color='k')
    ax14.set_xlabel('$\\theta$ [$^\circ$]', ha='right',
            x=1.0)
    ax14.set_ylabel('Ratio [selected/total]', ha='right', y=1.0)
    ax14.set_ylim(0.0, 0.2)
    bb.savefig('truthTheta_all_vs_selected_ratio.pdf')

    print('Number of wild theta errs (greater than 20 degrees):',
            len(np.where(folded_theta_errs[angle_sels] > 20)[0]))
    print('Number of wild phi errs: (greater than 20 degrees)',
            len(np.where(folded_phi_errs[angle_sels] > 20)[0]))

    plt.show()

def cut_study(simpath, datapath):
    #truth_file = '/Users/BEASTzilla/BEAST/sim/v4.1/mc_beast_run_2016-02-09.root'

    sumQ = []
    tlengths = []
    thetas = []
    phis = []
    pdg = []
    detnbs = []
    truthKE = []
    phi_errs = []
    theta_errs = []
    npoints = []
    hitside = []
    neutrons = []
    truth_theta = []
    truth_phi = []
    min_rets = []
    chi2s = []

    # For HitOR veto
    tots = []
    bcids = []

    for file in os.listdir(simpath):
        if file == 'old_ver_noLER' : continue
        #if 'HER' in file : continue
        infile = simpath + file 
        print(infile)
        try :
           data = root2rec(infile)
        except :
            print('\nFile %s is empty.  Continuing ...\n' % (infile) )
        sumQ = np.concatenate([sumQ, (data.e_sum)])
        tlengths = np.concatenate([tlengths, (data.t_length)])
        thetas = np.concatenate([thetas, (data.theta)])
        phis = np.concatenate([phis, (data.phi)])
        pdg = np.concatenate([pdg, (data.pdg)])
        detnbs = np.concatenate([detnbs, (data.detnb + 1)])
        truthKE = np.concatenate([truthKE, data.truth_KineticEnergy])
        phi_errs = np.concatenate([phi_errs, data.par_fit_err[:,4]])
        theta_errs = np.concatenate([theta_errs, data.par_fit_err[:,3]])
        npoints = np.concatenate([npoints, data.npoints])
        neutrons = np.concatenate([neutrons, data.neutron])
        hitside = np.concatenate([hitside,data.hitside])
        min_rets = np.concatenate([min_rets, data.min_ret])
        chi2s = np.concatenate([chi2s, data.chi2])

        tots = np.concatenate([tots, data.tot])
        bcids = np.concatenate([bcids, data.bcid])

    #for name in names:
    #    data = root2rec(truth_file, name)
    #    truth_theta = np.concatenate([truth_theta, data.truth_Theta])
    #    truth_phi = np.concatenate([truth_phi, data.truth_phi])
    #    truth_detnbs = np.concatenate([truth_detnbs, (data.detnb + 1) ])
    
    phi_errs *= (180.0/np.pi)
    theta_errs *= (180.0/np.pi)
    if 'v4' in datapath: 
        print('Applying 4/3 correction factor for TPC4 gain (Sim v4) ...')
        sumQ[detnbs==4] *= (4.0/3.0)

    dQdx = sumQ/tlengths
    dQdx[tlengths == 0] = 0 

    # Calculate hitOR pulse width
    pulse_widths = []

    for i in range(len(bcids)) :
        lowest_bcid = np.min(bcids[i])
        pulses = bcids[i] + tots[i]
        largest_pulse = np.max(pulses)
        largest_pulse_element = np.where(pulses==largest_pulse)
        
        evt_pulse_width = largest_pulse - lowest_bcid
        pulse_widths.append(evt_pulse_width)
        
    pulse_widths = np.array(pulse_widths)

    print('Total number of neutrons:', len(dQdx[pdg>10000]))
    print('Total number of background', len(dQdx[pdg<10000]))
    print('Total number of events', len(dQdx))

    from collections import Counter
    p = Counter(pdg)
    print('PDG numbers:\n',p)


    ### Define ctional cuts
    bak_hitOR_cut = (
                    (pdg < 10000)
                    & (pulse_widths > 3)
                    )

    bak_edge_cut = (
                   (pdg < 10000)
                   & (pulse_widths > 3)
                   & (hitside == 0)
                   )

    bak_min_rets_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      )

    bak_dQdx_cut = (
                   (pdg < 10000)
                   & (pulse_widths > 3)
                   & (hitside == 0)
                   & (min_rets == 0)
                   & (dQdx > 500)
                   )

    bak_npoints_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      )

    bak_npoints_vs_dQdx_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      )

    bak_theta_range_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      )

    bak_phi_range_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      & (np.abs(phis) < 360)
                      )

    bak_tlength_cut = (
                      (pdg < 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      & (np.abs(phis) < 360)
                      & (tlengths > 2000)
                      )


    sig_hitOR_cut = (
                    (pdg > 10000)
                    & (pulse_widths > 3)
                    )

    sig_edge_cut = (
                   (pdg > 10000)
                   & (pulse_widths > 3)
                   & (hitside == 0)
                   )

    sig_min_rets_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      )

    sig_dQdx_cut = (
                   (pdg > 10000)
                   & (pulse_widths > 3)
                   & (hitside == 0)
                   & (min_rets == 0)
                   & (dQdx > 500)
                   )

    sig_npoints_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & ( ((npoints < 400) & (dQdx < 1500)) |
                          ((npoints < 1500) & (dQdx > 1500) ))
                      )

    sig_npoints_vs_dQdx_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      #& ( (npoints > 40) & ((npoints < 400) & (dQdx < 1500)) | 
                      #    (dQdx > 1500) )
                      & ( ((npoints < 400) & (dQdx < 1500)) |
                          ((npoints < 1500) & (dQdx > 1500) ))
                      )


    sig_theta_range_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      )

    sig_phi_range_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      & (np.abs(phis) < 360)
                      )

    sig_tlength_cut = (
                      (pdg > 10000)
                      & (pulse_widths > 3)
                      & (hitside == 0)
                      & (min_rets == 0)
                      & (dQdx > 500)
                      & (npoints > 40)
                      & (thetas > 0)
                      & (thetas < 180)
                      & (np.abs(phis) < 360)
                      & (tlengths > 2000)
                      )

    ### Populate data arrays
    data_sumQ = []
    data_tlengths = []
    data_thetas = []
    data_phis = []
    data_detnbs = []
    data_truthKE = []
    data_phi_errs = []
    data_theta_errs = []
    data_npoints = []
    data_hitside = []
    data_neutrons = []
    data_truth_theta = []
    data_truth_phi = []
    data_min_rets = []
    data_chi2s = []

    # For using v2 of TOT -> charge calibration
    data_tots = []
    data_sumQ_v2 = []

    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    branches = [
               'hitside',
               'e_sum',
               't_length',
               'detnb',
               'phi',
               'theta',
               'neutron',
               'min_ret',
               'chi2',
               'npoints',
               'par_fit_err',
               'tot',
               ]

    for file in os.listdir(datapath):
        strs = file.split('_')
        if int(strs[-2]) not in good_files : continue #For data files
        infile = datapath + file 
        print(infile)
        try :
           data = root2rec(infile,branches=branches)
        except :
            print('\nFile %s is empty.  Continuing ...\n' % (infile) )
        data_sumQ = np.concatenate([data_sumQ, (data.e_sum)])
        data_tlengths = np.concatenate([data_tlengths, (data.t_length)])
        data_thetas = np.concatenate([data_thetas, (data.theta)])
        data_phis = np.concatenate([data_phis, (data.phi)])
        data_detnbs = np.concatenate([data_detnbs, (data.detnb)])
        data_phi_errs = np.concatenate([data_phi_errs, data.par_fit_err[:,4]])
        data_theta_errs = np.concatenate([data_theta_errs, data.par_fit_err[:,3]])
        data_npoints = np.concatenate([data_npoints, data.npoints])
        data_neutrons = np.concatenate([data_neutrons, data.neutron])
        data_hitside = np.concatenate([data_hitside,data.hitside])
        data_min_rets = np.concatenate([data_min_rets, data.min_ret])
        data_chi2s = np.concatenate([data_chi2s, data.chi2])
        data_tots = np.concatenate([data_tots, data.tot])


    # Correct data energy to match simulation with v1 charge conversion (see energy_cal() )
    data_sumQ[data_detnbs==3] *= 1.18
    data_sumQ[data_detnbs==4] *= 1.64

    data_phi_errs *= (180.0/np.pi)
    data_theta_errs *= (180.0/np.pi)

    data_dQdx = data_sumQ/data_tlengths
    data_dQdx[data_tlengths == 0] = 0 

    print('Total amount of events in data:', len(data_npoints))

    data_edge_cut = (
                    (data_hitside == 0)
                   )

    data_min_rets_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      )

    data_dQdx_cut = (
                    (data_hitside == 0)
                   & (data_min_rets == 0)
                   & (data_dQdx > 500)
                   )

    data_npoints_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      & (data_dQdx > 500)
                      & (data_npoints > 40)
                      )

    data_npoints_vs_dQdx_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      & (data_dQdx > 500)
                      & (data_npoints > 40)
                      #& ( (npoints > 40) & ((npoints < 400) & (dQdx < 1500)) | 
                      #    (dQdx > 1500) )
                      #& ( ((data_npoints < 400) & (data_dQdx < 1500)) |
                      #    ((data_npoints < 1500) & (data_dQdx > 1500) ))
                      )


    data_theta_range_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      & (data_dQdx > 500)
                      & (data_npoints > 40)
                      & (data_thetas > 0)
                      & (data_thetas < 180)
                      )

    data_phi_range_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      & (data_dQdx > 500)
                      & (data_npoints > 40)
                      & (data_thetas > 0)
                      & (data_thetas < 180)
                      & (np.abs(data_phis) < 360)
                      )

    data_tlength_cut = (
                       (data_hitside == 0)
                      & (data_min_rets == 0)
                      & (data_dQdx > 500)
                      & (data_npoints > 40)
                      & (data_thetas > 0)
                      & (data_thetas < 180)
                      & (np.abs(data_phis) < 360)
                      & (data_tlengths > 2000)
                      )

    ### Apply v2 of TOT -> charge conversion
    # Populate tot arrays for selected events, and recalculate sumQ 
    print('Recalculating charge with v2 conversion scheme ...')
    grPlsrDACvTOT = TGraph("TOTcalibration.txt")
    PlsrDACtoQ = 52.0

    data_tots = data_tots[data_npoints_cut]
    data_tots_v2 = []

    for i in range(len(data_tots) ):
        event_tot_v2 = []
        for k in range(len(data_tots[i])):
            tot_v2 = (1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_tots[i][k]+0.5) if
                    data_tots[i][k]<13 else 1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_tots[i][k]) )
            event_tot_v2.append(tot_v2)
        data_tots_v2.append(event_tot_v2)
        event_tot_v2 = np.array(event_tot_v2)
        data_sumQ_v2.append(np.sum(event_tot_v2))

    data_sumQ_v2 = np.array(data_sumQ_v2)
    data_tots_v2 = np.array(data_tots_v2)
    data_sumQ_v2[data_detnbs[data_npoints_cut] == 3] *= 1.07
    data_sumQ_v2[data_detnbs[data_npoints_cut] == 4] *= 1.43
    data_dQdx_v2 = data_sumQ_v2/data_tlengths[data_npoints_cut]

    ### Plot dQdx for sim vs data
    d = plt.figure()
    ax00 = d.add_subplot(111)
    ax00.scatter(tlengths[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                label='MC He Recoils', color='C1' )

    ax00.scatter(tlengths[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                label='MC C/O Recoils', color='C2' )

    ax00.scatter(tlengths[( (bak_npoints_cut) & (pdg < 10000) )],
                sumQ[( (bak_npoints_cut) & (pdg < 10000) )],
                label='MC Protons', color='C0')
    ax00.scatter(data_tlengths[( (data_npoints_cut) )],
                #data_sumQ_v2[( (data_npoints_cut) )],
                data_sumQ_v2,
                facecolor='none', label='Experiment', color='k', s=8.8)

    ax00.set_xlabel('Track Length [$\mu$m]', ha='right', x=1.0)
    ax00.set_ylim(plt.ylim()[0], 1E8)
    ax00.set_ylabel('Detected Charge v2 [electrons]', ha='right', y=1.0)
    ax00.legend(loc='best')
    plotname = 'tlength_vs_Erecoil_v52_and_data.pdf'
    d.savefig(plotname)
    plt.show()

    # Print out number of events surviving each cut:
    signum_hitOR_cut = len(dQdx[sig_hitOR_cut])/len(npoints[pdg>10000])
    baknum_hitOR_cut = len(dQdx[bak_hitOR_cut])/len(npoints[pdg<10000])
    print('Number of events passing hitOR veto:\n Signal: %f Bkg: %f'
            % (signum_hitOR_cut, baknum_hitOR_cut) )

    signum_edge_cut = len(dQdx[sig_edge_cut])/len(npoints[pdg>10000])
    baknum_edge_cut = len(dQdx[bak_edge_cut])/len(npoints[pdg<10000])
    datanum_edge_cut = len(data_dQdx[data_edge_cut])/len(data_npoints)
    print('Number of events passing edge cut:\n Signal: %f Bkg: %f Data: %f'
            % (signum_edge_cut, baknum_edge_cut, datanum_edge_cut) )

    signum_min_rets_cut = len(npoints[sig_min_rets_cut])/len(npoints[pdg>10000])
    baknum_min_rets_cut = len(npoints[bak_min_rets_cut])/len(npoints[pdg<10000])
    datanum_min_rets_cut = len(data_npoints[data_min_rets_cut])/len(data_npoints)
    print('Number of events with converged fit:\n Signal: %f Bkg: %f Data: %f'
            % (signum_min_rets_cut, baknum_min_rets_cut, datanum_min_rets_cut) )

    signum_dQdx_cut = len(npoints[sig_dQdx_cut])/len(npoints[pdg>10000])
    baknum_dQdx_cut = len(npoints[bak_dQdx_cut])/len(npoints[pdg<10000])
    datanum_dQdx_cut = len(data_npoints[data_dQdx_cut])/len(data_npoints)
    print('Number of events passing dQdx cut:\n Signal: %f Bkg: %f Data: %f'
            % (signum_dQdx_cut, baknum_dQdx_cut, datanum_dQdx_cut) )

    signum_npoints_cut = len(npoints[sig_npoints_cut])/len(npoints[pdg>10000])
    baknum_npoints_cut = len(npoints[bak_npoints_cut])/len(npoints[pdg<10000])
    datanum_npoints_cut = len(data_npoints[data_npoints_cut])/len(data_npoints)
    print('Number of events passing npoints cut:\n Signal: %f Bkg: %f Data: %f'
            % (signum_npoints_cut, baknum_npoints_cut, datanum_npoints_cut) )

    #signum_npoints_vs_dQdx_cut = len(npoints[sig_npoints_vs_dQdx_cut])/len(npoints[pdg>10000])
    #baknum_npoints_vs_dQdx_cut = len(npoints[bak_npoints_vs_dQdx_cut])/len(npoints[pdg<10000])
    #datanum_npoints_vs_dQdx_cut = len(data_npoints[data_npoiints_vs_dQdx_cut])/len(data_npoints)
    #print('Number of events passing npoints_dQdx box cut:\n Signal: %f Bkg: %f\
    #        Data: %f'
    #        % (signum_npoints_vs_dQdx_cut, baknum_npoints_vs_dQdx_cut,
    #            datanum_npoints_vs_dQdx_cut) )

    signum_theta_range_cut = len(npoints[sig_theta_range_cut])/len(npoints[pdg>10000])
    baknum_theta_range_cut = len(npoints[bak_theta_range_cut])/len(npoints[pdg<10000])
    datanum_theta_range_cut = len(data_npoints[data_theta_range_cut])/len(data_npoints)
    print('Number of events within theta range:\n Signal: %f Bkg: %f Data: %f'
            % (signum_theta_range_cut, baknum_theta_range_cut,
            datanum_theta_range_cut) )

    signum_phi_range_cut = len(npoints[sig_phi_range_cut])/len(npoints[pdg>10000])
    baknum_phi_range_cut = len(npoints[bak_phi_range_cut])/len(npoints[pdg<10000])
    datanum_phi_range_cut = len(data_npoints[data_phi_range_cut])/len(data_npoints)
    print('Number of events within phi range:\n Signal: %f Bkg: %f Data: %f'
            % (signum_phi_range_cut, baknum_phi_range_cut,
                datanum_phi_range_cut) )

    signum_tlength_cut = len(npoints[sig_tlength_cut])/len(npoints[pdg>10000])
    baknum_tlength_cut = len(npoints[bak_tlength_cut])/len(npoints[pdg<10000])
    datanum_tlength_cut = len(data_npoints[data_tlength_cut])/len(data_npoints)
    print('Number of events longer than 2mm:\n Signal: %f Bkg: %f Data: %f'
            % (signum_tlength_cut, baknum_tlength_cut, datanum_tlength_cut) )

    # Print out number of events of each type:


    ### Print out number of events of each type passing all cuts:

    bak_hitside = hitside[pdg<10000]
    int_bak_hitside = []
    for i in range(len(bak_hitside)):
        int_bak_hitside.append(int(str(int(bak_hitside[i])),2))
    int_bak_hitside = np.asarray(int_bak_hitside)

    sig_hitside = hitside[pdg>10000]
    int_sig_hitside = []
    for i in range(len(sig_hitside)):
        int_sig_hitside.append(int(str(int(sig_hitside[i])),2))
    int_sig_hitside = np.asarray(int_sig_hitside)

    ### Show edge cut distribution
    e = plt.figure()
    ax0 = e.add_subplot(111)
    ax0.hist(int_bak_hitside, 
            bins=16,
            range=[0,16],
            label='MC Background')
    ax0.hist(int_sig_hitside,
            bins=16,
            range=[0,16],
            label='MC Nuclear Recoils',
            histtype='step',
            color='k')
    ax0.set_xlabel('Edge code', ha='right', x=1.0)
    ax0.set_ylabel('Events per bin', ha='right', y=1.0)
    ax0.get_xaxis().set_major_locator(MaxNLocator(integer=True) )
    ax0.legend(loc='best')
    if 'v4.1' in simpath : 
        plotname= 'cuts_edgecode_v41.pdf'
    elif 'v5.2' in simpath :
        plotname = 'cuts_edgecode_v52.pdf'
    e.savefig(plotname)

    ### Show dQ/dx distribution for events passing edge_cut
    # Zoomed
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.hist(dQdx[bak_edge_cut], bins=100,
            range=[0,np.max(dQdx)], label='MC Background')
    ax1.hist(dQdx[sig_edge_cut], bins=100,
            range=[0,np.max(dQdx)], label='MC Nuclear Recoils',
            histtype='step', color='k')
    ax1.set_xlabel('dQ/dx [charge/$\mu$m]', ha='right', x=1.0)
    ax1.set_xlim(0,1000)
    ax1.set_ylabel('Events per bin', ha='right', y=1.0)
    ax1.legend(loc='best')
    if 'v4.1' in simpath :
        plotname = 'cuts_dQdx_zoomed_v41.pdf'
    elif 'v5.2' in simpath:
        plotname = 'cuts_dQdx_zoomed_v52.pdf'
    f.savefig(plotname)

    # Unzoomed
    g = plt.figure()
    ax2 = g.add_subplot(111)
    #weights = 
    ax2.hist(dQdx[bak_edge_cut], bins=100,
            range=[0,np.max(dQdx)], label='MC Background')
    ax2.hist(dQdx[sig_edge_cut], bins=100,
            range=[0,np.max(dQdx)], label='MC Nuclear Recoils',
            histtype='step', color='k')
    ax2.set_xlabel('dQ/dx [charge/$\mu$m]', ha='right', x=1.0)
    ax2.set_ylabel('Events per bin', ha='right', y=1.0)
    ax2.legend(loc='best')
    if 'v4.1' in simpath :
        plotname = 'cuts_dQdx_v41.pdf'
    elif 'v5.2' in simpath :
        plotname = 'cuts_dQdx_v52.pdf'
    g.savefig(plotname)


    ### Show npoints distribution for events passing dQ/dx cut
    # Zoomed
    h = plt.figure()
    ax3 = h.add_subplot(111)
    ax3.hist(npoints[bak_dQdx_cut], bins=100, range=[0,np.max(npoints)],
            label='MC Background')
    ax3.hist(npoints[sig_dQdx_cut], bins=100, range=[0,np.max(npoints)],
            label='MC Nuclear Recoils', histtype='step', color='k')
    ax3.legend(loc='best')
    ax3.set_xlabel('Pixels over threshold', ha='right', x=1.0)
    ax3.set_ylabel('Events per bin', ha='right', y=1.0)
    ax3.set_xlim(0,1000)
    if 'v4.1' in simpath :
        plotname = 'cuts_npoints_zoomed_v41.pdf'
    elif 'v5.2' in simpath :
        plotname = 'cuts_npoints_zoomed_v52.pdf'
    h.savefig(plotname)

    # Unzoomed
    l = plt.figure()
    ax4 = l.add_subplot(111)
    ax4.hist(npoints[bak_dQdx_cut], bins=100,
            range=[0,np.max(npoints)], label='MC Background')
    ax4.hist(npoints[sig_dQdx_cut], bins=100,
            range=[0,np.max(npoints)], label='MC Nuclear Recoils',
            histtype='step', color='k')
    ax4.legend(loc='best')
    ax4.set_xlabel('Pixels over threshold', ha='right', x=1.0)
    ax4.set_ylabel('Events per bin', ha='right', y=1.0)
    if 'v4.1' in simpath:
        plotname = 'cuts_npoints_v41.pdf'
    elif 'v5.2' in simpath:
        plotname = 'cuts_npoints_v52.pdf'
    l.savefig(plotname)

    m = plt.figure()
    ax5 = m.add_subplot(111)
    #ax5.scatter(dQdx[sig_min_rets_cut], npoints[sig_min_rets_cut],
    #            label='MC Nuclear Recoils',color='k', facecolor='none')
    ax5.scatter(dQdx[bak_min_rets_cut], npoints[bak_min_rets_cut],
                label='MC Background', color='C0')
    ax5.scatter(dQdx[( (hitside==0) & (pdg==1000020040.0) & (dQdx>500.0) )],
                npoints[( (hitside==0) & (pdg==1000020040.0) & (dQdx>500.0) )],
                label='MC He Recoils', color='C1')
    ax5.scatter(dQdx[( (hitside==0) & (pdg>1000020040.0) & (dQdx>500.0) )], 
                npoints[( (hitside==0) & (pdg>1000020040.0) & (dQdx>500.0) )],
                label='MC C/O Recoils', color='C2')
    ax5.scatter(data_dQdx[data_npoints_cut], data_npoints[data_npoints_cut],
                label='Experiment', color='k', facecolor='none', s=8.8)
    ax5.set_xlabel('dQ/dx [charge/$\mu$m]', ha='right', x=1.0)
    ax5.set_ylabel('Pixels over threshold', ha='right', y=1.0)
    ax5.legend(loc='best')
    if 'v4.1' in simpath:
        plotname = 'cuts_dQdx_vs_npoints_v41.pdf'
    elif 'v5.2' in simpath:
        plotname = 'cuts_dQdx_vs_npoints_v52.pdf'
    m.savefig(plotname)

    p = plt.figure()
    ax8 = p.add_subplot(111)
    #ax8.scatter(dQdx[sig_min_rets_cut], npoints[sig_min_rets_cut],
    #            label='MC Nuclear Recoils',color='k', facecolor='none')
    ax8.scatter(dQdx[bak_min_rets_cut], npoints[bak_min_rets_cut],
                label='MC Background', color='C0')
    ax8.scatter(dQdx[( (hitside==0) & (pdg==1000020040.0) & (dQdx>500.0) )],
                npoints[( (hitside==0) & (pdg==1000020040.0) & (dQdx>500.0) )],
                label='MC He Recoils', facecolor='none', color='C1')
    ax8.scatter(dQdx[( (hitside==0) & (pdg>1000020040.0) & (dQdx>500.0) )], 
                npoints[( (hitside==0) & (pdg>1000020040.0) & (dQdx>500.0) )],
                label='MC C/O Recoils', facecolor='none', color='C2')
    ax8.set_xlabel('dQ/dx [charge/$\mu$m]', ha='right', x=1.0)
    ax8.set_ylabel('Pixels over threshold', ha='right', y=1.0)
    ax8.set_xlim(plt.xlim()[0],1500)
    ax8.set_ylim(plt.ylim()[0],1000)
    ax8.legend(loc='best')
    if 'v4.1' in simpath:
        plotname = 'cuts_dQdx_vs_npoints_v41_zoomed.pdf'
    elif 'v5.2' in simpath:
        plotname = 'cuts_dQdx_vs_npoints_v52_zoomed.pdf'
    p.savefig(plotname)

    #surviving_protons
    print('PDG numbers of surviving bkg:', pdg[bak_tlength_cut])

    n = plt.figure()
    ax6 = n.add_subplot(111)
    ax6.hist(npoints[pdg<10000], label='MC Background', range=[0,100],
            bins=20)
    ax6.hist(npoints[pdg>10000], label='MC Nuclear Recoils', range=[0,100],
            bins=20, histtype='step', color='k')
    ax6.set_xlabel('Pixels over threshold', ha='right', x=1.0)
    ax6.set_ylabel('Events per bin', ha='right', y=1.0)
    #ax6.legend(loc='best')

    o = plt.figure()
    ax7 = o.add_subplot(111)
    ax7.scatter(tlengths[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                label='MC He Recoils', color='C1' )

    ax7.scatter(tlengths[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                label='MC C/O Recoils', color='C2' )

    ax7.scatter(tlengths[( (bak_npoints_cut) & (pdg < 10000) )],
                sumQ[( (bak_npoints_cut) & (pdg < 10000) )],
                label='MC Protons', color='C0')
    ax7.scatter(data_tlengths[( (data_npoints_cut) )],
                data_sumQ[( (data_npoints_cut) )],
                facecolor='none', label='Experiment', color='k', s=8.8)

    ax7.set_xlabel('Track Length [$\mu$m]', ha='right', x=1.0)
    ax7.set_ylim(plt.ylim()[0], 1E8)
    ax7.set_ylabel('Detected Charge [electrons]', ha='right', y=1.0)
    ax7.legend(loc='best')
    if 'v4.1' in simpath:
        plotname = 'tlength_vs_Erecoil_v41.pdf'
    elif 'v5.2' in simpath:
        plotname = 'tlength_vs_Erecoil_v52_and_data.pdf'
    plotname = 'tlength_vs_Erecoil_v52_and_data.pdf'
    o.savefig(plotname)

    o = plt.figure()
    ax7 = o.add_subplot(111)
    ax7.scatter(tlengths[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg == 1000020040.0) )],
                label='MC He Recoils', color='C1' )

    ax7.scatter(tlengths[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                sumQ[( (sig_npoints_cut) & (pdg > 1000020040.0) )],
                label='MC C/O Recoils', color='C2' )

    ax7.scatter(tlengths[( (bak_npoints_cut) & (pdg < 10000) )],
                sumQ[( (bak_npoints_cut) & (pdg < 10000) )],
                label='MC Protons', color='C0')
    ax7.scatter(data_tlengths[( (data_npoints_cut) )],
                data_sumQ[( (data_npoints_cut) )],
                facecolor='none', label='Experiment', color='k', s=8.8)

    ax7.set_xlabel('Track Length [$\mu$m]', ha='right', x=1.0)
    ax7.set_xlim(plt.xlim()[0], 15000)
    ax7.set_ylim(plt.ylim()[0], 0.5E8)
    ax7.set_ylabel('Detected Charge [electrons]', ha='right', y=1.0)
    ax7.legend(loc='best')
    plotname = 'tlength_vs_Erecoil_v52_and_data_zoomed.pdf'
    o.savefig(plotname)

    plt.show()

def energy_cal(simpath, datapath):
    ### Populate simulation alpha arrays
    sim_topSource_Q = []
    sim_botSource_Q = []

    sim_topSource_phi = []
    sim_botSource_phi = []
    sim_topSource_theta = []
    sim_botSource_theta = []

    sim_topSource_tlengths = []
    sim_botSource_tlengths = []

    sim_topSource_dQdx = []
    sim_botSource_dQdx = []

    branches = ['e_sum',
                'phi',
                'theta',
                'de_dx',
                'npoints',
                'tot',
                'event',
                'hitside',
                't_length',
                ]

    print('Getting simulated top source ...')
    ifile = simpath + str('TPC4_alpha_top.root')
    data = root2rec(ifile, branches=branches)
    alpha_sels = (
                 (data.hitside == 11)
                 & (data.phi < 1 )
                 & (data.phi > -1)
                 & (data.theta < 91)
                 & (data.theta > 89)
                 & (data.event < 10000)
                 )
    #cuts = (
    #        )
    sim_topSource_Q = np.concatenate([sim_topSource_Q,
        data.e_sum[alpha_sels]])
    sim_topSource_phi = np.concatenate([sim_topSource_phi,
        data.phi[alpha_sels]])
    sim_topSource_theta = np.concatenate([sim_topSource_theta,
        data.theta[alpha_sels]])
    sim_topSource_tlengths = np.concatenate([sim_topSource_tlengths,
        data.t_length[alpha_sels] ])
    sim_topSource_dQdx = np.concatenate([sim_topSource_dQdx,
        data.de_dx[alpha_sels] ])

    # For v2 of TOT->Q converstion
    print('Recalculating charge with v2 conversion scheme ...')
    grPlsrDACvTOT = TGraph("TOTcalibration.txt")
    PlsrDACtoQ = 52.0

    sim_topSource_Q_v2 = []
    sim_topSource_tots = []
    sim_topSource_tots_v2 = []
    sim_topSource_npoints = []
    
    sim_topSource_tots = np.concatenate([sim_topSource_tots,
        data.tot[alpha_sels]])
    sim_topSource_npoints = np.concatenate([sim_topSource_npoints,
        data.npoints[alpha_sels]])

    for i in range(len(sim_topSource_tots)):
        event_tot_v2 = []
        for k in range(int(sim_topSource_npoints[i])):
            tot_v2 = (PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(sim_topSource_tots[i][k]+0.5) if
                    sim_topSource_tots[i][k]<13 else PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(sim_topSource_tots[i][k]) )
            event_tot_v2.append(tot_v2)
        sim_topSource_tots_v2.append(event_tot_v2)
        sim_topSource_Q_v2.append(np.sum(event_tot_v2))

    sim_topSource_Q_v2 = np.array(sim_topSource_Q_v2)


    print('Getting simulated bottom source ...')
    ifile = simpath + str('TPC4_alpha_bot.root')
    data = root2rec(ifile, branches=branches)
    alpha_sels = (
                 (data.hitside == 11)
                 & (data.phi < 1 )
                 & (data.phi > -1)
                 & (data.theta < 91)
                 & (data.theta > 89)
                 & (data.event < 10000)
                 )
    sim_botSource_Q = np.concatenate([sim_botSource_Q,
        data.e_sum[alpha_sels]])
    sim_botSource_phi = np.concatenate([sim_topSource_phi,
        data.phi[alpha_sels]])
    sim_botSource_theta = np.concatenate([sim_topSource_theta,
        data.theta[alpha_sels]])
    sim_botSource_tlengths = np.concatenate([sim_botSource_tlengths,
        data.t_length[alpha_sels] ])
    sim_botSource_dQdx = np.concatenate([sim_botSource_dQdx,
        data.de_dx[alpha_sels] ])
    sim_botSource_dQdx = np.concatenate([sim_botSource_dQdx,
        data.de_dx[alpha_sels] ])

    print('Recalculating charge with v2 conversion scheme ...')
    sim_botSource_Q_v2 = []
    sim_botSource_tots = []
    sim_botSource_tots_v2 = []
    sim_botSource_npoints = []
    
    sim_botSource_tots = np.concatenate([sim_botSource_tots,
        data.tot[alpha_sels]])
    sim_botSource_npoints = np.concatenate([sim_botSource_npoints,
        data.npoints[alpha_sels]])

    for i in range(len(sim_botSource_tots)):
        event_tot_v2 = []
        for k in range(int(sim_botSource_npoints[i])):
            tot_v2 = (PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(sim_botSource_tots[i][k]+0.5) if
                    sim_botSource_tots[i][k]<13 else PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(sim_botSource_tots[i][k]) )
            event_tot_v2.append(tot_v2)
        sim_botSource_tots_v2.append(event_tot_v2)
        sim_botSource_Q_v2.append(np.sum(event_tot_v2))

    sim_botSource_Q_v2 = np.array(sim_botSource_Q_v2)


    sim_topSource_dQdx_v2 = sim_topSource_Q_v2/sim_topSource_tlengths
    sim_botSource_dQdx_v2 = sim_botSource_Q_v2/sim_botSource_tlengths

    print('Number of alphas:', len(sim_topSource_Q) + len(sim_botSource_Q) )

    

    ### Populate data alpha arrays
    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    tmax = 1464505369.0
    tmin = 1464485170.0
    
    branches = [
                'hitside',
                'top_alpha',
                'bottom_alpha',
                'e_sum',
                'phi',
                'theta',
                'detnb',
                't_length',
                'de_dx',
                'tot',
                'npoints',
                ]

    data_Q = []
    data_phi = []
    data_theta = []

    data_top_source = []
    data_bot_source = []

    detnbs = []

    data_ch3Top_Q = []
    data_ch4Top_Q = []
    data_ch3Bot_Q = []
    data_ch4Bot_Q = []

    data_ch3Top_Q_v2 = []
    data_ch4Top_Q_v2 = []
    data_ch3Bot_Q_v2 = []
    data_ch4Bot_Q_v2 = []

    data_ch3Top_tot_v2 = []
    data_ch4Top_tot_v2 = []
    data_ch3Bot_tot_v2 = []
    data_ch4Bot_tot_v2 = []

    data_ch3Top_dQdx = []
    data_ch4Top_dQdx = []
    data_ch3Bot_dQdx = []
    data_ch4Bot_dQdx = []

    data_ch3Top_tot = []
    data_ch4Top_tot = []
    data_ch3Bot_tot = []
    data_ch4Bot_tot = []

    data_ch3Top_npoints = []
    data_ch4Top_npoints = []
    data_ch3Bot_npoints = []
    data_ch4Bot_npoints = []

    data_ch3Top_tlengths = []
    data_ch4Top_tlengths = []
    data_ch3Bot_tlengths = []
    data_ch4Bot_tlengths = []

    total_alphas = 0

    print('Moving to data files ...')
    for f in os.listdir(datapath):
        #if 'tpc4' in f : continue
        strs = f.split('_')
        if int(strs[-2]) not in good_files : continue #For data files
        print(f)
        dfile = datapath + f
        data = root2rec(dfile, branches=branches)
        total_alphas += len(data[data.hitside == 11])
        alpha_sels = (
                     (data.hitside == 11)
                     & (data.phi < 1 )
                     & (data.phi > -1)
                     & (data.theta < 91.0)
                     & (data.theta > 89.0)
                     )

        detnbs = np.concatenate([ detnbs, data.detnb[(alpha_sels)] ])
        data_Q = np.concatenate([ data_Q, data.e_sum[(alpha_sels)] ])
        data_phi = np.concatenate( [data_phi, data.phi[(alpha_sels)] ])
        #data_theta = np.concatenate( [data_theta, data.theta[(alpha_sels)] ])

        data_theta = np.concatenate( [data_theta, 
            data.theta[((alpha_sels) & (data.detnb==4) & (data.bottom_alpha==1))] ])
        data_top_source = np.concatenate([data_top_source,
            data.top_alpha[( (data.top_alpha == 1 ) & (alpha_sels) )] ])
        data_bot_source = np.concatenate([data_bot_source,
            data.bottom_alpha[( (data.bottom_alpha == 1 ) & (alpha_sels) )] ])

        data_ch3Top_Q = np.concatenate([data_ch3Top_Q,
            data.e_sum[(
                      (alpha_sels)
                      & (data.detnb == 3)
                      & (data.top_alpha == 1)
                      )]
                      ])

        data_ch4Top_Q = np.concatenate([data_ch4Top_Q,
            data.e_sum[(
                      (alpha_sels)
                      & (data.detnb == 4)
                      & (data.top_alpha == 1)
                      )]
                      ])

        data_ch3Bot_Q = np.concatenate([data_ch3Bot_Q,
            data.e_sum[(
                      (alpha_sels)
                      & (data.detnb == 3)
                      & (data.bottom_alpha == 1)
                      )]
                      ])

        data_ch4Bot_Q = np.concatenate([data_ch4Bot_Q,
            data.e_sum[(
                      (alpha_sels)
                      & (data.detnb == 4)
                      & (data.bottom_alpha == 1)
                      )]
                      ])

        data_ch3Top_dQdx = np.concatenate([data_ch3Top_dQdx,
            data.de_dx[(
                      (alpha_sels)
                      & (data.detnb == 3)
                      & (data.top_alpha == 1)
                      )]
                      ])

        data_ch4Top_dQdx = np.concatenate([data_ch4Top_dQdx,
            data.de_dx[(
                      (alpha_sels)
                      & (data.detnb == 4)
                      & (data.top_alpha == 1)
                      )]
                      ])

        data_ch3Bot_dQdx = np.concatenate([data_ch3Bot_dQdx,
            data.de_dx[(
                      (alpha_sels)
                      & (data.detnb == 3)
                      & (data.bottom_alpha == 1)
                      )]
                      ])

        data_ch4Bot_dQdx = np.concatenate([data_ch4Bot_dQdx,
            data.de_dx[(
                      (alpha_sels)
                      & (data.detnb == 4)
                      & (data.bottom_alpha == 1)
                      )]
                      ])

        data_ch3Top_tot = np.concatenate([data_ch3Top_tot,
            data.tot[(
                    (alpha_sels)
                    & (data.detnb == 3)
                    & (data.top_alpha == 1)
                    )]
                    ])
        data_ch4Top_tot = np.concatenate([data_ch4Top_tot,
            data.tot[(
                    (alpha_sels)
                    & (data.detnb == 4)
                    & (data.top_alpha == 1)
                    )]
                    ])
        data_ch3Bot_tot = np.concatenate([data_ch3Bot_tot,
            data.tot[(
                    (alpha_sels)
                    & (data.detnb == 3)
                    & (data.bottom_alpha == 1)
                    )]
                    ])
        data_ch4Bot_tot = np.concatenate([data_ch4Bot_tot,
            data.tot[(
                    (alpha_sels)
                    & (data.detnb == 4)
                    & (data.bottom_alpha == 1)
                    )]
                    ])

        data_ch3Top_npoints = np.concatenate([data_ch3Top_npoints,
            data.npoints[(
                        (alpha_sels)
                        & (data.detnb == 3)
                        & (data.top_alpha == 1)
                        )]
                        ])
        data_ch4Top_npoints = np.concatenate([data_ch4Top_npoints,
            data.npoints[(
                        (alpha_sels)
                        & (data.detnb == 4)
                        & (data.top_alpha == 1)
                        )]
                        ])

        data_ch3Bot_npoints = np.concatenate([data_ch3Bot_npoints,
            data.npoints[(
                        (alpha_sels)
                        & (data.detnb == 3)
                        & (data.bottom_alpha == 1)
                        )]
                        ])
        data_ch4Bot_npoints = np.concatenate([data_ch4Bot_npoints,
            data.npoints[(
                        (alpha_sels)
                        & (data.detnb == 4)
                        & (data.bottom_alpha == 1)
                        )]
                        ])

        data_ch3Top_tlengths = np.concatenate([data_ch3Top_tlengths,
            data.t_length[(
                         (alpha_sels)
                         & (data.detnb == 3)
                         & (data.top_alpha == 1)
                         )]
                         ])
        data_ch4Top_tlengths = np.concatenate([data_ch4Top_tlengths,
            data.t_length[(
                         (alpha_sels)
                         & (data.detnb == 4)
                         & (data.top_alpha == 1)
                         )]
                         ])
        data_ch3Bot_tlengths = np.concatenate([data_ch3Bot_tlengths,
            data.t_length[(
                         (alpha_sels)
                         & (data.detnb == 3)
                         & (data.bottom_alpha == 1)
                         )]
                         ])
        data_ch4Bot_tlengths = np.concatenate([data_ch4Bot_tlengths,
            data.t_length[(
                         (alpha_sels)
                         & (data.detnb == 4)
                         & (data.bottom_alpha == 1)
                         )]
                         ])

    for i in range(len(data_ch3Top_tot)):
        event_tot_v2 = []
        for k in range(int(data_ch3Top_npoints[i])):
            tot_v2 = (1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch3Top_tot[i][k]+0.5) if
                    data_ch3Top_tot[i][k]<13 else 1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch3Top_tot[i][k]))
            event_tot_v2.append(tot_v2)
        data_ch3Top_tot_v2.append(event_tot_v2)
        data_ch3Top_Q_v2.append(np.sum(event_tot_v2))

    data_ch3Top_Q_v2 = np.array(data_ch3Top_Q_v2)

    for i in range(len(data_ch4Top_tot)):
        event_tot_v2 = []
        for k in range(int(data_ch4Top_npoints[i])):
            tot_v2 = (1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch4Top_tot[i][k]+0.5) if
                    data_ch4Top_tot[i][k]<13 else 1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch4Top_tot[i][k]))
            event_tot_v2.append(tot_v2)
        data_ch4Top_tot_v2.append(event_tot_v2)
        data_ch4Top_Q_v2.append(np.sum(event_tot_v2))

    data_ch4Top_Q_v2 = np.array(data_ch4Top_Q_v2)

    for i in range(len(data_ch3Bot_tot)):
        event_tot_v2 = []
        for k in range(int(data_ch3Bot_npoints[i])):
            tot_v2 = (1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch3Bot_tot[i][k]+0.5) if
                    data_ch3Bot_tot[i][k]<13 else 1500 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch3Bot_tot[i][k]))
            event_tot_v2.append(tot_v2)
        data_ch3Bot_tot_v2.append(event_tot_v2)
        data_ch3Bot_Q_v2.append(np.sum(event_tot_v2))

    data_ch3Bot_Q_v2 = np.array(data_ch3Bot_Q_v2)

    for i in range(len(data_ch4Bot_tot)):
        event_tot_v2 = []
        for k in range(int(data_ch4Bot_npoints[i])):
            tot_v2 = (1500.0 + PlsrDACtoQ *
                    grPlsrDACvTOT.Eval(data_ch4Bot_tot[i][k]+0.5) if
                    data_ch4Bot_tot[i][k]<13 else 1500.0 + PlsrDACtoQ * 
                    grPlsrDACvTOT.Eval(data_ch4Bot_tot[i][k])+ 1500.0 )
            event_tot_v2.append(tot_v2)
        data_ch4Bot_tot_v2.append(event_tot_v2)
        data_ch4Bot_Q_v2.append(np.sum(event_tot_v2))

    data_ch4Bot_Q_v2 = np.array(data_ch4Bot_Q_v2)

    data_ch3Top_dQdx_v2 = data_ch3Top_Q_v2/data_ch3Top_tlengths
    data_ch4Top_dQdx_v2 = data_ch4Top_Q_v2/data_ch4Top_tlengths
    data_ch3Bot_dQdx_v2 = data_ch3Bot_Q_v2/data_ch3Bot_tlengths
    data_ch4Bot_dQdx_v2 = data_ch4Bot_Q_v2/data_ch4Bot_tlengths

    
    print('Total number vs selected alphas:', total_alphas, len(data_Q) )

    print('\nMeans of dQdx of alpha sources with v1 charge conversion:')
    print('Sim Bottom: %f Sim Top: %f' % (np.mean(sim_botSource_dQdx),
        np.mean(sim_topSource_dQdx) ))
    print('Ch3 Bottom: %f Ch3 Top: %f' % (np.mean(data_ch3Bot_dQdx),
        np.mean(data_ch3Top_dQdx) ) )
    print('Ch4 Bottom: %f Ch4 Top: %f' % (np.mean(data_ch4Bot_dQdx),
        np.mean(data_ch4Top_dQdx) ) )

    print('\nMeans of dQdx of alpha sources with v2 charge conversion:')
    print('Sim Bottom: %f Sim Top: %f' % (np.mean(sim_botSource_dQdx_v2),
        np.mean(sim_topSource_dQdx_v2) ))
    print('Ch3 Bottom: %f Ch3 Top: %f' % (np.mean(data_ch3Bot_dQdx_v2),
        np.mean(data_ch3Top_dQdx_v2) ) )
    print('Ch4 Bottom: %f Ch4 Top: %f' % (np.mean(data_ch4Bot_dQdx_v2),
        np.mean(data_ch4Top_dQdx_v2) ) )


    ### weights
    sim_botSource_Q_weights = np.ones(len(sim_botSource_Q))
    sim_botSource_Q_weights /= len(sim_botSource_Q)

    sim_topSource_Q_weights = np.ones(len(sim_topSource_Q))
    sim_topSource_Q_weights /= len(sim_topSource_Q)

    ### Plots
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.hist(sim_botSource_Q, weights=sim_botSource_Q_weights, bins=100,
            histtype='step', label='Sim Bot',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.hist(sim_topSource_Q, weights=sim_topSource_Q_weights, bins=100,
            histtype='step', label='Sim Top',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.hist(data_ch3Bot_Q, weights=[1/len(data_ch3Bot_Q)]*len(data_ch3Bot_Q),
            bins=100, histtype='step', label='Ch3 Bot',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.hist(data_ch3Top_Q,
            weights=[1/len(data_ch3Top_Q)]*len(data_ch3Top_Q),bins=100,
            histtype='step', label='Ch3 Top',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.hist(data_ch4Bot_Q,
            weights=[1/len(data_ch4Bot_Q)]*len(data_ch4Bot_Q),bins=100,
            histtype='step', label='Ch4 Bot',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.hist(data_ch4Top_Q,
            weights=[1/len(data_ch4Top_Q)]*len(data_ch4Top_Q),bins=100,
            histtype='step', label='Ch4 Top',
            range=[np.min(data_ch4Top_Q), np.max(sim_botSource_Q)])

    ax1.set_xlabel('Event sumQ\t\t\t\t\t\t\t\t\t', ha='right', x=1.0)
    ax1.set_ylabel('Events per bin', ha='right', y=1.0)
    ax1.legend(loc='best')
    #f.savefig('alpha_sim_gain1500.pdf')

    sim_botSource_dQdx_weights = np.ones(len(sim_botSource_dQdx))
    sim_botSource_dQdx_weights /= len(sim_botSource_dQdx)

    sim_topSource_dQdx_weights = np.ones(len(sim_topSource_dQdx))
    sim_topSource_dQdx_weights /= len(sim_topSource_dQdx)

    g = plt.figure()
    ax2 = g.add_subplot(111)
    ax2.hist(sim_botSource_dQdx, weights=sim_botSource_dQdx_weights, bins=100,
            histtype='step', label='Sim Bot',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.hist(sim_topSource_dQdx, weights=sim_topSource_dQdx_weights, bins=100,
            histtype='step', label='Sim Top',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.hist(data_ch3Bot_dQdx,
            weights=[1/len(data_ch3Bot_dQdx)]*len(data_ch3Bot_dQdx), bins=100,
            histtype='step', label='TPC 3 Bot',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.hist(data_ch3Top_dQdx,
            weights=[1/len(data_ch3Top_dQdx)]*len(data_ch3Top_dQdx),bins=100,
            histtype='step', label='TPC 3 Top',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.hist(data_ch4Bot_dQdx,
            weights=[1/len(data_ch4Bot_dQdx)]*len(data_ch4Bot_dQdx),bins=100,
            histtype='step', label='TPC 4 Bot',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.hist(data_ch4Top_dQdx,
            weights=[1/len(data_ch4Top_dQdx)]*len(data_ch4Top_dQdx),bins=100,
            histtype='step', label='TPC 4 Top',
            range=[np.min(data_ch4Top_dQdx), np.max(sim_botSource_dQdx)])

    ax2.set_xlabel('Event dQdx [charge/$\mu$m]', ha='right', x=1.0)
    ax2.set_ylabel('Events per bin', ha='right', y=1.0)
    ax2.set_ylim(0,0.225)
    ax2.legend(loc='best')
    g.savefig('alphas_data_mc.pdf')

    h = plt.figure()
    ax3 = h.add_subplot(111)
    ax3.scatter(data_ch4Bot_dQdx, data_theta) 
    ax3.set_xlabel('dQdx [charge/$\mu$m]')
    ax3.set_ylabel('Theta')

    sim_botSource_dQdx_v2_weights = np.ones(len(sim_botSource_dQdx_v2))
    sim_botSource_dQdx_v2_weights /= len(sim_botSource_dQdx_v2)

    sim_topSource_dQdx_v2_weights = np.ones(len(sim_topSource_dQdx_v2))
    sim_topSource_dQdx_v2_weights /= len(sim_topSource_dQdx_v2)

    k = plt.figure()
    ax3 = k.add_subplot(111)
    ax3.hist(sim_botSource_dQdx_v2, weights=sim_botSource_dQdx_v2_weights, bins=100,
            histtype='step', label='Sim Bot',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.hist(sim_topSource_dQdx_v2, weights=sim_topSource_dQdx_v2_weights, bins=100,
            histtype='step', label='Sim Top',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.hist(data_ch3Bot_dQdx_v2,
            weights=[1/len(data_ch3Bot_dQdx_v2)]*len(data_ch3Bot_dQdx_v2), bins=100,
            histtype='step', label='Ch3 Bot',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.hist(data_ch3Top_dQdx_v2,
            weights=[1/len(data_ch3Top_dQdx_v2)]*len(data_ch3Top_dQdx_v2),bins=100,
            histtype='step', label='Ch3 Top',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.hist(data_ch4Bot_dQdx_v2,
            weights=[1/len(data_ch4Bot_dQdx_v2)]*len(data_ch4Bot_dQdx_v2),bins=100,
            histtype='step', label='Ch4 Bot',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.hist(data_ch4Top_dQdx_v2,
            weights=[1/len(data_ch4Top_dQdx_v2)]*len(data_ch4Top_dQdx_v2),bins=100,
            histtype='step', label='Ch4 Top',
            range=[np.min(data_ch4Top_dQdx_v2), np.max(sim_botSource_dQdx_v2)]
            )

    ax3.set_xlabel('Event dQdx_v2 [charge/$\mu$m]', ha='right', x=1.0)
    ax3.set_ylabel('Events per bin', ha='right', y=1.0)
    ax3.legend(loc='best')

    plt.show()

def hitOR_study(simpath, datapath):

    ### Populate sim arrays
    sim_tots = []
    sim_bcids = []
    sim_pdgs = []

    # Define branches in TTree to read
    branches = ['tot', 'bcid', 'pdg']

    for f in os.listdir(simpath):
        infile = simpath + f
        print(infile)
        try :
            data = root2rec(infile, branches=branches)
        except :
            print(f, 'contains zero events.  Continuing...')
            continue
        sim_tots = np.concatenate([sim_tots, data.tot])
        sim_bcids = np.concatenate([sim_bcids, data.bcid])
        sim_pdgs = np.concatenate([sim_pdgs, data.pdg])

    sim_pulse_widths = []

    for i in range(len(sim_bcids)) :
        lowest_bcid = np.min(sim_bcids[i])
        pulses = sim_bcids[i] * sim_tots[i]
        largest_pulse = np.max(pulses)
        largest_pulse_element = np.where(pulses==largest_pulse)

        ### Debug
        #print(pulses)
        #print(sim_bcids[i])
        #print(sim_tots[i])
        #print('Largest pulse is:', largest_pulse)
        #print('Largest pulse located at', np.where(pulses==largest_pulse))
        #print('Largest pulse corresponds to element number',
        #        largest_pulse_element)
        #print('ToT & BCID values are:', sim_bcids[i][largest_pulse_element],
        #        sim_tots[i][largest_pulse_element])
        #input('well?')
        
        evt_pulse_width = largest_pulse - lowest_bcid
        sim_pulse_widths.append(evt_pulse_width)

    sim_pulse_widths = np.array(sim_pulse_widths)

    ### Populate data arrays
    data_tots = []
    data_bcids = []

    good_files = [1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200,
            1464483600,
            1464487200,
            1464490800,
            1464494400,
            1464498000,
            1464501600,
            1464505200]

    # Define branches in TTree to read
    branches = ['tot', 'bcid','top_alpha', 'bottom_alpha']

    for f in os.listdir(datapath):
        strs = f.split('_')
        if int(strs[-2]) not in good_files : continue
        print(infile)
        infile = datapath + f
        data = root2rec(infile, branches=branches)
        data_tots = np.concatenate([data_tots, data.tot])
        #data_tots = np.concatenate([data_tots, 
        #        data.tot[(data.top_alpha !=1) & (data.bottom_alpha != 1)] ])
        data_bcids = np.concatenate([data_bcids, data.bcid])
    
    data_pulse_widths = []

    for i in range(len(data_bcids)) :
        lowest_bcid = np.min(data_bcids[i])
        pulses = data_bcids[i] + data_tots[i]
        largest_pulse = np.max(pulses)
        largest_pulse_element = np.where(pulses==largest_pulse)
        
        evt_pulse_width = largest_pulse - lowest_bcid
        data_pulse_widths.append(evt_pulse_width)
        
    data_pulse_widths = np.array(data_pulse_widths)

    print('Number of simulation points:', len(sim_bcids))
    print('Number of calculated simulation pulses:', len(sim_pulse_widths))

    print('Number of data points:', len(data_bcids))
    print('Number of calculated data pulses:', len(data_pulse_widths))

    data_weights = np.ones(len(data_pulse_widths))/float(len(data_pulse_widths))
    sim_weights = np.ones(len(sim_pulse_widths))/float(len(sim_pulse_widths))
    
    f = plt.figure()
    ax1 = f.add_subplot(111)

    sig_pulses = np.histogram(sim_pulse_widths[sim_pdgs>10000], bins=10,
            range=[0,10])
    sig_pulses_normed = sig_pulses[0]/len(sim_pulse_widths)
    sig_bin_centers = (sig_pulses[1][:-1] + sig_pulses[1][1:]) / 2.0

    bak_pulses = np.histogram(sim_pulse_widths[sim_pdgs<10000], bins=10,
            range=[0,10])
    bak_pulses_normed = bak_pulses[0]/len(sim_pulse_widths)
    bak_bin_centers = (bak_pulses[1][:-1] + bak_pulses[1][1:]) / 2.0

    data_pulses = np.histogram(data_pulse_widths, bins=10,
            range=[0,10])
    data_pulses_normed = data_pulses[0]/len(data_pulse_widths)
    data_bin_centers = (data_pulses[1][:-1] + data_pulses[1][1:]) / 2.0

    ax1.hist(sig_bin_centers, weights=sig_pulses_normed, bins=sig_pulses[1],
            label='MC Nuclear Recoils', color='C0')
    ax1.hist(bak_bin_centers, weights=bak_pulses_normed, bins=bak_pulses[1],
            label='MC Background', color='C1', bottom=sig_pulses_normed)
    ax1.hist(data_bin_centers, weights=data_pulses_normed, bins=data_pulses[1],
             color='k', histtype='step', label='Data')
    ax1.set_xlabel('Pulse width [25 ns]', ha='right', x=1.0)
    ax1.set_ylabel('Events per bin', ha='right', y=1.0)
    ax1.legend(loc='best')
    f.savefig('data_mc_hitOR_pulseWidth_zoomed.pdf')

    g = plt.figure()
    ax2 = g.add_subplot(111)

    sig_pulses = np.histogram(sim_pulse_widths[sim_pdgs>10000], bins=25,
            range=[0,np.max(data_pulse_widths)])
    sig_pulses_normed = sig_pulses[0]/len(sim_pulse_widths)
    sig_bin_centers = (sig_pulses[1][:-1] + sig_pulses[1][1:]) / 2.0

    bak_pulses = np.histogram(sim_pulse_widths[sim_pdgs<10000], bins=25,
        range=[0,np.max(data_pulse_widths)])
    bak_pulses_normed = bak_pulses[0]/len(sim_pulse_widths)
    bak_bin_centers = (bak_pulses[1][:-1] + bak_pulses[1][1:]) / 2.0

    data_pulses = np.histogram(data_pulse_widths, bins=25,
        range=[0,np.max(data_pulse_widths)])
    data_pulses_normed = data_pulses[0]/len(data_pulse_widths)
    data_bin_centers = (data_pulses[1][:-1] + data_pulses[1][1:]) / 2.0

    ax2.hist(sig_bin_centers, weights=sig_pulses_normed, bins=sig_pulses[1],
            label='MC Nuclear Recoils', color='C0')
    ax2.hist(bak_bin_centers, weights=bak_pulses_normed, bins=bak_pulses[1],
            label='MC Background', color='C1', bottom=sig_pulses_normed)
    ax2.hist(data_bin_centers, weights=data_pulses_normed, bins=data_pulses[1],
             color='k', histtype='step', label='Data')
    ax2.set_xlabel('Pulse width [25 ns]', ha='right', x=1.0)
    ax2.set_ylabel('Events per bin', ha='right', y=1.0)
    ax2.legend(loc='best')

    g.savefig('data_mc_hitOR_pulseWidth.pdf')

    plt.show()



def main():

    home = expanduser('~')

    ### Use BEAST v1 data
    #datapath = str(home) + '/BEAST/data/v1/'
    
    ### Use BEAST v2 data
    v2_datapath = str(home) + '/BEAST/data/v2/'
    v31_datapath = str(home) + '/BEAST/data/v3.1/'
    #simpath = str(home) + '/BEAST/sim/v4.1/'
    v4_simpath = '/Users/BEASTzilla/BEAST/sim/sim_refitter/v4.1/'
    v50_simpath = '/Users/BEASTzilla/BEAST/sim/sim_refitter/v5.0/'

    #v52_simpath = '/Users/BEASTzilla/BEAST/sim/sim_refitter/v5.2/QGSP_BERT_HP/'
    v52_simpath = '/Users/BEASTzilla/BEAST/sim/sim_refitter/v5.2/FTFP_BERT_HP/'

    v53_simpath = '/Users/BEASTzilla/BEAST/sim/v5.3/QGSP_BERT_HP/'

    #v54_simpath = '/Users/BEASTzilla/BEAST/sim/v5.4/QGSP_BERT_HP/'
    v54_simpath = '/Users/BEASTzilla/BEAST/sim/v5.4/FTFP_BERT_HP/'

    inpath = str(home) + '/BEAST/data/TPC/tpc_toushekrun/2016-05-29/'

    compare_toushek(v31_datapath, v54_simpath)
    #compare_angles(v31_datapath, v52_simpath)
    #rate_vs_beamsize(datapath)
    #sim_rate_vs_beamsize(simpath)

    #neutron_study_raw(inpath)
    #neutron_study_sim(v4_simpath)
    #energy_study(inpath, v52_simpath)
    #energy_study(v31_datapath, v52_simpath)
    #gain_study(inpath)
    #energy_eff_study(inpath)
    #pid_study(inpath, simpath)

    #event_inspection(inpath)
    #event_inspection(v52_simpath)

    #cut_study_data(inpath) 
    #fit_study(v52_simpath)
    #cut_study(v52_simpath, inpath)

    #energy_cal(v50_simpath, inpath)

    #hitOR_study(v52_simpath, inpath)
    
if __name__ == "__main__":
    main()

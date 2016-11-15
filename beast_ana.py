import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt

from matplotlib import rc

from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.plotting.style import set_style
from root_numpy import root2rec, hist2array
from ROOT import TFile, TH1F, gROOT

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

    if run_name == 'LER_ToushekTPC': return LER_ToushekTPC
    elif run_name == 'HER_ToushekTPC': return HER_ToushekTPC


# Analyze neutron rate vs beamsize for studying the Toushek effect
def rate_vs_beamsize(datapath):
    runs = run_names('LER_ToushekTPC')

    avg_rates = []
    rates_3 = []
    rates_4 = []
    avg_inv_beamsizes = []
    invbs_errs = []
    rate_errs = []
    rate3_errs = []
    rate4_errs = []

    for f in os.listdir(datapath):
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

        #print(set(data.subrun))
        #input('well?')

        run_avg_rate = []
        run_avg_beamsize = []

        counter_3 = 0
        counter_4 = 0

        neutron_counter = 0
        #sizes = []
        timestamps = []

        for event in data:
            if event.subrun != 0 :
                if event.SKB_LER_beamSize_xray_Y[0] > 0. :
                    run_avg_beamsize.append(1./event.SKB_LER_correctedBeamSize_xray_Y[0])
                subrun = True

                tpc3_neutrons = event.TPC3_PID_neutrons
                tpc4_neutrons = event.TPC4_PID_neutrons

                for i in range(len(tpc3_neutrons)):
                    if tpc3_neutrons[i] == 1 :
                        neutron_counter += 1
                        counter_3 += 1
                        timestamps.append(event.ts)

                for k in range(len(tpc4_neutrons)):
                    if tpc4_neutrons[k] == 1 :
                        neutron_counter += 1
                        counter_4 += 1
                        timestamps.append(event.ts)

            elif event.subrun == 0 and subrun == True :
                if neutron_counter == 0 : continue

                if len(timestamps) > 1 :
                    t_range = max(timestamps) - min(timestamps)
                else : t_range = 1

                rate = float(neutron_counter)/float(t_range)
                rate_3 = float(counter_3)/float(t_range)
                rate_4 = float(counter_4)/float(t_range)
                #run_avg_rate.append(rate)

                if rate == 1 or rate == 0 :
                    subrun = False
                    timestamps = []
                    neutron_counter = 0
                    counter_3 = 0
                    counter_4 = 0
                    run_avg_rate = []
                    run_avg_beamsize = []
                    continue

                avg_rates.append(rate)
                rates_3.append(rate_3)
                rates_4.append(rate_4)
                rate_errs.append(sqrt(rate*t_range)/t_range)
                rate3_errs.append(sqrt(rate_3*t_range)/t_range)
                rate4_errs.append(sqrt(rate_4*t_range)/t_range)
                print('Rate and err:', rate, sqrt(rate*t_range)/t_range)

                run_avg_beamsize = np.array(run_avg_beamsize)
                avg_inv_beamsizes.append(np.mean(run_avg_beamsize))
                invbs_errs.append(np.std(run_avg_beamsize)/np.sqrt(len(run_avg_beamsize)))
                print('Inv_bs and err:', np.mean(run_avg_beamsize), 
                        np.std(run_avg_beamsize))

                print('Number of neutrons in subrun = %i' % neutron_counter)
                print('Time range:', t_range)
                #print(max(timestamps), min(timestamps))
                print('Ending at event number', event.event)
                subrun = False
                timestamps = []
                neutron_counter = 0
                counter_3 = 0
                counter_4 = 0
                run_avg_rate = []
                run_avg_beamsize = []
                
            else : continue

        if neutron_counter == 0 : continue

        if len(timestamps) > 1 :
            t_range = max(timestamps) - min(timestamps)
        else : t_range = 1

        rate = float(neutron_counter)/float(t_range)
        rate_3 = float(counter_3)/float(t_range)
        rate_4 = float(counter_4)/float(t_range)

        if rate == 1 or rate == 0 :
            timestamps = []
            neutron_counter = 0
            counter_3 = 0
            counter_4 = 0
            run_avg_rate = []
            run_avg_beamsize = []
            continue

        #run_avg_rate.append(rate)
        avg_rates.append(rate)
        rates_3.append(rate_3)
        rates_4.append(rate_4)

        print('Number of neutrons in subrun = %i' % neutron_counter)
        print('Time range:', t_range)
        #print(max(timestamps), min(timestamps))
        print('Ending at event number', event.event)

        #print('Avg rates:\n', run_avg_rate)
        run_avg_beamsize = np.array(run_avg_beamsize)
        avg_inv_beamsizes.append(np.mean(run_avg_beamsize))
        invbs_errs.append(np.std(run_avg_beamsize)/np.sqrt(len(run_avg_beamsize)))

        rate_errs.append(sqrt(rate*t_range)/t_range)
        rate3_errs.append(sqrt(rate_3*t_range)/t_range)
        rate4_errs.append(sqrt(rate_4*t_range)/t_range)
        print('Rate and err:', rate, sqrt(rate*t_range)/t_range)
        print('Inv_bs and err:', np.mean(run_avg_beamsize), 
                np.std(run_avg_beamsize))

        #input('well?')

        run_avg_rate = []
        run_avg_beamsize = []
        timestamps = []
        neutron_counter = 0
        counter_3 = 0
        counter_4 = 0


    avg_beamsize = np.array(avg_inv_beamsizes)
    avg_rate = np.array(avg_rates)
    rate_3 = np.array(rates_3)
    rate_4 = np.array(rates_4)
    invbs_errs = np.array(invbs_errs)

    rate_errs = np.array(rate_errs)
    rate3_errs = np.array(rate3_errs)
    rate4_errs = np.array(rate4_errs)

    print('Inverse beam size values:\n',avg_beamsize)
    print('Inv_bs error values:\n', invbs_errs)
    print('Neutron rate values:\n', avg_rate)
    print('Rate error values:\n', rate_errs)
    bs_errbars = np.array([0.])

    ### Convert beamsize and rate arrays into pandas dataframe for fun
    #arr = np.array([[0.0,0.0]]*len(avg_beamsize))
    #df = pd.DataFrame({'Average Beamsize': avg_beamsize,
    #                   'Average Rate'    : avg_rate, })

    ### Try Seaborn regplot()
    #sns.regplot(x='Average Beamsize', y='Average Rate', data=df, ci=99)
    #input('well?')

    ### Fit distribution to a line using probfit
    fit_range = (0.01, 0.03)
    chi2 = probfit.Chi2Regression(probfit.linear, avg_beamsize, avg_rate,
            rate_errs)
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
    f = plt.figure()
    ax1 = f.add_subplot(111)
    chi2.draw(minu)
    ax1.errorbar(avg_beamsize, avg_rate, yerr=rate_errs, fmt='o', color=color)
    ax1.set_xlabel('Beamsize ($\mu$$m$$^{-1}$)')
    ax1.set_ylabel('Fast neutron rate (Hz)')
    ax1.set_xlim([0.0,0.030])
    ax1.set_ylim([0.0,0.2])

    chi23 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_3,
            rate3_errs)
    minu3 = iminuit.Minuit(chi23)
    minu3.migrad()

    #g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    g = plt.figure()
    bx1 = g.add_subplot(111)
    chi23.draw(minu3)
    bx1.errorbar(avg_beamsize, rate_3, yerr=rate3_errs, fmt='o', color=color)
    bx1.set_xlabel('Beamsize ($\mu$$m$$^{-1}$)')
    bx1.set_ylabel('Fast neutron rate in TPC3 (Hz)')
    bx1.set_xlim([0.0,0.030])
    bx1.set_ylim([0.0,0.09])

    chi24 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_4,
            rate4_errs)
    minu4 = iminuit.Minuit(chi24)
    minu4.migrad()

    h = plt.figure()
    bx2 = h.add_subplot(111)
    chi24.draw(minu4)
    bx2.errorbar(avg_beamsize, rate_4, yerr=rate4_errs, fmt='o', color=color)
    bx2.set_xlabel('Beamsize ($\mu$$m$$^{-1}$)')
    bx2.set_ylabel('Fast neutron rate in TPC4 (Hz)')
    bx2.set_xlim([0.0,0.030])
    bx2.set_ylim([0.0,0.09])
    
    print(rate_3)
    print(rate_4)
    plt.show()



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

    ### Begin plotting
    if root_style == True :
        color = 'black'
        facecolor=None
    elif root_style == False :
        sns.set(color_codes=True)
        color = None

    ## Plot theta and phi, weighted by energy
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_energies_array, color = color, histtype='step')
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_energies_array, color = color, histtype='step')
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
            color = color, histtype='step')
    ax2.set_title('TPC3 Neutron Recoil $\\theta$')
    ax4.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
            color = color, histtype='step')
    ax4.set_title('TPC4 Neutron Recoil $\\theta$')
    ax4.set_xlabel('Degrees')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
            sharey='col')
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_energies_array, color = color, histtype='step')
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_energies_array, color = color, histtype='step')
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    tpc3_energies_array_bp, color = color, histtype='step')
    ax2.set_title('TPC3 Neutron Recoil $\\theta$ (Beampipe cut)')
    ax4.hist(tpc4theta_array_beampipe, theta_bins, weights = 
            tpc4_energies_array_bp, color = color, histtype='step')
    ax4.set_title('TPC4 Neutron Recoil $\\theta$ (Beampipe cut)')
    ax4.set_xlabel('Degrees')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
            sharey='col')
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_energies_array, color = color, histtype='step')
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_energies_array, color = color, histtype='step')
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array_notbp, theta_bins, weights = 
            tpc3_energies_array_notbp, color = color, histtype='step')
    ax2.set_title('TPC3 Neutron Recoil $\\theta$ (Outside beampipe)')
    ax4.hist(tpc4theta_array_notbp, theta_bins, weights = 
            tpc4_energies_array_notbp, color = color, histtype='step')
    ax4.set_title('TPC4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ax4.set_xlabel('Degrees')

    g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    bx1.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    bx1.set_title('TPC3 Track Length vs Sum Q')
    bx1.set_ylabel('Sum Q')
    bx1.set_xlabel('$\mu$m')
    bx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    bx2.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    bx2.set_title('TPC4 Track Length vs Sum Q')
    bx2.set_xlabel('$\mu$m')
    bx2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    g, ((cx1, cx2), (cx3, cx4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    cx1.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    cx1.set_title('TPC3 Track Length vs Sum Q (beampipe)')
    #cx1.set_xlim(-5000., 35000.)
    cx1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    cx1.set_ylabel('Sum Q')
    cx3.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp,
            color = color)
    cx3.set_title('TPC3 Track Length vs Sum Q (not beampipe)')
    cx3.set_xlabel('$\mu$m')
    cx3.set_ylabel('Sum Q')
    cx2.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    cx2.set_title('TPC4 Track Length vs Sum Q (beampipe)')
    cx2.set_xlim(-5000., 35000.)
    cx4.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp,
            color = color)
    cx4.set_title('TPC4 Track Length vs Sum Q (not beampipe)')
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


def energy_study(datapath):
    runs = run_names('LER_ToushekTPC')

    all_e = []
    n_e = []
    topa_e = []
    bota_e = []
    p_e = []
    x_e = []
    others_e = []
    for f in os.listdir(datapath):
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

        #all_e = []
        #n_e = []

        #for event in data:
        #    all_e.append(event.e_sum)
        #    if event.neutron == 1 : n_e.append(event.e_sum)

        for event in data:
            # TPC3
            for i in range(len(event.TPC3_PID_neutrons)):
                all_e.append(event.TPC3_sumQ[i])

                # Neutrons
                if(event.TPC3_PID_neutrons[i] == 1) :
                    n_e.append(event.TPC3_sumQ[i])

                # Top alphas
                if(event.TPC3_PID_alphas_top[i] == 1) :
                    topa_e.append(event.TPC3_sumQ[i])

                # Bottom alphas
                if(event.TPC3_PID_alphas_bottom[i] == 1) :
                    bota_e.append(event.TPC3_sumQ[i])

                # Protons
                if(event.TPC3_PID_protons[i] == 1):
                    p_e.append(event.TPC3_sumQ[i])

                # X-rays
                if(event.TPC3_PID_xrays[i] == 1):
                    x_e.append(event.TPC3_sumQ[i])

                # Others
                if(event.TPC3_PID_others[i] == 1):
                    others_e.append(event.TPC3_sumQ[i])
                    #trk = Hist2D(80,0.,80.,336,0.,336.)
                    #npoints = event.TPC3_npoints[i]
                    #cols = event.TPC3_hits_col[i]
                    #rows = event.TPC3_hits_row[i]
                    #tots = event.TPC3_hits_tot[i]
                    #for j in range(npoints):
                    #    trk.Fill(cols[j], rows[j], tots[j])
                    #trk.Draw('COLZ')
                    #input('well?')

            # TPC4
            for i in range(len(event.TPC4_PID_neutrons)):
                all_e.append(event.TPC4_sumQ[i])

                # Neutrons
                if(event.TPC4_PID_neutrons[i] == 1) :
                    n_e.append(event.TPC4_sumQ[i])

                # Top alphas
                if(event.TPC4_PID_alphas_top[i] == 1) :
                    topa_e.append(event.TPC4_sumQ[i])


                # Bottom alphas
                if(event.TPC4_PID_alphas_bottom[i] == 1) :
                    bota_e.append(event.TPC4_sumQ[i])

                # Protons
                if(event.TPC4_PID_protons[i] == 1):
                    p_e.append(event.TPC4_sumQ[i])

                # X-rays
                if(event.TPC4_PID_xrays[i] == 1):
                    x_e.append(event.TPC4_sumQ[i])

                # Others
                if(event.TPC4_PID_others[i] == 1):
                    others_e.append(event.TPC4_sumQ[i])
            
    #print(all_e)
    #print(n_e)
    #print(topa_e)
    #print(bota_e)
    #print(p_e)
    #print(x_e)
    #print(others_e)
    #input('well?')


    all_e = np.array(all_e)
    n_e = np.array(n_e)
    topa_e = np.array(topa_e)
    bota_e = np.array(bota_e)
    p_e = np.array(p_e)
    x_e = np.array(x_e)
    others_e = np.array(others_e)

    max_e = np.max(all_e)
    hist_all = Hist(100, 0., max_e)
    hist_all.fill_array(all_e)
    #hist_all.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    np_hist_all = np.histogram(all_e, bins=100, range=[0,max_e])

    np_hist_t = np.histogram(topa_e, bins=100, range=[0,max_e])
    np_hist_b = np.histogram(bota_e, bins=100, range=[0,max_e])
    np_hist_n = np.histogram(n_e, bins=100, range=[0,max_e])
    np_hist_p = np.histogram(p_e, bins=100, range=[0,max_e])
    np_hist_x = np.histogram(x_e, bins=100, range=[0,max_e])
    np_hist_o = np.histogram(others_e, bins=100, range=[0,max_e])

    max_ne = np.max(n_e)
    hist_n = Hist(100,0.,max_ne)
    hist_n.fill_array(n_e)
    #hist_n.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    np_hist_n = np.histogram(n_e, bins=100, range=[0,max_e])

    #print(np_hist_all,'\n',np_hist_n)

    all_e = np.array(all_e)
    #n_e = np.array(n_e)
    print('All energies:\n', np_hist_all)
    print('Neutron energies:\n', np_hist_n)
    divided_e = np.array([0.]*100)
    divided_bins = np.array([0.]*100)
    div_errs = np.array([0.]*100)

    t_errs = np.array([0.]*100)
    b_errs = np.array([0.]*100)
    n_errs = np.array([0.]*100)
    p_errs = np.array([0.]*100)
    x_errs = np.array([0.]*100)
    o_errs = np.array([0.]*100)

    for i in range(100):
        e_a = np_hist_all[0][i]
        n_a = np_hist_n[0][i]
        divided_e[i] = n_a/e_a if e_a != 0 else 0
        divided_bins[i] = (np_hist_n[1][i]+np_hist_n[1][i+1])/2.0
        div_errs[i] = (np.sqrt(n_a)/n_a)*divided_e[i] if n_a != 0 else 0

        t_err = np_hist_t[0][i]
        t_errs[i] = (np.sqrt(t_err)/t_err)

        n_err = np_hist_n[0][i]
        n_errs[i] = (np.sqrt(n_err)/n_err)

        b_err = np_hist_b[0][i]
        b_errs[i] = (np.sqrt(b_err)/b_err)

        p_err = np_hist_p[0][i]
        p_errs[i] = (np.sqrt(p_err)/p_err)

        x_err = np_hist_x[0][i]
        x_errs[i] = (np.sqrt(x_err)/x_err)

        o_err = np_hist_o[0][i]
        o_errs[i] = (np.sqrt(o_err)/o_err)

    #print(divided_e)
    #input('well?')

    #hist_n.Divide(hist_all)
    np_hist = hist2array(hist_n)
    #hist_n.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #print(np_hist)
    #print(divided_bins)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.hist(n_e, bins = 100)
    ax1.set_xlabel('Neutron Sum Q')
    ax3.hist(all_e, bins = 100)
    ax3.set_xlabel('All Sum Q')
    #print(len(divided_e), len(np_hist_n[1]))
    #ax2.scatter(divided_bins, divided_e)
    ax2.errorbar(divided_bins, divided_e, yerr=div_errs, fmt='o', capsize=0)
    ax2.set_xlabel('Neutron Sum Q')
    ax2.set_ylabel('Efficiency')

    # "Dot" histogram of all PID
    ax4.errorbar(divided_bins, np_hist_t[0], yerr=t_errs, fmt='o', capsize=0,
            color='orange')
    ax4.errorbar(divided_bins, np_hist_b[0], yerr=b_errs, fmt='o', capsize=0,
            color='yellow')
    ax4.errorbar(divided_bins, np_hist_n[0], yerr=n_errs, fmt='o', capsize=0)

    ax4.errorbar(divided_bins, np_hist_p[0], yerr=p_errs, fmt='o', capsize=0,
            color='purple')
    ax4.errorbar(divided_bins, np_hist_x[0], yerr=x_errs, fmt='o', capsize=0,
            color='black')
    ax4.errorbar(divided_bins, np_hist_o[0], yerr=o_errs, fmt='o', capsize=0,
            color='red')

    # Bar histogram of all PID
    #ax4.hist(topa_e, bins = 100, color='orange')
    #ax4.hist(bota_e, bins = 100, color='yellow')
    #ax4.hist(p_e, bins = 100, color='purple')
    #ax4.hist(x_e, bins = 100, color='black')
    #ax4.hist(others_e, bins = 100, color='red')
    #ax4.hist(n_e, bins = 100)
    plt.show()

    #hist_n.Draw()
    #input('well?')


def main():
    home = expanduser('~')

    ### Use BEAST v1 data
    #datapath = str(home) + '/BEAST/data/v1/'
    
    ### Use BEAST v2 data
    datapath = str(home) + '/BEAST/data/v2/'

    rate_vs_beamsize(datapath)
    neutron_study(datapath)
    #energy_study(datapath)
    #energy_study('~/BEAST/data/TPC/tpc4_th50_data_cordir16_1464505200_skim.root')


if __name__ == "__main__":
    main()

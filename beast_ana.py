import os

import numpy as np
import matplotlib as mpl
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

    sim_LER_ToushekTPC =(
            ['mc_beast_run_10002.root','mc_beast_run_10003.root','mc_beast_run_10004.root'])

    if run_name == 'LER_ToushekTPC': return LER_ToushekTPC
    if run_name == 'sim_LER_ToushekTPC': return sim_LER_ToushekTPC
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
    ts_3 = []
    ts_4 = []
    
    tot_counter = 0

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
                dedx_3 = event.TPC3_dEdx
                dedx_4 = event.TPC4_dEdx


                for i in range(len(tpc3_neutrons)):
                    #if tpc3_neutrons[i] == 1 :
                    if tpc3_neutrons[i] == 1 and dedx_3[i] > 0.35 :
                    #if tpc3_neutrons[i] == 1 and event.TPC3_dEdx[i] > 0.15 :
                        neutron_counter += 1
                        tot_counter += 1
                        counter_3 += 1
                        timestamps.append(event.ts)
                        ts_3.append(event.ts)

                for k in range(len(tpc4_neutrons)):
                    #if tpc4_neutrons[k] == 1 :
                    if tpc4_neutrons[k] == 1 and dedx_4[k] > 0.35 :
                    #if tpc4_neutrons[k] == 1 and event.TPC4_dEdx[k] > 0.15 :
                        neutron_counter += 1
                        tot_counter += 1
                        counter_4 += 1
                        timestamps.append(event.ts)
                        ts_4.append(event.ts)

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
        tot_counter += counter_3
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

    print('\nTotal # of neutrons =', tot_counter, '\n')
    ### Get delta_t distribution for each TPC
    ts_3 = np.array(ts_3)
    ts_4 = np.array(ts_4)

    delta_t3 = []
    delta_t4 = []

    for i in range(len(ts_3)):
        if i == len(ts_3) - 1: continue
        delt_t3 = ts_3[i+1]-ts_3[i]
        delta_t3.append(delt_t3)
    delta_t3 = np.array(delta_t3)

    for i in range(len(ts_4)):
        if i == len(ts_4) - 1: continue
        delt_t4 = ts_4[i+1]-ts_4[i]
        delta_t4.append(delt_t4)

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
    chi2.draw(minu, print_par=False)
    ax1.errorbar(avg_beamsize, avg_rate, yerr=rate_errs, fmt='o', color=color)
    ax1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    ax1.set_ylabel('Fast neutron rate (Hz)')
    ax1.set_xlim([0.0,0.030])
    ax1.set_ylim([0.0,0.2])
    #f.savefig('TPC_toushek_measurement.eps')
    f.savefig('TPC_toushek_measurement_sim_cuts.eps')
    plt.show()

    chi23 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_3,
            rate3_errs)
    minu3 = iminuit.Minuit(chi23)
    minu3.migrad()

    #g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    g = plt.figure()
    bx1 = g.add_subplot(111)
    chi23.draw(minu3, print_par=False)
    bx1.errorbar(avg_beamsize, rate_3, yerr=rate3_errs, fmt='o', color=color)
    bx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    bx1.set_ylabel('Fast neutron rate in Ch. 3 (Hz)')
    bx1.set_xlim([0.0,0.030])
    bx1.set_ylim([0.0,0.09])

    chi24 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_4,
            rate4_errs)
    minu4 = iminuit.Minuit(chi24)
    minu4.migrad()
    #g.savefig('TPC3_toushek_measurement.eps')
    g.savefig('TPC3_toushek_measurement_sim_cuts.eps')
    plt.show()

    h = plt.figure()
    bx2 = h.add_subplot(111)
    chi24.draw(minu4, print_par=False)
    bx2.errorbar(avg_beamsize, rate_4, yerr=rate4_errs, fmt='o', color=color)
    bx2.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    bx2.set_ylabel('Fast neutron rate in Ch. 4 (Hz)')
    bx2.set_xlim([0.0,0.030])
    bx2.set_ylim([0.0,0.09])
    #h.savefig('TPC4_toushek_measurement.eps')
    h.savefig('TPC4_toushek_measurement_sim_cuts.eps')
    plt.show()
    
    #l = plt.figure()
    #cx1 = l.add_subplot(111)
    #chi24.draw(minu4)
    #cx1.errorbar(avg_beamsize, rate_4, yerr=rate4_errs, fmt='o', color=color)
    #cx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    #cx1.set_ylabel('Fast neutron rate in Ch. 4 (Hz)')
    #cx1.set_xlim([0.0,0.030])
    #cx1.set_ylim([0.0,0.09])
    ##print(rate_3)
    ##print(rate_4)

    print(np.mean(delta_t3))
    print(np.mean(delta_t4))
    #l, (cx1, cx2 ) = plt.subplots(1, 2)
    #bins = 100
    #cx1.hist(delta_t3, bins=max(delta_t3), histtype='step', color=color)
    #cx1.set_title('$\Delta$$t$ of Sequential Neutron Events in Ch. 3')
    #cx1.set_xlabel('$\Delta$$t$ ($s$)')
    #cx1.set_yscale('log')
    #cx2.hist(delta_t4, bins=max(delta_t4), histtype='step', color=color)
    #cx2.set_title('$\Delta$$t$ of Sequential Neutron Events in Ch. 4')
    #cx2.set_xlabel('$\Delta$$t$ ($s$)')
    #cx2.set_yscale('log')
    #plt.show()

    ### Plot figures individually
    bins = 100
    plt.hist(delta_t3, bins=max(delta_t3), histtype='step', color=color)
    plt.xlabel('$\Delta$$t$ ($s$)')
    plt.ylabel('Events per bin')
    plt.yscale('log')
    plt.savefig('tpc3_deltat.eps')
    plt.show()

    plt.hist(delta_t4, bins=max(delta_t4), histtype='step', color=color)
    plt.xlabel('$\Delta$$t$ ($s$)')
    plt.ylabel('Events per bin')
    plt.yscale('log')
    plt.savefig('tpc4_deltat.eps')
    plt.show()

def sim_rate_vs_beamsize(datapath):
    runs = run_names('sim_LER_ToushekTPC')

    avg_rates = []
    rates_3 = []
    rates_4 = []
    avg_inv_beamsizes = []
    invbs_errs = []
    rate_errs = []
    rate3_errs = []
    rate4_errs = []
    ts_3 = []
    ts_4 = []
    
    tot_counter = 0


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

        data = root2rec(ifile, 'tout')

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

                tpc3_neutrons = event.TPC_rate_av[0][0]
                tpc4_neutrons = event.TPC_rate_av[1][0]
                #dedx_3 = event.TPC3_dEdx
                #dedx_4 = event.TPC4_dEdx
                neutron_counter += tpc3_neutrons
                neutron_counter += tpc4_neutrons
                tot_counter += tpc3_neutrons
                tot_counter += tpc4_neutrons

                counter_3 += tpc3_neutrons
                counter_4 += tpc4_neutrons


                #for i in range(len(tpc3_neutrons)):
                #    #if tpc3_neutrons[i] == 1 :
                #    #if tpc3_neutrons[i] == 1 and dedx_3[i] > 0.35 :
                #    #if tpc3_neutrons[i] == 1 and event.TPC3_dEdx[i] > 0.15 :
                #        neutron_counter += 1
                #        tot_counter += 1
                #        counter_3 += 1
                #        timestamps.append(event.ts)
                #        ts_3.append(event.ts)

                #for k in range(len(tpc4_neutrons)):
                #    #if tpc4_neutrons[k] == 1 :
                #    #if tpc4_neutrons[k] == 1 and dedx_4[k] > 0.35 :
                #    #if tpc4_neutrons[k] == 1 and event.TPC4_dEdx[k] > 0.15 :
                #        neutron_counter += 1
                #        tot_counter += 1
                #        counter_4 += 1
                #        timestamps.append(event.ts)
                #        ts_4.append(event.ts)

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
        tot_counter += counter_3
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

    print('\nTotal # of neutrons =', tot_counter, '\n')
    ### Get delta_t distribution for each TPC
    ts_3 = np.array(ts_3)
    ts_4 = np.array(ts_4)

    delta_t3 = []
    delta_t4 = []

    for i in range(len(ts_3)):
        if i == len(ts_3) - 1: continue
        delt_t3 = ts_3[i+1]-ts_3[i]
        delta_t3.append(delt_t3)
    delta_t3 = np.array(delta_t3)

    for i in range(len(ts_4)):
        if i == len(ts_4) - 1: continue
        delt_t4 = ts_4[i+1]-ts_4[i]
        delta_t4.append(delt_t4)

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
    chi2.draw(minu, print_par=False)
    ax1.errorbar(avg_beamsize, avg_rate, yerr=rate_errs, fmt='o', color=color)
    ax1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    ax1.set_ylabel('Fast neutron rate (Hz)')
    #ax1.set_xlim([0.0,0.030])
    #ax1.set_ylim([0.0,0.2])
    #f.savefig('TPC_toushek_measurement.eps')
    f.savefig('TPC_toushek_measurement_sim.eps')
    plt.show()

    chi23 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_3,
            rate3_errs)
    minu3 = iminuit.Minuit(chi23)
    minu3.migrad()

    #g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    g = plt.figure()
    bx1 = g.add_subplot(111)
    chi23.draw(minu3, print_par=False)
    bx1.errorbar(avg_beamsize, rate_3, yerr=rate3_errs, fmt='o', color=color)
    bx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    bx1.set_ylabel('Fast neutron rate in Ch. 3 (Hz)')
    bx1.set_xlim([0.0,0.030])
    bx1.set_ylim([0.0,0.09])

    chi24 = probfit.Chi2Regression(probfit.linear, avg_beamsize, rate_4,
            rate4_errs)
    minu4 = iminuit.Minuit(chi24)
    minu4.migrad()
    #g.savefig('TPC3_toushek_measurement.eps')
    g.savefig('TPC3_toushek_measurement_sim.eps')
    plt.show()

    h = plt.figure()
    bx2 = h.add_subplot(111)
    chi24.draw(minu4, print_par=False)
    bx2.errorbar(avg_beamsize, rate_4, yerr=rate4_errs, fmt='o', color=color)
    bx2.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    bx2.set_ylabel('Fast neutron rate in Ch. 4 (Hz)')
    bx2.set_xlim([0.0,0.030])
    bx2.set_ylim([0.0,0.09])
    #h.savefig('TPC4_toushek_measurement.eps')
    h.savefig('TPC4_toushek_measurement_sim.eps')
    plt.show()
    
    #l = plt.figure()
    #cx1 = l.add_subplot(111)
    #chi24.draw(minu4)
    #cx1.errorbar(avg_beamsize, rate_4, yerr=rate4_errs, fmt='o', color=color)
    #cx1.set_xlabel('Inverse Beamsize ($\mu$$m$$^{-1}$)')
    #cx1.set_ylabel('Fast neutron rate in Ch. 4 (Hz)')
    #cx1.set_xlim([0.0,0.030])
    #cx1.set_ylim([0.0,0.09])
    ##print(rate_3)
    ##print(rate_4)

    print(np.mean(delta_t3))
    print(np.mean(delta_t4))
    #l, (cx1, cx2 ) = plt.subplots(1, 2)
    #bins = 100
    #cx1.hist(delta_t3, bins=max(delta_t3), histtype='step', color=color)
    #cx1.set_title('$\Delta$$t$ of Sequential Neutron Events in Ch. 3')
    #cx1.set_xlabel('$\Delta$$t$ ($s$)')
    #cx1.set_yscale('log')
    #cx2.hist(delta_t4, bins=max(delta_t4), histtype='step', color=color)
    #cx2.set_title('$\Delta$$t$ of Sequential Neutron Events in Ch. 4')
    #cx2.set_xlabel('$\Delta$$t$ ($s$)')
    #cx2.set_yscale('log')
    #plt.show()

    ### Plot figures individually
    bins = 100
    plt.hist(delta_t3, bins=max(delta_t3), histtype='step', color=color)
    plt.xlabel('$\Delta$$t$ ($s$)')
    plt.ylabel('Events per bin')
    plt.yscale('log')
    plt.savefig('tpc3_deltat_sim.eps')
    plt.show()

    plt.hist(delta_t4, bins=max(delta_t4), histtype='step', color=color)
    plt.xlabel('$\Delta$$t$ ($s$)')
    plt.ylabel('Events per bin')
    plt.yscale('log')
    plt.savefig('tpc4_deltat_sim.eps')
    plt.show()

def peter_toushek(datapath):

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

        neutrons = 0
        neutrons3 = 0
        neutrons4 = 0
        data = root2rec(ifile)

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
                if len(event.TPC3_N_neutrons) > 0 :
                    n3 += event.TPC3_N_neutrons[0]

                #n4 += (event.TPC4_N_neutrons[0] if
                #        len(event.TPC4_N_neutrons) > 0 else 0)

                if len(event.TPC4_N_neutrons) > 0 :
                    n4 += event.TPC4_N_neutrons[0]

                current = event.SKB_LER_current[0]
                local_pressure = event.SKB_LER_pressures_local[0]
                beam_size = event.SKB_LER_beamSize_xray_Y[0]

                ip += local_pressure * current
                ps += beam_size * local_pressure
                x += current / (local_pressure * beam_size)


        neutrons3 = n3 / counter #* 1000.
        neutrons4 = n4 / counter #* 1000.
        neutrons = ((neutrons3 +  neutrons4) / counter)# * 1000.

        mean_ip = ip / counter
        ip_err = mean_ip / np.sqrt(counter)
        mean_x = x / counter
        x_err = mean_x / np.sqrt(counter)
        #mean_ps = ps / len(data)

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
    input('well?')

    if root_style == True :
        color = 'black'
    elif root_style == False :
        color = 'blue'

    f = plt.figure()
    ax1 = f.add_subplot(111)
    chi2.draw(minu)
    #chi2.draw(minu, print_par=False)
    ax1.errorbar(peter_x, peter_y, xerr=x_errs, yerr=y_errs, fmt='o', color=color)
    #ax1.scatter(peter_x, peter_y, color=color)
    ax1.set_xlabel('current / (pressure * beamsize)')
    ax1.set_ylabel('n_neutrons/(current * pressure)')
    ax1.set_xlim([0.0,4.0E7])
    ax1.set_ylim([0.0,1000])
    #f.savefig('TPC_toushek_measurement.eps')
    plt.show()

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
    #g.savefig('TPC3_toushek_measurement.eps')
    #plt.show()

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
    plt.savefig('tpc3_phi_weighted.eps')
    plt.show()

    plt.hist(tpc3phi_array, phi_bins, range=[-100,100],
            color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Neutron Recoil $\phi$')
    plt.savefig('tpc3_phi_unweighted.eps')
    plt.show()

    plt.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_energies_array, color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    plt.savefig('tpc4_phi_weighted.eps')
    plt.show()

    plt.hist(tpc4phi_array, phi_bins, range=[-100,100],
            color = color, histtype='step')
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Neutron Recoil $\phi$')
    plt.savefig('tpc4_phi_unweighted.eps')
    plt.show()

    plt.hist(tpc3theta_array, theta_bins, weights = tpc3_energies_array,
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Energy Weighted Neutron Recoil $\\theta$')
    plt.savefig('tpc3_theta_weighted.eps')
    plt.show()

    plt.hist(tpc3theta_array, theta_bins, 
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC3 Neutron Recoil $\\theta$')
    plt.savefig('tpc3_theta_unweighted.eps')
    plt.show()

    plt.hist(tpc4theta_array, theta_bins, weights = tpc4_energies_array,
            color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Energy Weighted Neutron Recoil $\\theta$')
    plt.savefig('tpc4_theta_weighted.eps')
    plt.show()

    plt.hist(tpc4theta_array, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    #plt.title('TPC4 Neutron Recoil $\\theta$')
    plt.savefig('tpc4_theta_unweighted.eps')
    plt.show()

    plt.hist(tpc3theta_array_beampipe, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc3_theta_unweighted_bp.eps')
    plt.show()

    plt.hist(tpc4theta_array_beampipe, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc4_theta_unweighted_bp.eps')
    plt.show()

    plt.hist(tpc3theta_array_notbp, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc3_theta_unweighted_nbp.eps')
    plt.show()

    plt.hist(tpc4theta_array_notbp, theta_bins, color = color, histtype='step')
    plt.xlabel('$\\theta$ ($^{\circ}$)')
    plt.ylabel('Events per bin')
    plt.savefig('tpc4_theta_unweighted_nbp.eps')
    plt.show()

    plt.scatter(tpc3_tlengths_array, tpc3_energies_array, color = color)
    #plt.set_title('Ch. 3 Track Length vs Sum Q')
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_all.eps')
    plt.show()

    plt.scatter(tpc4_tlengths_array, tpc4_energies_array, color = color)
    #plt.set_title('Ch. 4 Track Length vs Sum Q')
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_all.eps')
    plt.show()
    
    plt.scatter(tpc3_tlengths_array_bp, tpc3_energies_array_bp, color = color)
    #plt.set_title('Ch. 3 Track Length vs Sum Q (beampipe)')
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.savefig('tpc3_dedx_bp.eps')
    plt.show()

    plt.scatter(tpc3_tlengths_array_notbp, tpc3_energies_array_notbp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.xlabel('$\mu$m')
    plt.ylabel('Detected Charge (q)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_nbp.eps')
    plt.show()

    plt.scatter(tpc4_tlengths_array_bp, tpc4_energies_array_bp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_bp.eps')
    plt.show()

    plt.scatter(tpc4_tlengths_array_notbp, tpc4_energies_array_notbp, color = color)
    plt.xlim(-5000., 35000.)
    plt.ylim(-1E7, 7E7)
    plt.ylabel('Detected Charge (q)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_nbp.eps')
    plt.show()

    plt.scatter(tpc3_energies_array, tpc3phi_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.savefig('tpc3_evsphi_scatter.eps')
    plt.show()

    plt.hist2d(tpc3_energies_array, tpc3phi_array, bins=(25,phi_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.colorbar()
    plt.savefig('tpc3_evsphi_heatmap.eps')
    plt.show()

    plt.scatter(tpc3_energies_array, tpc3theta_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc3_evstheta_scatter.eps')
    plt.show()

    plt.hist2d(tpc3_energies_array, tpc3theta_array, bins=(25,theta_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc3_evstheta_heatmap.eps')
    plt.show()

    plt.scatter(tpc4_energies_array, tpc4phi_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.savefig('tpc4_evsphi_scatter.eps')
    plt.show()

    plt.hist2d(tpc4_energies_array, tpc4phi_array, bins=(25,phi_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\phi$ ($^{\circ}$)')
    plt.ylim(-100.0,100.0)
    plt.colorbar()
    plt.savefig('tpc4_evsphi_heatmap.eps')
    plt.show()

    plt.scatter(tpc4_energies_array, tpc4theta_array, color = color)
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc4_evstheta_scatter.eps')
    plt.show()

    plt.hist2d(tpc4_energies_array, tpc4theta_array, bins=(25,theta_bins))
    plt.xlabel('Detected Charge (q)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc4_evstheta_heatmap.eps')
    plt.show()

    plt.scatter(tpc3phi_array, tpc3theta_array, color = color)
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylim(0.,180.0)
    plt.savefig('tpc3_thetavsphi_scatter.eps')
    plt.show()

    plt.hist2d(tpc3phi_array, tpc3theta_array, bins=(phi_bins,theta_bins))
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc3_thetavsphi_heatmap.eps')
    plt.show()

    plt.scatter(tpc4phi_array, tpc4theta_array, color = color)
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.savefig('tpc4_thetavsphi_scatter.eps')
    plt.show()

    plt.hist2d(tpc4phi_array, tpc4theta_array, bins=(phi_bins,theta_bins))
    plt.xlabel('$\phi$ ($^{\circ}$)')
    plt.xlim(-100.0,100.0)
    plt.ylabel('$\\theta$ ($^{\circ}$)')
    plt.ylim(0.,180.0)
    plt.colorbar()
    plt.savefig('tpc4_thetavsphi_heatmap.eps')
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
    plt.savefig('tpc3_recoil_energies.eps')
    plt.show()

    plt.hist(tpc3_kev_array_bp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc3_recoil_energies_bp.eps')
    plt.show()

    plt.hist(tpc3_kev_array_notbp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc3_recoil_energies_notbp.eps')
    plt.show()

    plt.hist(tpc4_kev_array_bp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies_bp.eps')
    plt.show()

    plt.hist(tpc4_kev_array_notbp, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies_notbp.eps')
    plt.show()

    plt.hist(tpc4_kev_array, bins=25, color=color, histtype='step')
    plt.xlabel('Detected Energy (keV)')
    plt.ylabel('Events per bin')
    plt.xlim(0, 1600)
    plt.yscale('log')
    plt.ylim(0,4E2)
    plt.savefig('tpc4_recoil_energies.eps')
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
    ##f.savefig('theta_phi_weighted.eps')

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
    ##f.savefig('theta_phi_weighted-beampipe.eps')

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
    ##f.savefig('theta_phi_weighted-outside_beampipe.eps')

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
    ##f.savefig('theta_phi_unweighted.eps')

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
    ##f.savefig('theta_phi_unweighted-inside_beampipe.eps')

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
    ##f.savefig('theta_phi_unweighted-outside_beampipe.eps', dpi=200)

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


#def energy_study(datapath):
def energy_study(gain_path):
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
                if event.neutron == 1 : n_e.append(event.e_sum)
                if event.neutron == 1 and event.detnb == 3:
                    n_3.append(event.e_sum)
                if event.neutron == 1 and event.detnb == 4:
                    n_4.append(event.e_sum)
                #if event.top_alpha == 1 : topa_e.append(event.e_sum)
                #if event.bottom_alpha == 1 : bota_e.append(event.e_sum)
                #if event.proton == 1 : p_e.append(event.e_sum)


            ### Format for BEAST v2 ntuples
            #for event in data:
            #    # TPC3
            #    for i in range(len(event.TPC3_PID_neutrons)):
            #        print(event.dtype.names)
            #        input('well?')
            #        all_e.append(event.TPC3_sumQ[i])

            #        # Neutrons
            #        if(event.TPC3_PID_neutrons[i] == 1) :
            #            n_e.append(event.TPC3_sumQ[i])
            #            n_3.append(event.TPC3_sumQ[i])

            #        # Top alphas
            #        if(event.TPC3_PID_alphas_top[i] == 1) :
            #            topa_e.append(event.TPC3_sumQ[i])

            #        # Bottom alphas
            #        if(event.TPC3_PID_alphas_bottom[i] == 1) :
            #            bota_e.append(event.TPC3_sumQ[i])

            #        # Protons
            #        if(event.TPC3_PID_protons[i] == 1):
            #            p_e.append(event.TPC3_sumQ[i])

            #        # X-rays
            #        if(event.TPC3_PID_xrays[i] == 1):
            #            x_e.append(event.TPC3_sumQ[i])

            #        # Others
            #        if(event.TPC3_PID_others[i] == 1):
            #            others_e.append(event.TPC3_sumQ[i])
            #            #trk = Hist2D(80,0.,80.,336,0.,336.)
            #            #npoints = event.TPC3_npoints[i]
            #            #cols = event.TPC3_hits_col[i]
            #            #rows = event.TPC3_hits_row[i]
            #            #tots = event.TPC3_hits_tot[i]
            #            #for j in range(npoints):
            #            #    trk.Fill(cols[j], rows[j], tots[j])
            #            #trk.Draw('COLZ')
            #            #input('well?')

            #    # TPC4
            #    for i in range(len(event.TPC4_PID_neutrons)):
            #        print(event.dtype.names)
            #        input('well?')
            #        all_e.append(event.TPC4_sumQ[i])

            #        # Neutrons
            #        if(event.TPC4_PID_neutrons[i] == 1) :
            #            n_e.append(event.TPC4_sumQ[i])
            #            n_4.append(event.TPC4_sumQ[i])

            #        # Top alphas
            #        if(event.TPC4_PID_alphas_top[i] == 1) :
            #            topa_e.append(event.TPC4_sumQ[i])


            #        # Bottom alphas
            #        if(event.TPC4_PID_alphas_bottom[i] == 1) :
            #            bota_e.append(event.TPC4_sumQ[i])

            #        # Protons
            #        if(event.TPC4_PID_protons[i] == 1):
            #            p_e.append(event.TPC4_sumQ[i])

            #        # X-rays
            #        if(event.TPC4_PID_xrays[i] == 1):
            #            x_e.append(event.TPC4_sumQ[i])

            #        # Others
            #        if(event.TPC4_PID_others[i] == 1):
            #            others_e.append(event.TPC4_sumQ[i])
            
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
    #print(all_e)
    #input('well?')
    #topa_e = np.array(topa_e)
    #bota_e = np.array(bota_e)
    #p_e = np.array(p_e)
    #x_e = np.array(x_e)
    #others_e = np.array(others_e)

    n_3 = np.array(n_3)
    n_4 = np.array(n_4)

    ### Bar histograms for TPC energies
    #h, (cx1) = plt.subplots(1, 1)
    #cx1.hist(n_e, bins = 100, color='black', histtype='step')
    #    
    #cx1.set_xlabel('Recoil Energy (KeV)')
    #cx1.set_ylabel('Events per Bin')
    #cx1.set_yscale('log')
    #h.savefig('neutron_energies.eps') 
    #plt.show()
    
    h, (cx1) = plt.subplots(1, 1)
    cx1.hist(n_3, bins = 100, color='black', histtype='step')
        
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    h.savefig('tpc3_neutron_energies.eps') 
    plt.show()

    h, (cx1) = plt.subplots(1, 1)
    cx1.hist(n_4, bins = 100, color='black', histtype='step')
        
    cx1.set_xlabel('Recoil Energy (KeV)')
    cx1.set_ylabel('Events per Bin')
    cx1.set_yscale('log')
    h.savefig('tpc4_neutron_energies.eps') 
    plt.show()

    input('done plotting')
    max_e = np.max(all_e)
    hist_all = Hist(100, 0., max_e)
    hist_all.fill_array(all_e)
    #hist_all.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    np_hist_all = np.histogram(all_e, bins=500, range=[0,max_e])

    #np_hist_t = np.histogram(topa_e, bins=100, range=[0,max_e])
    #np_hist_b = np.histogram(bota_e, bins=100, range=[0,max_e])
    #np_hist_n = np.histogram(n_e, bins=100, range=[0,max_e])
    #np_hist_p = np.histogram(p_e, bins=100, range=[0,max_e])
    #np_hist_x = np.histogram(x_e, bins=100, range=[0,max_e])
    #np_hist_o = np.histogram(others_e, bins=100, range=[0,max_e])

    max_ne = np.max(n_e)

    np_hist_n3 = np.histogram(n_3, bins=500, range=[0,max_ne])
    np_hist_n4 = np.histogram(n_4, bins=500, range=[0,max_ne])

    #hist_n = Hist(100,0.,max_ne)
    #hist_n.fill_array(n_e)
    #hist_n.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    np_hist_n = np.histogram(n_e, bins=500, range=[0,max_e])

    #print(np_hist_all,'\n',np_hist_n)

    all_e = np.array(all_e)
    #n_e = np.array(n_e)
    #print('All energies:\n', np_hist_all)
    #print('Neutron energies:\n', np_hist_n)
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
    #hist_n.Divide(hist_all)
    #np_hist = hist2array(hist_n)
    #hist_n.Draw()
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #input('well?')
    #print(np_hist)
    #print(divided_bins)

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
    h.savefig('neutron_efficiency_energy.eps')
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

def gain_study(gain_path):
    ### For looking at one file
    #home = expanduser('~')
    ##t3file = (str(home) +
    ##'/BEAST/data/TPC/tpc_toushekrun/2016-05-29/TPC3/2016-05-29/tpc3_th50_data_cordir15_1464501600_skim.root')
    #t3file = (str(home) +
    #'/BEAST/data/TPC/tpc_toushekrun/2016-05-29/TPC3/2016-05-29/tpc3_th50_data_cordir12_1464490800_skim.root')

    ##t4file = (str(home) +
    ##'/BEAST/data/TPC/tpc_toushekrun/2016-05-29/TPC4/2016-05-29/tpc4_th50_data_cordir15_1464501600_skim.root')
    #t4file = (str(home) +
    #'/BEAST/data/TPC/tpc_toushekrun/2016-05-29/TPC4/2016-05-29/tpc4_th50_data_cordir12_1464490800_skim.root')
    #data_3 = root2rec(t3file)
    #data_4 = root2rec(t4file)
    
    t3_etop = []
    t3_ebottom = []
    t3t_ts = []
    t3b_ts = []
    t4_etop = []
    t4_ebottom = []
    t4t_ts = []
    t4b_ts = []

    #t3_start = 1464487449.294441
    ##t3_start = 1464447730.5185933
    #t4_start = 1464487405.9908841
    #t4_start = 1464447730.5185933 
    tmax = 1464505369.0
    tmin = 1464485170.0
    t3_start = tmin
    t4_start = tmin

    print(gain_path)
    #input('well?')

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
            #if int(strs[-2]) in good_files : print('yes!')
            #else : print('nope!')
            if int(strs[-2]) not in good_files : continue
            #input('well?')
            r_file = str(subdir) + str('/') + str(f)
            data = root2rec(r_file)
            #if np.all(data.tstamp < tmin) or np.all(data.tstamp > tmax) :
            #    continue
            #print(r_file)
            #print(data.detnb)
            #input('well?')



            for event in data:
                if event.tstamp > tmax or event.tstamp < tmin: continue
                if (data.detnb[0] == 3) :
                    #print('Is it det3?', data.detnb[0])
                    #input('well?')

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
                    #print('Is it det4?', data.detnb[0])
                    #input('well?')
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
    #print(bins_t4t)
    #print(len(bins_t4t))
    #print(bin_centers_t4t)
    #print(len(bin_centers_t4t))
    #input('well?')
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

    #t3_tr = np.std(t3_etop)/sqrt(len(t3_etop))
    #t3_br = np.std(t3_ebottom)/sqrt(len(t3_ebottom))
    #t4_tr = np.std(t4_etop)/sqrt(len(t4_etop))
    #t4_br = np.std(t4_ebottom)/sqrt(len(t4_ebottom))

    print(s % (t3_tm, t3_tr, t3_bm, t3_br, t4_tm, t4_tr, t4_bm, t4_br))
    input('well?')
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

    #t3_tr = np.std(result_t3['mean'].values[:-10])/sqrt(len(result_t3['mean'].values[:-10]))
    #t3_br = np.std(result_b3['mean'].values[:-10])/sqrt(len(result_b3['mean'].values[:-10]))
    #t4_tr = np.std(result_t4['mean'].values[:-10])/sqrt(len(result_t4['mean'].values[:-10]))
    #t4_br = np.std(result_b4['mean'].values[:-10])/sqrt(len(result_b4['mean'].values[:-10]))

    #t3_tm = df3t.t3_etop
    #print(t3_tm)

    print(s % (t3_tm, t3_tr, t3_bm, t3_br, t4_tm, t4_tr, t4_bm, t4_br))
    input('well?')

    #g, (bx1, bx2) = plt.subplots(1, 2)
    #result_t3.plot.scatter(x='x', y='mean', yerr='sem', ax=bx1,
    #    label='Top', color='black')
    #result_b3.plot.scatter(x='x', y='mean', yerr='sem', ax=bx1,
    #    label='Bottom')
    #bx1.set_title('Alpha Sum Q vs Time in Ch. 3 (profile)')
    #bx1.legend(loc='best')
    #bx1.set_ylim(0,5E7)
    #bx1.set_xlabel('Time (s)')
    #bx1.set_ylabel('Sum Q')
    #result_t4.plot(x='x', y='mean', yerr='sem', ax=bx2,
    #   label='Top', color='black')
    #result_b4.plot(x='x', y='mean', yerr='sem', ax=bx2,
    #    label='Bottom')
    #bx2.set_title('Alpha Sum Q vs Time in TPC4 (profile)')
    #bx2.legend(loc='lower left')
    #bx2.set_ylim(0,5E7)
    #bx2.set_xlabel('Time (s)')
    #bx2.set_ylabel('Sum Q')
    #plt.show()

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
    h.savefig('tpc3_gainstability.eps')
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
    h.savefig('tpc4_gainstability.eps')
    plt.show()

def pid_study(datapath):
    tlengths_all = []
    ch3_tlengths_all = []
    ch4_tlengths_all = []

    tlengths_n = []
    ch3_tlengths_n = []
    ch4_tlengths_n = []

    energies_all = []
    ch3_energies_all = []
    ch4_energies_all = []

    energies_n = []
    ch3_energies_n = []
    ch4_energies_n = []

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

    for subdir, dirs, files in os.walk(datapath):
        for f in files:
            strs = f.split('_')
            if int(strs[-2]) not in good_files : continue
            r_file = str(subdir) + str('/') + str(f)
            print(r_file)
            data = root2rec(r_file)
            for event in data:
                    if event.hitside == 0 and event.neutron == 0 :
                        tlengths_all.append(event.t_length)
                        energies_all.append(event.e_sum)
                        if (event.detnb == 3) :
                            ch3_tlengths_all.append(event.t_length)
                            ch3_energies_all.append(event.e_sum)
                        elif (event.detnb == 4) :
                            ch4_tlengths_all.append(event.t_length)
                            ch4_energies_all.append(event.e_sum)

                    elif event.neutron == 1 :
                        tlengths_n.append(event.t_length)
                        energies_n.append(event.e_sum)
                        if (event.detnb == 3) :
                            ch3_tlengths_n.append(event.t_length)
                            ch3_energies_n.append(event.e_sum)
                        elif (event.detnb == 4) :
                            ch4_tlengths_n.append(event.t_length)
                            ch4_energies_n.append(event.e_sum)

    tlengths_all = np.array(tlengths_all )
    ch3_tlengths_all = np.array(ch3_tlengths_all)
    ch4_tlengths_all = np.array(ch4_tlengths_all)
    tlengths_n = np.array(tlengths_n)
    ch3_tlengths_n = np.array(ch3_tlengths_n)
    ch4_tlengths_n = np.array(ch4_tlengths_n)

    energies_all = np.array(energies_all)
    ch3_energies_all = np.array(ch3_energies_all)
    ch4_energies_all = np.array(ch4_energies_all)
    energies_n = np.array(energies_n)
    ch3_energies_n = np.array(ch3_energies_n)
    ch4_energies_n = np.array(ch4_energies_n)

    gain1 = 30.0
    gain2 = 50.0
    w = 35.075
    energies_all = energies_all/(gain1 * gain2)*w*1E-3
    ch3_energies_all = ch3_energies_all/(gain1 * gain2)*w*1E-3
    ch4_energies_all = ch4_energies_all/(gain1 * gain2)*w*1E-3
    energies_n = energies_n/(gain1 * gain2)*w*1E-3
    ch3_energies_n = ch3_energies_n/(gain1 * gain2)*w*1E-3
    ch4_energies_n = ch4_energies_n/(gain1 * gain2)*w*1E-3

    plt.scatter(tlengths_all, energies_all, facecolors='none', edgecolors='red')
    plt.scatter(tlengths_n, energies_n,color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc_dedx_cuts_combined.eps')
    plt.show()

    plt.scatter(ch3_tlengths_all, ch3_energies_all, facecolors='none', edgecolors='red')
    plt.scatter(ch3_tlengths_n, ch3_energies_n, color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc3_dedx_cuts.eps')
    plt.show()

    plt.scatter(ch4_tlengths_all, ch4_energies_all, facecolors='none', edgecolors='red')
    plt.scatter(ch4_tlengths_n, ch4_energies_n, color='black')
    plt.xlim(-5000., 35000.)
    plt.ylim(-500, 2000)
    plt.ylabel('Detected Energy (keV)')
    plt.xlabel('$\mu$m')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('tpc4_dedx_cuts.eps')
    plt.show()

def event_inspection(datapath):
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

    #rows = 45
    #cols = 49
    #fig, axs = plt.subplots(rows, cols)
    #n_neutrons = 0

    #counter = 0
    x_bins = 80
    y_bins = 336
    for subdir, dirs, files in os.walk(datapath):
        for f in files:
            strs = f.split('_')
            if int(strs[-2]) not in good_files : continue
            r_file = str(subdir) + str('/') + str(f)
            print(r_file)
            data = root2rec(r_file)
            n_neutrons = sum(data.neutron)
            grid = int(np.sqrt(n_neutrons)) + 1
            print(n_neutrons, grid)
            fig, axs = plt.subplots(grid, grid)
            counter = 0
            for event in data:
                if event.neutron == 1 :
                    #plt.hist2d(event.col, event.row, 
                    #plt.hist2d(event.col, event.row, bins = (
                    #    range(0, x_bins, 1) , range(0, y_bins, 1) ), weights = event.tot + 1)
                    axs.flat[counter].hist2d( event.col, event.row, bins = (
                            range(0, x_bins, 1) , range(0, y_bins, 1) ),
                            weights = event.tot, cmin = 0, cmax = 15)
                    axs.flat[counter].set_frame_on(False)
                    axs.flat[counter].get_yaxis().set_visible(False)
                    axs.flat[counter].get_xaxis().set_visible(False)
                    axs.flat[counter].set_xlim(0, x_bins)
                    axs.flat[counter].set_ylim(0, y_bins)
                    #axs.flat[counter].clim(0,15)
                    cbar = mpl.colorbar.ColorbarBase(axs.flat[counter],
                            norm=mpl.colors.Normalize(vmin=0, vmax = 15) )
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
            diff = grid**2 - n_neutrons
            for i in range(1,diff+1):
                axs.flat[-i].set_frame_on(False)
                axs.flat[-i].get_yaxis().set_visible(False)
                axs.flat[-i].get_xaxis().set_visible(False)
                
            #fig.colorbar(_, cax=axs)
            
            #plt.colorbar()
            fname = f.split('.')[0]
            pname = fname + str('.eps')
            print(pname)
            fig.savefig(pname, format='eps')
    #print(n_neutrons)
    #input('well?')

def main():

    home = expanduser('~')

    ### Use BEAST v1 data
    #datapath = str(home) + '/BEAST/data/v1/'
    
    ### Use BEAST v2 data
    datapath = str(home) + '/BEAST/data/v2/'
    simpath = str(home) + '/BEAST/sim/'

    rate_vs_beamsize(datapath)
    #sim_rate_vs_beamsize(simpath)
    #peter_toushek(datapath)
    #neutron_study(datapath)
    #energy_study(datapath)

    inpath = str(home) + '/BEAST/data/TPC/tpc_toushekrun/2016-05-29/'
    #energy_study(inpath)
    #gain_study(inpath)
    #pid_study(inpath)
    #event_inspection(inpath)


if __name__ == "__main__":
    main()

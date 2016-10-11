import os
import seaborn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc

from rootpy.plotting import Hist, Hist2D
from rootpy.plotting.style import set_style
import rootpy.plotting.root2matplotlib as rplt

from root_numpy import root2rec, hist2array
from ROOT import TFile, TH1F

from os.path import expanduser

np.set_printoptions(suppress=True, precision=2)

def run_names(run_name):
    LER_Beamsize = []

    LER_Fill = [6100.1,6100.2,6100.3,6100.4]
    HER_Fill = []

    LER_Vacuumbump = []
    HER_Vacuumbump = []

    HER_Toushek = []
    LER_Toushek = []

    HER_Chromaicity = []

    HER_Injection = []
    LER_Injection = []
    
    HER_ToushekTPC = ['BEAST_run5100.root', 'BEAST_run9001.root']

    LER_ToushekTPC =(
            ['BEAST_run10000.root','BEAST_run10001.root','BEAST_run10002.root','BEAST_run10003.root','BEAST_run10004.root'])

    if run_name == 'LER_ToushekTPC': return LER_ToushekTPC
    elif run_name == 'HER_ToushekTPC': return HER_ToushekTPC


def rate_vs_beamsize():
    stuff = 0


def neutron_study():
    stuff = 0

def main():
    home = expanduser('~')
    datapath = str(home) + '/BEAST/data/v1/'

    tpc3_phis = []
    tpc3_thetas = []
    tpc3_energies = []
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
    tpc4_sumtot_bp = []
    tpc4_sumtot_notbp = []
    tpc4_sumtot = []
    tpc4_tlengths = []
    tpc4_tlengths_bp = []
    tpc4_tlengths_notbp = []
    tpc4_thetas_beampipe = []
    tpc4_thetas_notbp = []

    beamsize = []

    rate_vs_beamsize = []
    rate_vs_beamsize_tpc3 = []
    rate_vs_beamsize_tpc4 = []

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

                    size = event.SKB_LER_beamSize_xray_Y[0]
                    if size < 300 :
                        beamsize.append(event.SKB_LER_beamSize_xray_Y[0])
                    
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

                    tpc3_energies.append(event.TPC3_sumE[i])
                    tpc3_sumtot.append(event.TPC3_sumTOT[i])
                    tpc3_tlengths.append(event.TPC3_sumTOT[i]/event.TPC3_dEdx[i])

                    #print(event.SKB_LER_beamSize_xray_X)
                    #print(event.SKB_LER_beamSize_SR_Y)
                    #print(event.SKB_LER_beamSize_SR_X)
                    #input('well?')

                    ### Select beampipe (+- 20 degrees)
                    if abs(phi) < 20.:
                        tpc3_thetas_beampipe.append(theta)
                        tpc3_sumtot_bp.append(event.TPC3_sumTOT[i])
                        tpc3_tlengths_bp.append(event.TPC3_sumTOT[i]/event.TPC3_dEdx[i])
                    elif abs(phi) > 40.:
                        tpc3_thetas_notbp.append(theta)
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

                    tpc4_energies.append(event.TPC4_sumE[j])
                    tpc4_sumtot.append(event.TPC4_sumTOT[j])
                    tpc4_tlengths.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])

                    ### Select beampipe (+- 20 degrees)
                    if abs(phi) < 20.:
                        tpc4_thetas_beampipe.append(theta)
                        tpc4_sumtot_bp.append(event.TPC4_sumTOT[j])
                        tpc4_tlengths_bp.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])
                    elif abs(phi) > 40.:
                        tpc4_thetas_notbp.append(theta)
                        tpc4_sumtot_notbp.append(event.TPC4_sumTOT[j])
                        tpc4_tlengths_notbp.append(event.TPC4_sumTOT[j]/event.TPC4_dEdx[j])
                        #if 110. < theta and theta < 130. :
                        #    print('Found one in the peak:')
                        #    print(event.TPC4_sumTOT[j], 
                        #            event.TPC4_sumTOT[j]/event.TPC4_dEdx[j], 
                        #            event.TPC4_npoints[j])


    tpc3phi_array = np.array(tpc3_phis)
    tpc3theta_array = np.array(tpc3_thetas)
    tpc3_sumtot_array = np.array(tpc3_sumtot)
    tpc3_sumtot_array_bp = np.array(tpc3_sumtot_bp)
    tpc3_sumtot_array_notbp = np.array(tpc3_sumtot_notbp)
    tpc3_energies_array = np.array(tpc3_energies)
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
    tpc4_tlengths_array = np.array(tpc4_tlengths)
    tpc4_tlengths_array_bp = np.array(tpc4_tlengths_bp)
    tpc4_tlengths_array_notbp = np.array(tpc4_tlengths_notbp)
    tpc4theta_array_beampipe = np.array(tpc4_thetas_beampipe)
    tpc4theta_array_notbp = np.array(tpc4_thetas_notbp)

    beamsize = np.array(beamsize)

    plt.hist(beamsize, bins=10)
    plt.show()
    input('well?')

    #df = pd.DataFrame(
    #        {'Track Length': tpc3_tlengths_array, 'Sum TOT':tpc3_sumtot_array},
    #        index = ['Event Number'], columns = ['Track Length', 'Sum TOT'])
    #print(df)
    #input('well?')

    ### Making a ROOT histogram
    #tpc3_tlengths_sumtot = []
    #for i in range(len(tpc3phi_array)):
    #    points = [tpc3_tlengths[i],tpc4_sumtot[i]]
    #    tpc3_tlengths_sumtot.append(points)

    ##print(tpc3_tlengths_sumtot)
    ##input('well?')
    #tpc3_tlengths_sumtot_array = np.array(tpc3_tlengths_sumtot)

    #phi_bins = int(180./10.)
    phi_bins = 20
    theta_bins = 18
    #theta_bins = int(180/10.)

    #
    #h1 = Hist(phi_bins, -90., 90., markersize=0)
    #h1.fill_array(tpc3phi_array, tpc3_sumtot_array)
    #h2 = Hist(phi_bins, -90., 90.)
    #h2.fill_array(tpc3phi_array)
    ##h1.Draw()
    ##print(h1.GetEntries())
    ##input('well?')
    ##h2.Draw()
    ##print(h2.GetEntries())
    ##input('well?')
    #h1.Divide(h2)
    #h1.fillstyle = 'solid'
    #h1.fillcolor = 'blue'
    #h1.Draw()
    #print(h1.GetEntries())
    #fig = plt.figure()
    #div=rplt.hist(h1, range=[-180.180])
    #input('well?')


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=True)
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_sumtot_array)
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_sumtot_array)
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array, theta_bins, weights = tpc3_sumtot_array)
    ax2.set_title('TPC3 Neutron Recoil $\\theta$')
    ax4.hist(tpc4theta_array, theta_bins, weights = tpc4_sumtot_array)
    ax4.set_title('TPC4 Neutron Recoil $\\theta$')
    ax4.set_xlabel('Degrees')

    #ax1.hist(tpc3phi_array, phi_bins, range=[-100,100])
    #ax1.set_title('TPC3 Neutron Recoil $\phi$')
    #ax3.hist(tpc4phi_array, phi_bins, range=[-100,100])
    #ax3.set_xlabel('Degrees')
    #ax3.set_title('TPC4 Neutron Recoil $\phi$')
    #ax2.hist(tpc3theta_array, theta_bins)
    #ax2.set_title('TPC3 Neutron Recoil $\\theta$')
    #ax4.hist(tpc4theta_array, theta_bins)
    #ax4.set_title('TPC4 Neutron Recoil $\\theta$')
    #ax4.set_xlabel('Degrees')

    ## Plot theta and phi, weighted by energy
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
            sharey='col')
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_sumtot_array)
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_sumtot_array)
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array_beampipe, theta_bins, weights = 
    tpc3_sumtot_array_bp)
    ax2.set_title('TPC3 Neutron Recoil $\\theta$ (Beampipe cut)')
    ax4.hist(tpc4theta_array_beampipe, theta_bins, weights = 
            tpc4_sumtot_array_bp)
    ax4.set_title('TPC4 Neutron Recoil $\\theta$ (Beampipe cut)')
    ax4.set_xlabel('Degrees')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', 
            sharey='col')
    ax1.hist(tpc3phi_array, phi_bins, range=[-100,100], weights = 
            tpc3_sumtot_array)
    ax1.set_title('TPC3 Energy Weighted Neutron Recoil $\phi$')
    ax3.hist(tpc4phi_array, phi_bins, range=[-100,100], weights = 
            tpc4_sumtot_array)
    ax3.set_xlabel('Degrees')
    ax3.set_title('TPC4 Energy Weighted Neutron Recoil $\phi$')
    ax2.hist(tpc3theta_array_notbp, theta_bins*5, weights = 
            tpc3_sumtot_array_notbp)
    ax2.set_title('TPC3 Neutron Recoil $\\theta$ (Outside beampipe)')
    ax4.hist(tpc4theta_array_notbp, theta_bins*5, weights = 
            tpc4_sumtot_array_notbp)
    ax4.set_title('TPC4 Neutron Recoil $\\theta$ (Outside beampipe)')
    ax4.set_xlabel('Degrees')

    g, (bx1, bx2 ) = plt.subplots(1, 2, sharex=True, sharey=True)
    bx1.scatter(tpc3_tlengths_array, tpc3_sumtot_array)
    bx1.set_title('TPC3 Track Length vs Sum TOT')
    bx1.set_ylabel('SumTOT')
    bx1.set_xlabel('$\mu$m')
    bx2.scatter(tpc4_tlengths_array, tpc4_sumtot_array)
    bx2.set_title('TPC4 Track Length vs Sum TOT')
    bx2.set_xlabel('$\mu$m')
    
    h, ((cx1, cx2), (cx3, cx4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    cx1.scatter(tpc3_tlengths_array_bp, tpc3_sumtot_array_bp)
    cx1.set_title('TPC3 Track Length vs Sum TOT (beampipe)')
    cx1.set_xlim(-5000., 35000.)
    cx1.set_ylabel('SumTOT')
    cx3.scatter(tpc3_tlengths_array_notbp, tpc3_sumtot_array_notbp)
    cx3.set_title('TPC3 Track Length vs Sum TOT (not beampipe)')
    cx3.set_xlabel('$\mu$m')
    cx3.set_ylabel('SumTOT')
    cx2.scatter(tpc4_tlengths_array_bp, tpc4_sumtot_array_bp)
    cx2.set_title('TPC4 Track Length vs Sum TOT (beampipe)')
    cx2.set_xlim(-5000., 35000.)
    cx4.scatter(tpc4_tlengths_array_notbp, tpc4_sumtot_array_notbp)
    cx4.set_title('TPC4 Track Length vs Sum TOT (not beampipe)')
    cx4.set_xlabel('$\mu$m')
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

if __name__ == "__main__":
    main()

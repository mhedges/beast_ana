import os
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from root_numpy import root2rec
from os.path import expanduser

np.set_printoptions(suppress=True, precision=2)

def run_names(run_name):
    LER_Beamsize = []
    HER_Fill = []

    LER_Vacuumbump = []
    HER_Vacuumbump

    HER_Toushek = []
    LER_Toushek = []

    HER_Chromaicity = []

    HER_Injection = []
    LER_Injection = []
    
    HER_ToushekTPC = [9001]
    LER_ToushekTPC = ([10001.1, 10001.2, 10001.3, 10001.4, 10002.1, 10002.2, 
            10002.3, 10002.4, 10002.5, 10002.6, 10002.7, 10003.1, 10003.2, 
            10003.3, 10003.4, 10003.5, 10003.6, 10003.7, 10004.1, 10004.2])

def main():
    
    home = expanduser('~')
    datapath = str(home) + '/BEAST/data/v1/'

    ### Template for looping over events
    #for f in os.listdir(datapath):
    #    ifile+=str(f)
    #    print(ifile)
    #    input('well?')

    ### Choosing one ntuple file for now
    #ifile = datapath
    #ifile += 'BEAST_run5100.root'

    tpc3_phis = []
    tpc4_phis = []

    ### Looping over many files
    for f in os.listdir(datapath):
        if '.root' not in f: continue
        ifile = datapath
        ifile += f

        ### The TTree in this root file is broken (no TBranches)
        if ifile == '/Users/BEASTzilla/BEAST/data/v1/BEAST_run8001.root': 
            continue

        print(ifile)

        data = root2rec(ifile)

        counter_3 = 0
        counter_4 = 0

        for event in data:
            if ('TPC3_PID_neutrons' in event.dtype.names):
                tpc3_neutrons = event.TPC3_PID_neutrons
            else : continue

            #tpc3_neutrons = event.TPC3_PID_neutrons if ('TPC3_PID_neutrons' in 
            #        event.dtype.names) else 0
            #if tpc3_neutrons == 0: continue
            
            for i in range(len(tpc3_neutrons)):
                if tpc3_neutrons[i] == 1:
                    phi = event.TPC3_phi[i]

                    ### Check for duplicate entries
                    if len(tpc3_phis) > 0 and phi == tpc3_phis[-1] :
                        print(event.TPC3_npoints, event.TPC3_sumTOT)
                        print(event.TPC3_theta, event.HE3_rate)

                    if phi in tpc3_phis:
                        print('Possible duplicate phi in TPC3:', event.event, 
                            phi)
                        count = tpc3_phis.count(phi)
                        print('Phi has been recorded %s times'% (count))

                    if phi < -360.0 :
                        #print('Phi in TPC3 is too small:', phi)
                        continue
                    if phi > 360.0 :
                        #print('Phi in TPC3 is too large', phi)
                        continue

                    if phi < 0 : phi += 180.0
                    if phi > 180.0 : phi -= 180.0
                    tpc3_phis.append(phi)
                    #print('Neutron boolean:', neutrons)
                    #print('Neutron index:', i)
                    #print('Phi array:', event.TPC3_phi)
                    #print('Phi value:', tpc3_phis[counter])
                    #input('well?')
                    counter_3 += 1

            if ('TPC4_PID_neutrons' in event.dtype.names):
                tpc4_neutrons = event.TPC4_PID_neutrons
            else : continue

            #tpc4_neutrons = event.TPC4_PID_neutrons if ('TPC4_PID_neutrons' in 
            #        event.dtype.names) else 0
            #if tpc4_neutrons == 0: continue

            for j in range(len(tpc4_neutrons)):
                if tpc4_neutrons[j] == 1:
                    phi = event.TPC4_phi[j]

                    ### Check for duplicate entries
                    if len(tpc4_phis) > 0 and phi == tpc4_phis[-1] :
                        print(event.TPC4_npoints, event.TPC4_sumTOT)
                        print(event.TPC4_theta, event.HE3_rate)

                    if phi in tpc4_phis:
                        print('Possible duplicate phi in TPC4:', event.event, 
                            phi)
                        count = tpc4_phis.count(phi)
                        print('Phi has been recorded %s times' % (count))

                    if phi < -360.0 :
                        #print('Phi in TPC4 is too small:', phi)
                        continue
                    if phi > 360.0 :
                        #print('Phi in TPC4 is too large', phi)
                        continue

                    if phi < 0 : phi += 180.0
                    if phi > 180.0 : phi -= 180.0
                    tpc4_phis.append(phi)
                    #print('Neutron boolean:', tpc4_neutrons)
                    #print('Neutron index:', j)
                    #print('Phi array:', event.TPC4_phi)
                    #print('Phi value:', tpc4_phis[counter_4])
                    #input('well?')
                    counter_4 += 1
    tpc3phi_array = np.array(tpc3_phis)
    tpc4phi_array = np.array(tpc4_phis)
    print(tpc3phi_array.min(), tpc4phi_array.min())
    print(tpc3phi_array.max(), tpc4phi_array.max())
    #print(tpc3_phis.min(), tpc4_phis.min())

    #hist3, bins = np.histogram(tpc3phi_array, bins=10)
    #center3 = (bins[:-1] + bins[1:]) / 2
    #hist4, bins = np.histogram(tpc4phi_array, bins=45)
    #width = 0.7 * (bins[1] - bins[0])
    #plt.bar(center3, hist3, align='center', width=width)

    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax4 = fig.add_subplot(111)
    bins = 25
    ax3.hist(tpc3phi_array, bins)
    ax4.hist(tpc4phi_array, bins)
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

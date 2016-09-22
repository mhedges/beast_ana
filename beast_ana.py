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
    ifile = datapath

    ### Template for looping over events
    #for f in os.listdir(datapath):
    #    ifile+=str(f)
    #    print(ifile)
    #    input('well?')

    ### Choosing one ntuple file for now
    ifile += 'BEAST_run5100.root'
    print(ifile)

    data = root2rec(ifile)
    tpc3_phis = []
    tpc4_phis = []

    counter_3 = 0
    counter_4 = 0

    for event in data:
        tpc3_neutrons = event.TPC3_PID_neutrons
        for i in range(len(tpc3_neutrons)):
            if tpc3_neutrons[i] == 1:
                phi = event.TPC3_phi[i]
                if phi < 0 : phi += 180.0
                tpc3_phis.append(phi)
                #print('Neutron boolean:', neutrons)
                #print('Neutron index:', i)
                #print('Phi array:', event.TPC3_phi)
                #print('Phi value:', tpc3_phis[counter])
                #input('well?')
                counter_3 += 1

        tpc4_neutrons = event.TPC4_PID_neutrons
        for j in range(len(tpc4_neutrons)):
            if tpc4_neutrons[j] == 1:
                phi = event.TPC4_phi[j]
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
    print(tpc3phi_array, tpc4phi_array)


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

import os
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from root_numpy import root2rec
from os.path import expanduser

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
    for f in os.listdir(datapath):
        ifile+=str(f)
        print(ifile)
        input('well?')


if __name__ == "__main__":
    main()

from __future__ import print_function
import os
import numpy as np
import midi
import random
import sys
import math
import h5py
import os.path
import argparse
from tools import *

step_size=1
segLen=48
track_num=2
maxrange=60 ## [36, 95]
vecLen=maxrange*track_num
maxdelta=33 ## [0, 32]
Normal=120.0
defaultRes=16.0

def main():
    notes = None
    times = None
    seedDir = str(sys.argv[1])
    fileDir = os.listdir(str(seedDir))
    for filename in fileDir:
        fullfile = str(seedDir)+'/'+str(filename)
        try:
            pattern = midi.read_midifile(fullfile)
        except:
            continue
        data = pattern2map(pattern,maxdelta-1)
        note, time = firstState(data, segLen)
        if notes is None or times is None:
            notes = note
            times = time
        else:
            notes = np.append(notes, note, axis=0)
            times = np.append(times, time, axis=0)
    np.savez("seed.npz", notes=notes, times=times)

if __name__ == "__main__":
    main()

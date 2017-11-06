from __future__ import print_function
import os
import keras
from keras.models import Sequential, load_model, Model
import numpy as np
import midi
import random
import sys
import math
import h5py
import os.path
import argparse
from tools import *

parser = argparse.ArgumentParser(description='Music Generation with Mutiple Instruments (testing)')
parser.add_argument('test_dir', metavar='testing', type=str,
                    help='Path to the testing set.')
parser.add_argument('--steps', type=int, default=65535, required=False,
                    help='Steps for testing.')
parser.add_argument('--batch_size', type=int, default=128, required=False,
                    help='Number of sample each step.')
parser.add_argument('--model', type=str, default='./multi.h5', required=False,
                    help='Path to model')
args = parser.parse_args()

compute_precision='float32'
step_size=1
segLen=48
track_num=1
maxrange=60 ## [36, 95]
vecLen=maxrange*track_num
maxdelta=33 ## [0, 32]
steps = args.steps
batch_size = args.batch_size
model_path = args.model
hidden_delta=128
hidden_note=256
drop_rate=0.2 ## for powerful computer
Normal=120.0
defaultRes=16.0

def generator(path_name, step_size, batch_size, train_what='', valid=False):
    while True:
        randomFile = os.listdir(str(path_name))
        random.shuffle(randomFile)
        total_num = len(randomFile)
        for filename in randomFile:
            fullfile = str(path_name)+'/'+str(filename)
            try:
                pattern = midi.read_midifile(fullfile)
                if pattern.format != 1: continue
                data = pattern2map(pattern,maxdelta-1)
                seg, nextseg  = makeSegment(data, segLen, step_size, valid)
                del data ## clean-up
                note, time, n_note, n_time = seg2vec(seg, nextseg, segLen, vecLen, maxdelta)
                del seg
                del nextseg
            except:
                sys.stderr.write('something wrong...:\\')
                continue
            for i in xrange(0, len(note)-batch_size, batch_size):
                idx = range(i, i+batch_size)
                if train_what=='note':
                    yield ([note[idx],time[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx]], [n_note[idx],n_time[idx]])
            l = len(note)%batch_size
            if l > 0:
                idx = range(len(note)-l,len(note))
                if train_what=='note':
                    yield ([note[idx],time[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx]], [n_note[idx],n_time[idx]])

def main():
    aiComposer = load_model(model_path)
    aiComposer.summary()
    result = aiComposer.evaluate_generator(generator(args.test_dir, step_size, batch_size, 'all', valid=True), steps=steps)
    print(result)
if __name__ == "__main__":
    main()

from __future__ import print_function
import os
import numpy as np
import midi
import random
import sys
import math
import h5py
import os.path

step_size=1
segLen=48
track_num=2
maxrange=60 ## [36, 95]
vecLen=maxrange*track_num
maxdelta=33 ## [0, 32]
defaultRes=16.0

def Tempo2BPM(x):
    ret = x.data[2] | x.data[1]<<8 | x.data[0]<<16
    ret = float(60000000)/float(ret)
    return ret
## merge all tracks into one track

def pattern2map(pattern, maxtick):
    ResScale = float(pattern.resolution) / float(defaultRes)
    instrument = -1
    data=[(0,0,0)]#tick , key (main+accompany)
    for track in pattern: ## main melody if instrument==0 else accompany
        if instrument==1: break ## if main & accompany is set, then break.
        temp=[(0,0,0)] #tick, note, instrument
        speedRatio = 1.0
        Normal = 120.0
        accumTick = 0.
        firstTempo = True
        noteOnDetected = False
        for v in track:
            if hasattr(v, 'tick') :
                accumTick = accumTick + float(v.tick)/speedRatio
            if isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                if firstTempo: ## first tempo chage event detected
                    Normal = changeBPM ## set default bpm to detected bpm
                    firstTempo = False ## not first event anymore
                    continue ## no change on unit of ticks
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=36 and v.data[0]<=95 and v.data[1]>0:
                if not noteOnDetected: instrument+=1
                noteOnDetected = True
                note = (v.data[0]-36)+instrument*maxrange
                tick = int(round(accumTick/ResScale))
                temp.append((tick, note, instrument))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    data.sort()
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        data[i] = (tick,note,inst)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def firstState(data, maxlen):
    segment = data[:maxlen]

    notes = np.zeros((1, maxlen, vecLen), dtype=np.bool)
    times = np.zeros((1, maxlen, maxdelta), dtype=np.bool)

    for t, note in enumerate(segment):
        times[0, t, int(note[0])] = 1
        notes[0, t, int(note[1])] = 1
    return notes, times

def makeSegment(data, maxlen, step):
    sentences = []
    nextseq = []
    for i in xrange(0, len(data) - maxlen, step):
        sentences.append(data[i: i + maxlen])
        nextseq.append(data[i + maxlen])
    randIdx = np.random.permutation(len(sentences))
    return np.array(sentences)[randIdx], np.array(nextseq)[randIdx]

def seg2vec(segment, nextseg, segLen, vecLen, maxdelta):
    notes = np.zeros((len(segment), segLen, vecLen), dtype=np.bool)
    times = np.zeros((len(segment), segLen, maxdelta), dtype=np.bool)

    notes_n = np.zeros((len(segment), vecLen), dtype=np.bool)
    times_n = np.zeros((len(segment), maxdelta), dtype=np.bool)
    for i, seg in enumerate(segment):
        for t, note in enumerate(seg):
            times[i, t, int(note[0])] = 1
            notes[i, t, int(note[1])] = 1
        times_n[i, int(nextseg[i][0])] = 1
        notes_n[i, int(nextseg[i][1])] = 1
    return notes, times, notes_n, times_n



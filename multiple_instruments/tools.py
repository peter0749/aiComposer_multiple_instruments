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
vecLen=97
maxdelta=33 ## [0, 32]
defaultRes=16.0

# Sorted x
def purge(x):
    y = [x[-1]] #pick last element
    i = len(x)-2 # iterate over all set in reversed order
    while i>=0:
        while i>=0 and x[i][1] == y[-1][1]: ## overlapped, find next different note
            i -= 1
        if i>=0: ## founded
            y.append(x[i])
        i -= 1
    return list(reversed(y))
# merge some overlaped intervals into single interval

def Tempo2BPM(x):
    ret = x.data[2] | x.data[1]<<8 | x.data[0]<<16
    ret = float(60000000)/float(ret)
    return ret
## merge all tracks into one track

def pattern2map(pattern, maxtick):
    if pattern.format!=1 or len(pattern)==0: raise Exception('Wrong file format!')
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0, 0)]#tick , key (main+accompany)
    for track in pattern: ## main melody if instrument==0 else accompany
        temp=[(0.0, 0)] #tick, note, instrument
        speedRatio = 1.0
        Normal = 120.0
        accumTick = 0.
        firstTempo = True
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
            elif isinstance(v, midi.ProgramChangeEvent):
                if v.channel==9 or v.data[0] >= 24:  ## is drum or is not keyboard instrument
                    temp=[(0.0, 0)]
                    break
            elif isinstance(v, midi.NoteOnEvent) and v.data[1]>0:
                note = v.data[0]
                temp.append((accumTick, note))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    if len(data)<2: raise Exception('No events!')
    data.sort()
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = (0 if i==0 else data[i+1][1] - data[i][1]) + 48
        while note<0: note+=12
        while note>=vecLen: note-=12
        data[i] = (tick,note)
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



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
track_num=1
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
    data=[(0,0)]#tick , key (main+accompany)
    for track in pattern: ## main melody if instrument==0 else accompany
        temp=[(0,0)] #tick, note
        speedRatio = 1.0
        Normal = 120.0
        accumTick = 0.
        firstTempo = True
        noteOnDetected = False
        for v in track:
            if isinstance(v, midi.ProgramChangeEvent):
                if (hasattr(v, 'channel') and v.channel==9) or v.data[0]>=8: ## must be piano
                    temp=[(0,0)] #tick, note
                    break
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
                note = v.data[0]-36
                tick = int(round(accumTick/ResScale))
                temp.append((tick, note))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    data.sort()
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(maxtick if tick>maxtick else tick) ## set a threshold
        note = int(data[i+1][1])
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

def enhanced(data, maxlen): ## offset -2, +2
    res = [[(-1,-1)]*maxlen + data]
    for offset in [-2,2,4,-4]: ## -2, +2
        temp = data
        for i, (d, n) in enumerate(temp):
            if n<maxrange: ## main
                temp[i] = (d, np.clip(n+offset,0,maxrange-1))
            else:
                temp[i] = (d, np.clip(n+offset,maxrange,vecLen-1))
            if i>0 and temp[i][0]==0:
                if np.random.rand()<.1:
                    temp[i-1], temp[i] = temp[i], temp[i-1] ## swap
        res.append([(-1,-1)]*maxlen+temp)
    return res

def makeSegment(data, maxlen, step, valid=False):
    if valid:
        data = [data]
    else:
        data = enhanced(data, maxlen) ## shift tune on training set
    mutation_rate = .05/maxlen ## E: every 20 segment has 1 mutated time step
    sentences = []
    nextseq = []
    for subdata in data:
        for i in xrange(0, len(subdata) - maxlen, step):
            X = subdata[i: i + maxlen] ## input
            if not valid: ## add noise
                for t, v in enumerate(X):
                    tick, pitch = v
                    tick_new = tick
                    if tick==-1 or pitch==-1: continue
                    if np.random.rand()<mutation_rate:
                        pitch = int(np.clip(0,vecLen-1,pitch+np.random.randn()*2.))
                    if np.random.rand()<mutation_rate:
                        tick_new = int(np.clip(0,maxdelta-1,tick+np.random.randn()*2.))
                    X[t] = (tick_new, pitch)
            Y_tick, Y_pitch = subdata[i + maxlen] ## label
            Y = (Y_tick+(tick-tick_new), Y_pitch) ## fix time shifting
            sentences.append(X)
            nextseq.append(Y)
    randIdx = np.random.permutation(len(sentences))
    return np.array(sentences)[randIdx], np.array(nextseq)[randIdx]
    #return np.array(sentences), np.array(nextseq)

def seg2vec(segment, nextseg, segLen, vecLen, maxdelta):
    notes = np.zeros((len(segment), segLen, vecLen), dtype=np.bool)
    times = np.zeros((len(segment), segLen, maxdelta), dtype=np.bool)

    notes_n = np.zeros((len(segment), vecLen), dtype=np.bool)
    times_n = np.zeros((len(segment), maxdelta), dtype=np.bool)
    for i, seg in enumerate(segment):
        for t, note in enumerate(seg):
            if note[0]==-1 or note[1]==-1: continue
            times[i, t, int(note[0])] = 1
            notes[i, t, int(note[1])] = 1
        times_n[i, int(nextseg[i][0])] = 1
        notes_n[i, int(nextseg[i][1])] = 1
    return notes, times, notes_n, times_n



from __future__ import print_function
import keras
from keras.models import Sequential, load_model
import numpy as np
import random
import sys
import midi
import math
import h5py
import os.path

endLen=64
step_size=1
segLen=128
vecLen=88
maxdelta=128
maxinst=129
maxpower=64
Normal=120.0
defaultRes=256.0

# Sorted x
def purge(x):
    y = [x[-1]] #pick last element
    i = len(x)-2 # iterate over all set in reversed order
    while i>=0:
        while i>=0 and x[i][1] == y[-1][1] and x[i][2] == y[-1][2]: ## overlapped, find next different note
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
    if len(pattern) > 1:
        return normal_pattern2map(pattern, maxtick)
    else:
        return ch_pattern2map(pattern, maxtick)

def normal_pattern2map(pattern, maxtick): ## tick range [0,63] #64
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0,0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0,0)] #tick, note, instrument, power
        speedRatio = 1.0
        accumTick = 0.
        for v in track:
            if hasattr(v, 'tick') :
                accumTick = accumTick + float(v.tick)/speedRatio
            if isinstance(v, midi.ProgramChangeEvent):
                if hasattr(v, 'channel') and v.channel==9:
                    instrument = 128 # Percussion Key. a.k.a. drums
                else:
                    instrument = v.data[0]
            elif isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=21 and v.data[0]<=108 and v.data[1]>0:
                note = v.data[0]-21
                power = v.data[1] / 2
                power = 64 if power>64 else power
                power = 1 if power<1 else power
                power -= 1 ## [0, 63]
                temp.append((accumTick, note, instrument, power))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    data.sort()
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        power= data[i+1][3]
        data[i] = (tick,note,inst,power)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def ch_pattern2map(pattern, maxtick): ## tick range [0,63] #64
    ch2ins = dict()
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0,0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0,0)] #tick, note, instrument, power
        speedRatio = 1.0
        accumTick = 0.
        for v in track:
            if hasattr(v, 'tick') :
                accumTick = accumTick + float(v.tick)/speedRatio
            if isinstance(v, midi.ProgramChangeEvent):
                if hasattr(v, 'channel') and v.channel==9:
                    instrument = 128 # Percussion Key. a.k.a. drums
                else:
                    instrument = v.data[0]
                ch2ins[v.channel] = instrument
            elif isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=21 and v.data[0]<=108 and v.data[1]>0:
                ch = v.channel
                note = v.data[0]-21
                power = v.data[1] / 2
                power = 64 if power>64 else power
                power = 1 if power<1 else power
                power -= 1 ## [0, 63]
                temp.append((accumTick, note, ch, power))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    for i in xrange(len(data)): ## channel -> instrument
        acc = data[i][0]
        note= data[i][1]
        ch  = data[i][2]
        power=data[i][3]
        inst=ch2ins[ch]
        data[i] = (acc,note,inst,power)
    data.sort() ## for better quality
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        power= data[i+1][3]
        data[i] = (tick,note,inst,power)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def sample(preds, temperature=1.0):
    if temperature < 1e-9:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    res = np.argmax(probas)
    return res

def main():
    global segLen, vecLen
    if(len(sys.argv)<6):
        return
    model = load_model('./ending.h5')
    src_midi = str(sys.argv[1])
    out_midi = str(sys.argv[2])
    temperature_note = float(sys.argv[3])
    temperature_delta = float(sys.argv[4])
    temperature_inst = float(sys.argv[5])
    try:
        output = midi.read_midifile(src_midi)
        data = pattern2map(output,maxdelta-1)
    except:
        exit(1)
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    insts = np.zeros((1, segLen, maxinst))
    last = np.zeros(maxinst)
    inputEnd = len(data)
    inputStart = max(inputEnd - segLen, 0)
    inputIdx = reversed(range(inputStart, inputEnd))
    idx = segLen-1
    for i in inputIdx:
        deltas[0,idx,int(data[i][0])] = 1
        notes[0,idx,int(data[i][1])] = 1
        insts[0,idx,int(data[i][2])] = 1
        idx-=1
    tickAccum = 0
    for i in xrange(maxinst):
        output[i].pop() ## remove eof
        acc = 0
        for v in output[i]:
            if hasattr(v, 'tick'):
                acc += v.tick
        last[i] = acc ## get real time
        tickAccum = max(tickAccum, acc)
    pred_note, pred_time, pred_inst, pred_power = model.predict([notes, deltas, insts], batch_size=1, verbose=0)
    for i in xrange(maxinst):
        if last[i]==0: ## i-th instrument not appears in original midi file
            pred_inst[:,:,i] = 1e-7 ## unset
    for t in xrange(endLen):
        note = int(sample(pred_note[0][t], temperature_note))
        delta = int(sample(pred_time[0][t], temperature_delta))
        inst = int(sample(pred_inst[0][t], temperature_inst))
        tickAccum += delta
        if last[inst]==0: ## no such instrument in original midi file
            continue
        ch = 9 if inst==128 else 1
        inst_code = 0 if inst==128 else inst
        diff = int(tickAccum - last[inst])
        power = (pred_power[0][t]+1)*2
        power = 127 if power>127 else power
        power = 1 if power<1 else power
        while diff>127:
            output[inst].append(midi.NoteOnEvent(tick=127, data=[ int(note+21), int(power)]))
            diff-=127
        output[inst].append(midi.NoteOnEvent(tick=diff, data=[ int(note+21), int(power)]))
        last[inst] = tickAccum
    for i in xrange(maxinst):
        output[i].append( midi.EndOfTrackEvent(tick=1) )
    midi.write_midifile(out_midi, output)

if __name__ == "__main__":
    main()

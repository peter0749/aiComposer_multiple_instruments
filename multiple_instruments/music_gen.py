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

segLen=128
vecLen=88 #[0, 87]
maxdelta=128
maxpower=64
maxinst =129

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
    model = load_model('./multi.h5')
    tar_midi = str(sys.argv[1])
    noteNum = int(math.ceil(float(sys.argv[2]) / float(segLen)))
    temperature_note = float(sys.argv[3])
    temperature_delta = float(sys.argv[4])
    temperature_inst = float(sys.argv[5])
    output = midi.Pattern(resolution=256)
    track = [midi.Track() for _ in xrange(maxinst)]
    for i in xrange(maxinst):
        output.append(track[i])
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    insts = np.zeros((1, segLen, maxinst))
    last = np.zeros(maxinst)
    for _ in xrange(maxinst):
        last[_] = -1
    tickAccum = 0
    for i in xrange(noteNum):
        pred_note, pred_time, pred_inst, pred_power = model.predict([notes, deltas, insts], batch_size=1, verbose=0)
        for t in xrange(segLen):
            note = int(sample(pred_note[0][t], temperature_note))
            delta = int(sample(pred_time[0][t], temperature_delta))
            inst = int(sample(pred_inst[0][t], temperature_inst))
            tickAccum += delta
            ch = 9 if inst==128 else 1
            inst_code = 0 if inst==128 else inst
            if last[inst]==-1:
                track[inst].append(midi.ProgramChangeEvent(tick=0, data=[inst_code], channel=ch))
                last[inst]=0
            diff = int(tickAccum - last[inst])
            power = (pred_power[0][t]+1)*2
            power = 127 if power>127 else power
            power = 1 if power<1 else power
            while diff>127:
                track[inst].append(midi.NoteOnEvent(tick=127, data=[ int(note+21), int(power)]))
                diff-=127
            track[inst].append(midi.NoteOnEvent(tick=diff, data=[ int(note+21), int(power)]))
            last[inst] = tickAccum
        print('processed: ', i+1, '/', noteNum)
        notes = pred_note[0]
        deltas = pred_time[0]
        insts = pred_inst[0]
        ## next sequence
    for i in xrange(maxinst):
        track[i].append( midi.EndOfTrackEvent(tick=1) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()

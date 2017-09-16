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

segLen=48
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
    noteNum = int(sys.argv[2])
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
        note = int(sample(pred_note[0], temperature_note))
        delta = int(sample(pred_time[0], temperature_delta))
        inst = int(sample(pred_inst[0], temperature_inst))
        notes = np.roll(notes, -1, axis=1)
        deltas = np.roll(deltas, -1, axis=1)
        insts = np.roll(insts, -1, axis=1)
        notes[0, segLen-1, :]=0 ## reset last event
        notes[0, segLen-1, note]=1 ## set predicted event
        deltas[0, segLen-1, :]=0 ## reset last event
        deltas[0, segLen-1, delta]=1 ## set predicted event
        insts[0, segLen-1, :]=0
        insts[0, segLen-1, inst]=1
        tickAccum += delta
        ch = 9 if inst==128 else 1
        inst_code = 0 if inst==128 else inst
        if last[inst]==-1:
            track[inst].append(midi.ProgramChangeEvent(tick=0, data=[inst_code], channel=ch))
            last[inst]=0
        diff = int(tickAccum - last[inst])
        power = (pred_power[0]+1)*2
        power = 127 if power>127 else power
        power = 1 if power<1 else power
        while diff>127:
            track[inst].append(midi.NoteOnEvent(tick=127, data=[ int(note+21), int(power)]))
            diff-=127
        track[inst].append(midi.NoteOnEvent(tick=diff, data=[ int(note+21), int(power)]))
        last[inst] = tickAccum
        print('processed: ', i+1, '/', noteNum)
    for i in xrange(maxinst):
        track[i].append( midi.EndOfTrackEvent(tick=1) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()
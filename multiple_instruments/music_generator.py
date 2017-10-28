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
import argparse

parser = argparse.ArgumentParser(description='Music Generation with Mutiple Instruments')

parser.add_argument('output_midi_path', metavar='midi', type=str,
                    help='Path to the output midi file.')
parser.add_argument('--n', type=int, default=120, required=False,
                    help='Number of notes to generate.')
parser.add_argument('--note_temp', type=float, default=0.7, required=False,
                    help='Temperture of notes.')
parser.add_argument('--delta_temp', type=float, default=0.7, required=False,
                    help='Temperture of time.')
parser.add_argument('--inst_temp', type=float, default=0.7, required=False,
                    help='Temperture of instruments.')
parser.add_argument('--pipe_lim', type=int, default=1, required=False,
                    help='Maximum number of pipe(s) play at the same time.')
parser.add_argument('--brass_lim', type=int, default=1, required=False,
                    help='Maximum number of brass(es) play at the same time.')
parser.add_argument('--string_lim', type=int, default=2, required=False,
                    help='Maximum number of string(s) play at the same time.')
parser.add_argument('--alignment', type=int, default=0, required=False,
                    help='Tick alignment.')
parser.add_argument('--bpm', type=float, default=120.0, required=False,
                    help='Bpm (speed)')

args = parser.parse_args()
tar_midi = args.output_midi_path
noteNum  = args.n
temperature_note = args.note_temp
temperature_delta = args.delta_temp
temperature_inst = args.inst_temp

pipe_lim = args.pipe_lim
string_lim = args.string_lim
brass_lim = args.brass_lim

align = args.alignment
bpm = args.bpm
defaultBpm = 120.0
speedRatio = bpm / defaultBpm
defaultUnit = 500000
changedSpeed= int(round(500000.0/speedRatio))

segLen=100
vecLen=60 #[36, 95]
maxdelta=33 #[0, 32]
maxinst =14

invMap = dict()
for i in xrange(8): invMap[i] = i+40 ## 0~7 -> 40~47
for i in xrange(8, 11): invMap[i] = i+48 ## 8~10 -> 56~58
invMap[11] = 60
invMap[12] = 72
invMap[13] = 73

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
    model = load_model('./multi.h5')
    output = midi.Pattern(resolution=16) ## reduce dimension of ticks...
    track = [midi.Track() for _ in xrange(maxinst)]
    for i in xrange(maxinst):
        output.append(track[i])
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    insts = np.zeros((1, segLen, maxinst))
    last = np.zeros(maxinst)
    limits = np.zeros(maxinst)
    limits[:8] = string_lim ## at most 2 instruments play for each type of instruments
    limits[8:12] = brass_lim ## for brass
    limits[12:] = pipe_lim ## for pipes
    for _ in xrange(maxinst):
        last[_] = -1
    tickAccum = 0
    for i in xrange(noteNum):
        pred_note, pred_time, pred_inst = model.predict([notes, deltas, insts], batch_size=1, verbose=0)
        inst = int(sample(pred_inst[0], temperature_inst))
        zs = 1 ## how many notes play at the same time? self += 1
        for t in reversed(range(len(track[inst]))): ## this limits # of notes play at the same time
            if hasattr(track[inst][t], 'tick'):
                if isinstance(track[inst][t], midi.NoteOnEvent):
                    pred_note[0][track[inst][t].data[0]-36] = 1e-100
                if track[inst][t].tick==0:
                    zs += 1 ## others
                else:
                    break
        if zs>=limits[inst]: ## no more
            pred_time[0][0] = 1e-100
        note = int(sample(pred_note[0], temperature_note))
        delta = int(sample(pred_time[0], temperature_delta))
        notes = np.roll(notes, -1, axis=1)
        deltas = np.roll(deltas, -1, axis=1)
        insts = np.roll(insts, -1, axis=1)
        notes[0, segLen-1, :]=0 ## reset last event
        notes[0, segLen-1, note]=1 ## set predicted event
        deltas[0, segLen-1, :]=0 ## reset last event
        deltas[0, segLen-1, delta]=1 ## set predicted event
        insts[0, segLen-1, :]=0
        insts[0, segLen-1, inst]=1
        ch = 1
        inst_code = invMap[inst]
        if last[inst]==-1:
            track[inst].append(midi.SetTempoEvent(tick=0, data=[(changedSpeed>>16) &0xff, (changedSpeed>>8) &0xff, changedSpeed &0xff], channel=inst))
            track[inst].append(midi.ProgramChangeEvent(tick=0, data=[inst_code], channel=inst))
            last[inst]=0
        diff = int(tickAccum - last[inst]) ## how many ticks passed before it plays
        if align>1:
            new_reach = tickAccum+delta ## accum ticks before this event + this key plays after received signal?
            if new_reach % align != 0: ## if not aligned
                new_reach += align-(new_reach%align)
            delta = min(32, max(0, new_reach - tickAccum)) ## aligned tick

        ## note alignment:
        while diff>127:
            track[inst].append(midi.ControlChangeEvent(tick=127, channel=inst, data=[3, 0])) ## append 'foo' event (data[0]==3 -> undefine)
            diff-=127
        if diff>0:
            track[inst].append(midi.ControlChangeEvent(tick=diff, channel=inst, data=[3, 0])) ## append 'foo' event

        ## note on:
        track[inst].append(midi.NoteOnEvent(tick=delta, data=[ int(note+36), 127], channel=inst))
        tickAccum += delta
        last[inst] = tickAccum
        print('processed: ', i+1, '/', noteNum)
    for i in xrange(maxinst):
        track[i].append( midi.EndOfTrackEvent(tick=1, channel=i) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()

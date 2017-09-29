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
parser.add_argument('--finger_number', type=int, default=5, required=False,
                    help='Maximum number of notes play at the same time.')
parser.add_argument('--align_melody', type=int, default=4, required=False,
                    help='Main melody alignment.')
parser.add_argument('--align_accompany', type=int, default=8, required=False,
                    help='Accompany alignment.')

args = parser.parse_args()
tar_midi = args.output_midi_path
noteNum  = args.n
temperature_note = args.note_temp
temperature_delta = args.delta_temp
finger_limit = args.finger_number
align_right = args.align_melody
align_left  = args.align_accompany

segLen=48
track_num=2
maxrange=60 #[36, 95]
vecLen=maxrange*track_num
maxdelta=33 #[0, 32]

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
    track = [midi.Track() for _ in xrange(track_num)]
    for i in xrange(track_num):
        output.append(track[i])
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    last = np.zeros(track_num)
    for _ in xrange(track_num):
        last[_] = -1
    tickAccum = 0
    for i in xrange(noteNum):
        pred_note, pred_time = model.predict([notes, deltas], batch_size=1, verbose=0)
        for inst in xrange(track_num):
            zs = 1 ## how many notes play at the same time? self += 1
            for t in reversed(range(len(track[inst]))): ## this limits # of notes play at the same time
                if hasattr(track[inst][t], 'tick'):
                    if isinstance(track[inst][t], midi.NoteOnEvent):
                        pred_note[0][(track[inst][t].data[0]-36)+inst*maxrange] = 1e-100
                    if track[inst][t].tick==0:
                        zs += 1 ## others
                    else:
                        break
            if zs>=finger_limit: ## no more fingers
                pred_time[0][0] = 1e-100
        key = int(sample(pred_note[0], temperature_note))
        note = key  % maxrange
        inst = key // maxrange
        delta = 0 if i==0 else int(sample(pred_time[0], temperature_delta))
        align = align_right if inst==0 else align_left
        if align>1:
            delta = int(round(delta/align)*align)
        notes = np.roll(notes, -1, axis=1)
        deltas = np.roll(deltas, -1, axis=1)
        notes[0, segLen-1, :]=0 ## reset last event
        notes[0, segLen-1, key]=1 ## set predicted event
        deltas[0, segLen-1, :]=0 ## reset last event
        deltas[0, segLen-1, delta]=1 ## set predicted event
        if last[inst]==-1:
            track[inst].append(midi.ProgramChangeEvent(tick=0, data=[0], channel=0)) ## first event: program change to piano
            last[inst]=0
        diff = int(tickAccum - last[inst]) ## how many ticks passed before it plays

        ## note alignment:
        while diff>127:
            track[inst].append(midi.ControlChangeEvent(tick=127, channel=0, data=[3, 0])) ## append 'foo' event (data[0]==3 -> undefine)
            diff-=127
        if diff>0:
            track[inst].append(midi.ControlChangeEvent(tick=diff, channel=0, data=[3, 0])) ## append 'foo' event

        ## note on:
        track[inst].append(midi.NoteOnEvent(tick=delta, data=[ int(note+36), 127]))
        tickAccum += delta
        last[inst] = tickAccum
        print('processed: ', i+1, '/', noteNum)
    for i in xrange(track_num):
        track[i].append( midi.EndOfTrackEvent(tick=1) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()

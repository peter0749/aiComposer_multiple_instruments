from __future__ import print_function
import numpy as np
import random
import sys
import midi
import math
import h5py
import os.path
import argparse
from fractions import gcd

parser = argparse.ArgumentParser(description='Music Generation with Mutiple Instruments')

parser.add_argument('output_midi_path', metavar='midi', type=str,
                    help='Path to the output midi file.')
parser.add_argument('--n', type=int, default=120, required=False,
                    help='Number of notes to generate.')
parser.add_argument('--note_temp', type=float, default=0.7, required=False,
                    help='Temperture of notes.')
parser.add_argument('--delta_temp', type=float, default=0.7, required=False,
                    help='Temperture of time.')
parser.add_argument('--temp_sd', type=float, default=0.01, required=False,
                    help='Standard deviation of temperture.')
parser.add_argument('--finger_number', type=int, default=5, required=False,
                    help='Maximum number of notes play at the same time.')
parser.add_argument('--align', type=int, default=4, required=False,
                    help='Main melody alignment.')
parser.add_argument('--bpm', type=float, default=120.0, required=False,
                    help='Bpm (speed)')
parser.add_argument('--do_format', action='store_true', default=False,
                    help='Format data before sending into model...')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Fix random seed')
parser.add_argument('--sticky', action='store_true', default=False,
                    help='')
parser.add_argument('--pitch_shift', type=int, default=48, required=False,
                    help='')

args = parser.parse_args()
tar_midi = args.output_midi_path
noteNum  = args.n
temperature_note = args.note_temp
temperature_delta = args.delta_temp
temperature_sd = args.temp_sd
finger_limit = args.finger_number
align = args.align
do_format = args.do_format

if args.debug:
    np.random.seed(7) ## for debugging

bpm = args.bpm
defaultBpm = 120.0
speedRatio = bpm / defaultBpm
defaultUnit = 500000
changedSpeed= int(round(500000.0/speedRatio))

segLen=48
vecLen=49 #[-24, +24]
maxdelta=33 #[0, 32]

import keras
from keras.models import Sequential, load_model

def sample(preds, temperature=1.0, temperature_sd=0.05):
    temperature += np.random.randn()*temperature_sd ## add some noise
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
    seed = np.load('./seed.npz')
    seedIdx = np.random.randint(len(seed['notes']))
    output = midi.Pattern(resolution=16) ## reduce dimension of ticks...
    track = midi.Track()
    output.append(track)
    track.append(midi.SetTempoEvent(tick=0, data=[(changedSpeed>>16) &0xff, (changedSpeed>>8) &0xff, changedSpeed &0xff]))
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    notes[:,:,:] = seed['notes'][seedIdx,:,:]
    deltas[:,:,:] = seed['times'][seedIdx,:,:]
    seed = None ## release
    note = args.pitch_shift
    assert note>=0 and note<=127
    tickAccum=0
    for i in xrange(noteNum):
        pred_note, pred_time = model.predict([notes, deltas], batch_size=1, verbose=0)
        zs = 1 ## how many notes play at the same time? self += 1
        for t in reversed(range(len(track))): ## this limits # of notes play at the same time
            if hasattr(track[t], 'tick'):
                if track[t].tick==0:
                    zs += 1 ## others
                else:
                    break
        if zs>=finger_limit: ## no more fingers
            pred_time[0][0] = 1e-100
        if note-24<36:
            skip = min(24-note+36, 24)
            pred_note[0][:skip] = 1e-100
        elif note+24>95: ## 127-note<24
            skip = max(95-note-24+vecLen, 25)
            pred_note[0][skip:] = 1e-100
        key =  int(sample(pred_note[0], temperature_note, temperature_sd)) ## [-24~+24]
        note += (key-24)
        while note<36: note+=12
        while note>95: note-=12
        delta = int(sample(pred_time[0], temperature_delta, temperature_sd))
        if align>1:
            new_reach = tickAccum+delta ## accum ticks before this event + this key plays after received signal?
            if new_reach % align != 0: ## if not aligned
                new_reach += align-(new_reach%align)
            delta = min(32, max(0, new_reach - tickAccum)) ## aligned tick
        ## note on:
        if args.sticky:
            for t in reversed(range(len(track))):
                if isinstance(track[t], midi.NoteOffEvent):
                    break
                elif isinstance(track[t], midi.NoteOnEvent):
                    findLastNoteOn = track[t].data[0]
                    track.append(midi.NoteOffEvent(tick=0, data=[ findLastNoteOn, 0]))
                    break
        track.append(midi.NoteOnEvent(tick=delta, data=[ int(min(127,max(0,note))), 127]))
        tickAccum += delta
        notes = np.roll(notes, -1, axis=1)
        deltas = np.roll(deltas, -1, axis=1)
        notes[0, segLen-1, :]=0 ## reset last event
        notes[0, segLen-1, key]=1 ## set predicted event
        deltas[0, segLen-1, :]=0 ## reset last event
        deltas[0, segLen-1, delta]=1 ## set predicted event
        if do_format:
            for t in reversed(range(1, segLen)):
                rd = np.where(deltas[0, t]==1)[0][0] ## right delta
                rn = np.where(notes[0, t]==1)[0][0] ## right note
                ld = np.where(deltas[0, t-1]==1)[0][0] ## left ..
                ln = np.where(notes[0, t-1]==1)[0][0]
                if rd!=0: break
                if ln>rn: ## swap
                    notes[0, t-1:t+1, :] = 0 ## t-1, t
                    notes[0, t, ln] = 1
                    notes[0, t-1, rn] = 1
                else: break
        print('processed: ', i+1, '/', noteNum)
    if args.sticky and len(track)>0 and isinstance(track[-1], midi.NoteOnEvent):
        track.append(midi.NoteOffEvent(tick=0, data=[ track[-1].data[0], 0]))
    track.append( midi.EndOfTrackEvent(tick=0) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()

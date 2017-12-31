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
parser.add_argument('--note_temp', type=float, default=0, required=False,
                    help='Temperture of notes.')
parser.add_argument('--delta_temp', type=float, default=0, required=False,
                    help='Temperture of time.')
parser.add_argument('--vol_temp', type=float, default=0, required=False,
                    help='Temperture of velocity.')
parser.add_argument('--temp_sd', type=float, default=0, required=False,
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
parser.add_argument('--init', type=str, default='seed', required=False,
        help='Initialization: seed/random/zero (default: seed)')
parser.add_argument('--sticky', action='store_true', default=False,
                    help='')
parser.add_argument('--main_instrument', type=int, default=0, required=False,
                    help='Main instrument')
parser.add_argument('--models', nargs='+', type=str, help='Path list to models', required=True)
parser.add_argument('--pitch_style_weights', nargs='*', type=float, help='Weights for style of pitch', required=False)
parser.add_argument('--tick_style_weights', nargs='*', type=float, help='Weights for style of tick', required=False)

args = parser.parse_args()
tar_midi = args.output_midi_path
noteNum  = args.n
temperature_note = args.note_temp
temperature_delta = args.delta_temp
temperature_vol = args.vol_temp
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
track_num=1
maxrange=60 #[36, 95]
vecLen=maxrange*track_num
maxdelta=33 #[0, 32]
maxvol=32

import keras
from keras.models import Sequential, load_model

models = [load_model(str(model_path)) for model_path in args.models]
note_weights = args.pitch_style_weights
tick_weights = args.tick_style_weights

note_weights.extend([1e-6]*(len(models)-len(note_weights)))
tick_weights.extend([1e-6]*(len(models)-len(tick_weights)))

def mix_multi(notes, ticks, model_list, note_weights, tick_weights):
    p_note = np.zeros((1, notes.shape[-2], notes.shape[-1]),dtype=np.bool)
    p_tick = np.zeros((1, ticks.shape[-2], ticks.shape[-1]),dtype=np.bool)
    for i, model in enumerate(model_list):
        p_n, p_t = model.predict([notes, ticks])
        p_note += p_n*note_weights[i]
        p_tick += p_t*tick_weights[i]
    p_note /= np.sum(note_weights)
    p_tick /= np.sum(tick_weights)
    return p_note, p_tick

def sample(preds, temperature=1.0, temperature_sd=0.05):
    if preds.ndim!=1: raise ValueError('Only support 1-D array!')
    temperature += np.random.randn()*temperature_sd ## add some noise
    if temperature < 1e-9:
        return np.argmax(preds)
    old = preds
    try:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) ## e_x -> x
        preds -= max(0.0, np.max(preds)) ## do max trick
        exp_preds = np.exp(preds/temperature)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), 1, p=preds)[0]
    except:
        return np.argmax(old)

def main():
    global segLen, vecLen
    instProgram = [ args.main_instrument ]
    vol_model = load_model('./velocity.h5')
    output = midi.Pattern(resolution=16) ## reduce dimension of ticks...
    track = [midi.Track() for _ in xrange(track_num)]
    for i in xrange(track_num):
        output.append(track[i])
    notes = np.zeros((1, segLen, vecLen))
    deltas = np.zeros((1, segLen, maxdelta))
    powers = np.zeros((1, segLen, maxvol), dtype=np.bool)
    if args.init=='seed':
        seed = np.load('./seed.npz')
        seedIdx = np.random.randint(len(seed['notes']))
        notes[:,:,:] = seed['notes'][seedIdx,:,:]
        deltas[:,:,:] = seed['times'][seedIdx,:,:]
        print('Using seed: %s' % seed['names'][seedIdx])
        seed = None ## release
    elif args.init=='random': ## random init
        notes[:,:,:] = np.eye(vecLen)[np.random.choice(vecLen, segLen)]
        deltas[:,:,:]= np.eye(maxdelta)[np.random.choice(maxdelta, segLen)]
    last = np.zeros(track_num)
    for _ in xrange(track_num):
        last[_] = -1
    tickAccum = 0
    for i in xrange(noteNum):
        pred_note, pred_time = mix_multi(notes, deltas, models, note_weights, tick_weights)
        volume = vol_model.predict([notes, deltas, powers], batch_size=1, verbose=0)
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
        key = int(sample(pred_note[0], temperature_note, temperature_sd))
        note = key
        inst = 0
        delta = int(sample(pred_time[0], temperature_delta, temperature_sd))
        if last[inst]==-1:
            track[inst].append(midi.SetTempoEvent(tick=0, data=[(changedSpeed>>16) &0xff, (changedSpeed>>8) &0xff, changedSpeed &0xff], channel=inst))
            track[inst].append(midi.ProgramChangeEvent(tick=0, data=[ instProgram[inst] ], channel=inst)) ## first event: program change to piano
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
        if args.sticky:
            for t in reversed(range(len(track[inst]))):
                if isinstance(track[inst][t], midi.NoteOffEvent):
                    break
                elif isinstance(track[inst][t], midi.NoteOnEvent):
                    findLastNoteOn = track[inst][t].data[0]
                    track[inst].append(midi.NoteOffEvent(tick=0, data=[ findLastNoteOn, 0], channel=inst))
                    break
        real_lab = int(sample(volume[0], temperature_vol, temperature_sd))
        real_vol = int(np.clip(real_lab*8, 1, 255))
        track[inst].append(midi.NoteOnEvent(tick=delta, data=[ int(note+36), real_vol], channel=inst))
        tickAccum += delta
        last[inst] = tickAccum
        notes = np.roll(notes, -1, axis=1)
        deltas = np.roll(deltas, -1, axis=1)
        powers = np.roll(powers, -1, axis=1)
        notes[0, -1, :]=0 ## reset last event
        notes[0, -1, key]=1 ## set predicted event
        deltas[0, -1, :]=0 ## reset last event
        deltas[0, -1, delta]=1 ## set predicted event
        powers[0,-1, :] = 0
        powers[0,-1, real_lab] = 1
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
    for i in xrange(track_num):
        if args.sticky and len(track[i])>0 and isinstance(track[i][-1], midi.NoteOnEvent):
            track[i].append(midi.NoteOffEvent(tick=0, data=[ track[i][-1].data[0], 0], channel=i))
        track[i].append( midi.EndOfTrackEvent(tick=0, channel=i) )
    midi.write_midifile(tar_midi, output)

if __name__ == "__main__":
    main()

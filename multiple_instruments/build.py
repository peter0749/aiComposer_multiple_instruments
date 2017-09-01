import os.path
import math
import random
import midi
import numpy as np
import h5py
import os
import sys
import collections

step_size=1
segLen=128
vecLen=88 #[0, 87]
maxdelta=128
maxpower=64
maxinst=129 #[0, 128], where 128 is drum on channel 10
defaultRes=256.0
Normal=120.0

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

def makeSegment(data, maxlen, step):
    sentences = []
    nextseq = []
    for i in xrange(0, len(data) - maxlen, step):
        sentences.append(data[i: i + maxlen])
        nextseq.append(data[i + maxlen])
    print('nb sequences:', len(sentences))
    return sentences, nextseq

def seg2vec(segment, nextseg, segLen, vecLen, maxdelta, maxinst):
    notes = np.zeros((len(segment), segLen, vecLen), dtype=np.bool)
    times = np.zeros((len(segment), segLen, maxdelta), dtype=np.bool)
    powers= np.zeros((len(segment), segLen, 1), dtype=np.uint8)
    insts = np.zeros((len(segment), segLen, maxinst), dtype=np.bool)

    notes_n = np.zeros((len(segment), vecLen), dtype=np.bool)
    times_n = np.zeros((len(segment), maxdelta), dtype=np.bool)
    powers_n= np.zeros((len(segment), 1), dtype=np.uint8)
    insts_n = np.zeros((len(segment), maxinst), dtype=np.bool)
    for i, seg in enumerate(segment):
        for t, note in enumerate(seg):
            times[i, t, int(note[0])] = 1
            notes[i, t, int(note[1])] = 1
            insts[i, t, int(note[2])] = 1
            powers[i, t,0] = int(note[3])
        times_n[i, int(nextseg[i][0])] = 1
        notes_n[i, int(nextseg[i][1])] = 1
        insts_n[i, int(nextseg[i][2])] = 1
        powers_n[i,0] = int(nextseg[i][3])
    return notes, times, insts, powers, notes_n, times_n, insts_n, powers_n

def main():
    if(len(sys.argv)<3):
        return
    h5Name = str(sys.argv[2])
    if ( not os.path.isfile(h5Name) ):
        h5file = h5py.File(h5Name,'w')
        h5file.create_dataset('Note', (1, segLen, vecLen), maxshape=(None, segLen, vecLen), dtype=np.bool )
        h5file.create_dataset('Delta', (1, segLen, maxdelta), maxshape=(None, segLen, maxdelta), dtype=np.bool )
        h5file.create_dataset('Inst', (1, segLen, maxinst), maxshape=(None, segLen, maxinst), dtype=np.bool )
        h5file.create_dataset('NextNote', (1, vecLen), maxshape=(None, vecLen), dtype=np.bool )
        h5file.create_dataset('NextDelta', (1, maxdelta), maxshape=(None, maxdelta), dtype=np.bool )
        h5file.create_dataset('NextPower', (1, 1), maxshape=(None, 1), dtype=np.uint8 )
        h5file.create_dataset('NextInst', (1, maxinst), maxshape=(None, maxinst), dtype=np.bool )
        h5file.close()
        x_rec=0
        y_rec=0
    else:
        h5file = h5py.File(h5Name, 'r')
        x_rec = np.shape(h5file['Note'])[0]
        y_rec = np.shape(h5file['NextNote'])[0]
        h5file.close()
    processed=0
    randomFile = os.listdir(str(sys.argv[1]))
    random.shuffle(randomFile)
    total_num = len(randomFile)
    for filename in randomFile:
        fullfile = str(sys.argv[1])+'/'+str(filename)
        try:
            pattern = midi.read_midifile(fullfile)
        except:
            continue
        data = pattern2map(pattern,maxdelta-1)
        seg, nextseg  = makeSegment(data, segLen, step_size)
        data = None ## clean-up
        note, time, inst, power, n_note, n_time, n_inst, n_power = seg2vec(seg, nextseg, segLen, vecLen, maxdelta, maxinst)
        seg = None
        nextseg = None
        unit = np.shape(note)[0]
        h5file = h5py.File(h5Name,'a')
        data0 = h5file['Note']
        data1 = h5file['Delta']
        data2  = h5file['Inst']
        lab0 = h5file['NextNote']
        lab1 = h5file['NextDelta']
        lab2 = h5file['NextInst']
        lab3 = h5file['NextPower']
        x_new_size = x_rec+unit
        data0.resize(x_new_size, axis=0)
        data1.resize(x_new_size, axis=0)
        data2.resize(x_new_size, axis=0)
        y_new_size = y_rec+unit
        lab0.resize(y_new_size, axis=0)
        lab1.resize(y_new_size, axis=0)
        lab2.resize(y_new_size, axis=0)
        lab3.resize(y_new_size, axis=0)
        data0[x_rec:x_new_size] = note
        data1[x_rec:x_new_size] = time
        data2[x_rec:x_new_size] = inst
        lab0[y_rec:y_new_size] = n_note
        lab1[y_rec:y_new_size] = n_time
        lab2[y_rec:y_new_size] = n_inst
        lab3[y_rec:y_new_size] = n_power
        h5file.close()
        x_rec=x_new_size
        y_rec=y_new_size

        processed += 1
        print "processed: ", processed, "/", total_num

if __name__ == "__main__":
    main()

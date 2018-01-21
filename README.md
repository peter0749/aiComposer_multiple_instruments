# aiComposer

Automatic music generation

### How to use

First, clone the branch: `dynamic`:

```
 git clone -b dynamic --single-branch https://github.com/peter0749/aiComposer_multiple_instruments
```

Enter the repository:

```
cd aiComposer_multiple_instruments/multiple_instruments
```

Download the model for dynamics and place it under the same directory of `music_generator.py`.

[model for dynamics](./)

Rename it to `velocity.h5`.

Download the model and place it under the same directory of `music_generator.py`.

[model with mozart style](./)

Rename it to `multi.h5`.

Finally, generate some music with mozart style!

```
python music_generator.py mz.mid --n 800 --note_temp 0.6 --delta_temp 0.6 --vol_temp 0.4 --align 2 --finger_number 5 --init zero --bpm 100 --debug
```

The generated midi file is named `mz.mid`. 

Under the same directory of `music_generator.py`

The `mz.mid` file should be identical to the file `example/mz.mid`.

If you want to generate different music, just remove the `--debug` argument.

You can also try to generate different style of music with different model.

Which is the `multi.h5` file.

All trained models are in the `release` tag.

### Model List



### Arguments

```

music_generator.py [-h] [--n N] [--note_temp NOTE_TEMP]
                          [--delta_temp DELTA_TEMP] [--temp_sd TEMP_SD]
                          [--finger_number FINGER_NUMBER] [--align ALIGN]
                          [--bpm BPM] [--do_format] [--debug] [--init INIT]
                          [--sticky] [--main_instrument MAIN_INSTRUMENT]
                          midi

Music Generation

positional arguments:
  midi                  Path to the output midi file.

optional arguments:
  -h, --help            show this help message and exit
  --n N                 Number of notes to generate.
  --note_temp NOTE_TEMP
                        Temperture of notes.
  --delta_temp DELTA_TEMP
                        Temperture of time.
  --temp_sd TEMP_SD     Standard deviation of temperture.
  --finger_number FINGER_NUMBER
                        Maximum number of notes play at the same time.
  --align ALIGN         Main melody alignment.
  --bpm BPM             Bpm (speed)
  --do_format           Format data before sending into model...
  --debug               Fix random seed
  --init INIT           Initialization: seed/random/zero (default: zero)

```

### Training loss plot

Only show validation loss plots and accuracy plots of `nott.h5` and `mz.h5`

Plots of `nott.h5`:

![nott loss](./multiple_instruments/plots/nott_loss.png)

![nott acc](./multiple_instruments/plots/nott_acc.png)

Plots of `mz.h5`:

![mz loss](./multiple_instruments/plots/mz_loss.png)

![mz acc](./multiple_instruments/plots/mz_acc.png)


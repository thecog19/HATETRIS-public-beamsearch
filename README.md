# HATETRIS-public-beamsearch
A publicly available implementation of knewjade's NNUE based search heuristic for [HATETRIS](https://qntm.org/hatetris). 

Based on knewjades description [here](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9)

This is the code used to obtain the current record of [302 points]()

## Setup

This should work in any environment that has rust set up, but we have only tried it on linux based systems, so we can't promise that. 

We assume familiarity with a command line system. 

First clone the repository 

```bash
git clone git@github.com:thecog19/HATETRIS-public-beamsearch.git
```

Then build and run with this command: 
```bash
cargo run --release
```

This should install all dependencies and run the program. We output logs to stdout so you should pipe them somewhere if you want to review them. Something like 
```bash
mkdir Output\ Logs/
cargo run --release 2>&1 | tee Output\ Logs/NAME-OF-OUTPUT-LOG.txt
```
should suffice

## The Network
By default we start with a fully untrained network. On our machine we saw a drastic improvement after one generation and took only about three weeks to reach 302 points. Check your logs to see how the network is performing. By default the data lives in the `training` folder. 

If the program crashes, you can just re-run it and as long as the training folder is populated, things will resume normally. You may want to check for a dangling file in these scenarios. 

In `Training/Aeon X/Generation Y/Replay/`, files will be of the type `parent_xxx.bin` and `move_xxx.bin`.  If either one is interrupted mid-write for a given timestep, you will have to delete both the move file and the parent file for that timestep in order for the beam search to resume properly.  The previous timesteps are not affected and will not need to be deleted.

If training, then in Training/Aeon X/Generation Y/Training/, files will be of the type `epoch_xxx.bin`.  These can also fail mid-write; if so, you will need to delete the file from the most recent timestep.  The other epochs will not be affected.

## Training

Training functions by taking a given well, and running a mini-beam search on it, exploring the potential of the well. It generates a few child wells, which are rated based on their performance as the beamsearch rolls out. Wells that end in defeat are rated poorly, ones that do not are rated better. We explore twice the training beam depth, if the game ends in the 0-beam depth move range, we rate it -1, if it ends in the beam-depth-beam-depth*2 range, we scale it between the best result and -1 to account for how close it got to being the "best" well of that generation. We do this for wells gathered from the last generation of games. We'll link to the longer description here, when we write it. For now, read knewjades description and go look at the code!   

## Metaparameters

Metaparameters live in the constants file. They are "self-documenting" in that we didn't write a lot of documentation for them, but here are some you may want to tweak. 

- WELL_HEIGHT, WIDTH: Dimensions of the well itself.
- HIDDEN: Number of hidden nodes in the network.  More hidden nodes means a potentially better network, but will increase evaluation time.
- CONVOLUTIONS: A list of the (x,y) rectangles used for convolutions.  If you want a different shape, or want to add more shapes, use these.
- MINIBATCH: How many positions are used for each epoch of the neural network training.
- MAX_EPOCHS: How many epochs the neural network is trained for.
- MASTER_BEAM_WIDTH: The size of the master beam search for a full generation.  This beam search is done once per generation.
- TRAINING_BEAM_WIDTH: The size of the small beams used to evaluate each position within an epoch.
- TRAINING_BEAM_DEPTH: The number of timesteps to run the small beams for.
- THREAD_NUMBER: Number of threads available.  Change based on your CPU and computational availability.
- AEON: Bookkeeping; used to separate multiple training runs from each other.

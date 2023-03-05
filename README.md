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

If the program crashes, you can just re-run it and as long as the traing folder is populated, things will resume normally. You may want to check for a dangling file in these scenarios. In the training folder 

## Metaparameters

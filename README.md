# HATETRIS-public-beamsearch
A publicly available implementation of knewjade's NNUE based search heuristic for [HATETRIS](https://qntm.org/hatetris). 

Based on knewjades description [here](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9)

This is the code used to obtain the current record of [302 points]()

## Setup

This should work in any environment that has rust set up, but we have only tried it on linux based systems, so we can't promise that. 

We assume familiarity with a command line system. 

First clone the repository 

```
git clone git@github.com:thecog19/HATETRIS-public-beamsearch.git
```

Then build and run with this command: 
```
cargo run --release
```

This should install all dependencies and run the program. That's it. 

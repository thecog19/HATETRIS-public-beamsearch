# Table of Contents
- [Introduction](#introduction)
	- [What is HATETRIS?](#what-is-hatetris)
	- [What is This Repository?](#what-is-this-repository)
		- [Emulator](#emulator)
		- [Beam Search](#beam-search--quiescence-lookahead)
		- [Neural Network](#neural-network)
		- [Loop Prevention](#loop-prevention)
- [Installation & Setup](#installation--setup)
- [Training](#training)
	- [Cycle](#cycle)
	- [Network](#network)
	- [Files](#files)
	- [Anatomy of a Log File](#anatomy-of-a-log-file)
- [Utilities](#utilities)
	- [Useful Functions & Types](#useful-functions--types)
	- [Useful Constants](#useful-constants)
- [Further Reading](#further-background-reading)

## Introduction

### What is HATETRIS?

[HATETRIS](https://qntm.org/hatetris) is a variation of the classic game [Tetris](https://en.wikipedia.org/wiki/Tetris), written as a JavaScript browser game in 2010 by sci-fi author and programmer [qntm](https://qntm.org/).  The piece selection, instead of being random, is based on the state of the game and minmaxed to make clearing lines vastly more difficult than Tetris; however, this difficulty is balanced by the game's deterministic nature, which allows the game tree to be searched to an arbitrary depth in an emulator.

The game can be played [here](https://qntm.org/files/hatetris/hatetris.html).

### What is This Repository?

This repository contains Rust code for a highly-optimized HATETRIS emulator and a [**beam search**](https://en.wikipedia.org/wiki/Beam_search) routine using a neural network as the pruning heuristic.  This repository was used to obtain the current world record of [366 points](https://qntm.org/hatetris#komment66d89ea487f7b).

A detailed explanation of the history of HATETRIS records, and of the development of this repository and the techniques in it, can be found on our blog:

1. [Getting The World Record In HATETRIS](https://hallofdreams.org/posts/hatetris) (2022)
2. [Losing The World Record In HATETRIS](https://hallofdreams.org/posts/hatetris-2) (2023)

#### Emulator

The emulator represents a standard 20x10 HATETRIS board as an array of 16 `u16` integers, with the four rows above the well line not stored (since they will always be 0).  To define some basic terminology:

- A **well** is the 20 x 10 area in which the game is played.  This does not include the piece moving around within the well.
- A **state** consists of the well and the current score.
- A **piece** is the [tetromino](https://en.wikipedia.org/wiki/Tetromino) being maneuvered around in the well. When the piece cannot move down any further, it merges with the well and stops moving. This generates a new well.
- A **position** refers to a legal placement of a piece within the well, be it terminal or not.
- A **terminal position** is a piece that has reached a position where it can no longer descend, that is, it is going to become part of the well, exactly where it is. Once this piece is placed, we have created a new well.
- A **move** is a placement of a piece within a well, such that it is terminal, and generates a new well. We do not consider non-terminal motion inside the well to be a ‘move’.

The emulator and the game graph created by the emulator work on **moves**, and not **positions**; for instance, a "depth 2" child of the starting well on the game graph does not describe a tetromino which has been moved twice; instead, it describes a well in which two tetrominoes have been placed into a terminal position.

The piece positions within a well are not directly emulated (except during replay code generation); instead, a 64-bit ['waveform'](https://hallofdreams.org/posts/hatetris-2/#cacheless-waveforms) is used to represent all 34 possible positions a piece at a given height.  Bitshifting is used to rotate the waveform, and to move it left or right:

	w_old    = abcd efgh ijkl mnop qrst
	w_left   = efgh ijkl mnop qrst 0000
	w_right  = 0000 abcd efgh ijkl mnop
	w_rotate = bcda fghe jkli nopm rstq

Bitmasks are created via `lazy_static` evaluation to calculate terminal positions, scores, and well heights of a given waveform, *without* the computationally expensive process of rendering the actual well.  No well is rendered unless it is a legal move.

Previous versions of this code cached common waveforms and well patterns, to avoid bit operations at the cost of memory and hashmap lookups; the hashmap lookups turned out to be slower than recalculating the bitshifts for each new move.  The current version of the emulator averages ~5 μs per move per core to generate all children of a typical well.

<img src="https://hallofdreams.org/assets/img/HATETRIS/EmulatorSpeedComparison.png" alt="A comparison of speeds of various versions of the HATETRIS emulator." title="A comparison the average move speed of various versions of the HATETRIS emulator."/>

#### Beam Search & Quiescence Lookahead

The beam search optionally uses [**quiescence lookahead**](https://www.chessprogramming.org/Quiescence_Search) to judge each well in a beam search not by its current heuristic, but by the maximum heuristic of any well reachable by a continuous series of lineclears.  While some wells which cannot force the piece needed to clear the line are overestimated this way, parametric studies across a variety of heuristics and beam widths show the quiescence lookahead to be a considerable improvement.

The concept of a beam search for HATETRIS was first implemented by [threepipes](https://github.com/threepipes/hatetris) in 2019, and has been used in every HATETRIS record achieved by anyone since 2021.  The quiescence lookahead was first implemented for HATETRIS by [us](https://hallofdreams.org/posts/hatetris-2/), in 2022.

#### Neural Network

The neural network is **[NNUE](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)** (stylized **ƎUИИ**), an efficiently updatable neural network design first implemented by Yu Nasu in 2018 for use in [computer shogi](https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf) ([English](https://github.com/asdfjkl/nnue/blob/main/nnue_en.pdf)) and later used in the [Stockfish chess engine](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#table-of-contents), among others.  It is a fast neural network which is sparse enough to use the CPU rather than GPU; this repository does not yet make use of optimizations such as the accumulator, quantization, or [Single Instruction Multiple Data](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data), but performance can likely be significantly improved.

NNUE was first implemented for HATETRIS by [knewjade](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9), in 2022.

#### Loop Prevention

While HATETRIS has always been deterministic, before [June of 2021](https://qntm.org/loops) it was possible (albeit not yet done) to force the game into an infinite loop and clear lines indefinitely.  To prevent this, qntm added a loop prevention rule to the piece selection algorithm, which now ranks pieces by [an ordered preference](https://github.com/qntm/hatetris/blob/main/src/enemy-ais/hatetris-ai.ts):

1) If all pieces lead into previously seen wells, choose the piece corresponding to the well which has been seen the fewest times.
2) If any piece does not lead into a previously seen well, choose that piece.
3) Otherwise, choose whichever piece minmaxes the well height (i.e. the piece which, when placed to minimize well height, causes the highest well height). 
4) Otherwise, break ties in the order S -> Z -> O -> I -> L -> J -> T. 

This repository currently accounts for rules 2-4; for performance reasons, we do not currently account for the case of all pieces leading to previously seen wells.  We have not yet found any case of even two pieces out of seven leading to a repeat well and suspect that rule 1) cannot be triggered under normal HATETRIS rules, so implementing the rule is not currently a priority, but we will be glad to implement it if you use this repository and encounter a repeat well, (and we would also approve a PR that adds this functionality).

## Installation & Setup

The program should work in any environment which has the [Rust programming language](https://www.rust-lang.org/tools/install) installed, but we have only run this on Linux, and the exact commands may differ on Mac or Windows.

First, go to the desired directory and clone the repository:

```bash
cd /path/to/desired/directory
git clone git@github.com:thecog19/HATETRIS-public-beamsearch.git
```

Then, build and run with the command:

```bash
cargo build --release
```

This should install all dependencies.

The code is run from the directory which contains `src` and `Cargo.toml`, with the command:

```bash
cargo run --release
```

We output all logs to `stdout`, so you should pipe `stdout` somewhere if you want to review the logs later:

```bash
mkdir Output\ Logs/
cargo run --release 2>&1 | tee -a Output\ Logs/NAME-OF-OUTPUT-LOG.txt
```

## Training

By default, training is performed with the `training_cycle()` function:

```rust
use neural::training_cycle;

fn main() {
	training_cycle();
}
```

You can run a beam search without active training; for instance, to run a beam with a width of 1,000 based on the Generation 3 network, without saving the beam or using it for training:

```rust
let mut conf = SearchConf::testing();
conf.quiescent = true;
conf.generation = 3;
conf.beam_width = 1_000_000;

let neural_network_path = conf.neural_network_path(); 
let weight = load_file(&neural_network_path, NET_VERSION).unwrap();
let starting_state = State::new();

println!("");
println!("Starting beam with width {}", conf.beam_width);

beam_search_network_full(&starting_state, &weight, &conf);
```

### Cycle

HATETRIS training consists of a series of discrete 'generations', each one of which trains and uses a new neural network:

0. The network is created, based on either the previous generation or (for Generation 0) with random weights.
1. The network is used for a beam search of width `MASTER_BEAM_WIDTH`, using the quiescent lookahead.
2. Wells are taken at random from the beam search results.  Each well is used to run a small beam of width `TRAINING_BEAM_WIDTH` for at most `TRAINING_BEAM_DEPTH` steps **without** the quiescent lookahead; the best heuristic in that beam is used as the training target for that well.  If the beam does not last `TRAINING_BEAM_DEPTH` steps, the well's training target is -1.0.
3. A new network is trained by taking the previous network and backpropagating the wells and training targets gathered in step 2.  This new network is used for the next generation of training.

Our results from training with nearly-default parameters showed rapid improvement.  Random initialization (and the fact that we modified this code and its default parameters several times during training) will ensure that no two runs are identical; still, we'd expect these general trends to hold when training beams of width ~10^6:

| Generation | Score w/ Quiescence | Score w/o Quiescence |
|------------|---------------------|----------------------|
|          0 |                   8 |                    4 |
|          1 |                  89 |                   26 |
|          2 |                 160 |                   41 |
|          3 |                 186 |                   69 |
|          4 |                 204 |                   87 |
|          5 |                 302 |                   91 |
|          6 |                 366 |                  123 |

We typically start to see loops at scores past ~300; we did not encounter any for generations 0 through 5.

### Network

The network has two fully-connected layers with the `tanh()` activation function within.

The input layer is sparse and convolutional.  The two convolutions used here are `4x3` and `1x10`:
- `4x3` is a rectangle containing four rows and three columns.  Each `4x3` window in a well can be any one of 2^(4*3) = 4096 values.
- `1x10` is a single row of has 10 squares, and can be any one of 2^10 - 1 = 1023 values (not 1024, since the row corresponding to a full clear can never be reached in a real well).
- This repository uses positional convolutions, meaning that e.g. an empty row on line 10 and an empty row on line 11 will contribute different amounts to the hidden layer.

We take this network from [the work of knewjade](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9#1-neural-network%E3%81%AE%E6%A7%8B%E6%88%90); the two differences between our network and his are that our network has positional convolutions and lacks a bias neuron.

<img src="https://hallofdreams.org/assets/img/HATETRIS/NNUE_Visualization.png" width=500, alt="A visualization of the network architecture." title="A visualization of the network architecture."/>

*Source: [knewjade](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9#1-neural-network%E3%81%AE%E6%A7%8B%E6%88%90)*

The hidden layer is dense, and converts `HIDDEN` neurons to a single output representing the 'quality' of the well.  The output is scaled from -1.0 to +1.0; to make some sorting functions easier, we represent this as an `isize` in the beam search, ranging from -1,000,000 to +1,000,000.

### Files

`training_cycle()` will check the `AEON` constant in `constants.rs` to determine which folder to start in.  For instance, if `AEON` is 0 (the default), then `training_cycle()` will check for (or create) a set of folders under `Training / Aeon 0` in the base directory:

```
HATETRIS-PUBLIC-BEAMSEARCH
├── Output Logs
├── src
├── target
│   └── release
│       ├── ...
└── Training
    └── Aeon 0
        ├── Generation 0
        ├── Generation 1
        ├── Generation 2
        ├── ...
```

The folder for `Generation N` will contain:
- An NNUE network file called `Network N.bin`.
- A `Replay` folder, containing for each move:
	- `move_i.bin`: The wells in the beam at depth `i`.  The beam will have a maximum width of `MASTER_BEAM_WIDTH`.
	- `parent_i.bin`: The parent graph of all wells in the beam at depth `i`.
	- `loops_i.bin`: .  If no loops were found during move `i`, this file will not be created.
- A `Training` folder, containing:
	- `all_epochs.bin`, containing `MAX_EPOCHS * TRAINING_BEAM_WIDTH` wells taken at random from the master beam search stored in `Replay`.
	- `epoch_i.bin`, where `i` ranges from 0 to `MAX_EPOCHS`.  Each epoch contains the starting well and the final result of `MINIBATCH` small beam searches from `MINIBATCH` different wells, each of which has maximum width `TRAINING_BEAM_WIDTH`.

```
Generation N
├── Network N.bin
├── Replay
│   ├── move_0.bin
│   ├── move_10.bin
│   ├── move_100.bin
│   ├── move_101.bin
│   ├── move_102.bin
│   ├── ...
│   ├── parent_10.bin
│   ├── parent_100.bin
│   ├── parent_101.bin
│   ├── parent_102.bin
│   ├── ...
│   ├── loops_47.bin [OPTIONAL]
└── Training
    ├── all_epochs.bin
    ├── epoch_0.bin
    ├── ...
    ├── epoch_255.bin
```

The training cycle can be restarted from any point within training.  However, if the program is stopped in the middle of a file-write, the resulting partially-written file will not be valid and the program will panic.  To clean this up, you will need to delete the relevant `parent_i.bin` and `move_i.bin` of the latest timestep (previous timesteps are unaffected and will not need to be deleted), or the relevant `epoch_i.bin` (again, the epoch will not be affected).

### Anatomy of a Log File

The output piped to `stdout` uses a consistent pattern from one run to another:

**Startup:**

```
Creating new training cycle.
No previous generation files found, starting from generation 0
```

**Within a Beam Search:**


Wells are always wrapped in a `State` (containing a well and a score), a `StateH` (containing a well, a score, and a heuristic estimate from the neural network), or `StateP` (containing a well, score, heuristic estimate, and index of a parent).  Each row is represented as a decimal number; for instance, `775` would represent a row `1100000111`, meaning that the leftmost two and rightmost three squares in that row are occupied, while the middle five squares are empty.

```
...
Depth 364
Time: 48604 seconds
Total children: 33344910
New well count: 1048576
Maximum score: StateH { heuristic: 806128, score: 142, well: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 515, 999, 975, 1007, 446] }
Worst heuristic: StateH { heuristic: 633172, score: 139, well: [0, 768, 896, 768, 257, 769, 771, 769, 257, 771, 783, 831, 513, 959, 999, 510] }
Best heuristic: StateH { heuristic: 915909, score: 137, well: [0, 0, 1, 63, 127, 575, 543, 783, 799, 775, 771, 775, 911, 1007, 1007, 446] }
Family distribution: [573463, 475113, 0, 0, 0, 0, 0, 0, 0, 0]
Score distribution: [(135, 344), (136, 40237), (137, 246369), (138, 362463), (139, 323610), (140, 62781), (141, 12674), (142, 98)]
Loops: 2
...
```

Most lines in the output are self-explanatory; `Family distribution` refers to the lowest rows of each well.  The lowest row has never been cleared (and might be impossible to clear); this makes it a decent proxity for well and strategy diversity in the early game.  For sufficiently large beams and sufficiently good networks, the lowest row will eventually become `0111111110` (in decimal, `510`), so this stops being useful in the late game.

The terms in this per-move output are controlled by the `print_progress()` function in `searches.rs`. You can modify your output names here if needed. 

**After a Beam Search**

After a beam search, the `keyframes` of the longest possible game (note: this is usually, but not necessarily, the game with the highest score) are output in reverse chronological order:

```
StateH { well: [1022, 1011, 507, 895, 991, 767, 1007, 991, 383, 959, 502, 442, 639, 1021, 637, 510], score: 302, heuristic: -999801 }
StateH { well: [962, 1011, 507, 895, 991, 767, 1007, 991, 383, 959, 502, 442, 639, 1021, 637, 510], score: 302, heuristic: -999771 }
StateH { well: [2, 1011, 507, 895, 991, 767, 1007, 991, 383, 959, 502, 442, 639, 1021, 637, 510], score: 302, heuristic: -999727 }
StateH { well: [0, 1008, 506, 895, 991, 767, 1007, 991, 383, 959, 502, 442, 639, 1021, 637, 510], score: 302, heuristic: -999741 }
...
StateH { well: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], score: 0, heuristic: 931545 }
```

Afterwards, the game's replay is output.  Each keystroke is represented by `L`, `R`, `D`, or `U` (for left, right, down, and rotate moves, respectively).  This is an extremely inefficient encoding, conveying only two bits of usable information per character, but any other encoding will not be guaranteed to split properly when wanting to e.g. separately describe the stem and repeating section of a loop.

HATETRIS has used hexadecimal (4 bits per character), [Base65536](https://github.com/qntm/base65536) (16 bits per character) and currently [Base2048](https://github.com/qntm/base2048) (11 bits per character, but Twitter-compatible) to represent replays.  Currently, the easist way to convert a run from this representation to Base2048 (for an equally valid, but far more compact, replay string) is to paste the replay into the [game itself](https://qntm.org/files/hatetris/hatetris.html), allow the game to finish replaying, and copy the resulting Base2048 replay from the end screen.  We may eventually port Base2048 to Rust to avoid this step. (Or if you're looking to make a contribution to this repository, we would approve a PR...)

```RRRDDDDDDDDDDDDDDDDDDRRDDDDDDDDDDDDDDDDDLLLDDD...```

Next, the training routine infers the number of well states present in each `move_i.bin` snapshot file and selects a random number of well states to extract from each snapshot, so that the total number of moves is `MAX_EPOCHS * MINIBATCH`.  This is unfortunately bottlenecked by file read times; it make take a couple hours to load and extract random wells from all of the files, even though we only want a miniscule amount of data from each one individually. This is an area for potential optimization, but not something we have prioritized. 

```
770163758 states identified from 789 timesteps in 0 seconds.
1048576 selections allocated in 0 seconds.
...
19 states extracted from timestep 782 by 7802 seconds.
23 states extracted from timestep 783 by 7803 seconds.
16 states extracted from timestep 784 by 7803 seconds.
0 states extracted from timestep 785 by 7803 seconds.
4 states extracted from timestep 786 by 7803 seconds.
0 states extracted from timestep 787 by 7803 seconds.
0 states extracted from timestep 788 by 7803 seconds.
Training path found, training data populated up to -1 out of 256.
```

Each epoch is then assigned `MINIBATCH` wells and processed in an independent thread.  Depending on the size of `MINIBATCH`, `TRAINING_BEAM_WIDTH`, and `TRAINING_BEAM_DEPTH`, each epoch could take 30 minutes to an hour to run.

```
...
Training data generated for epoch 237 out of 256 in 2903 seconds.
Training data generated for epoch 238 out of 256 in 2930 seconds.
Training data generated for epoch 239 out of 256 in 2927 seconds.
Training data generated for epoch 240 out of 256 in 2870 seconds.
...
```

The neural network is backpropagated on the wells and training targets on one epoch at a time.  If three consecutive epochs occur where the loss goes up rather than down, training is halted early to prevent overfitting.

```
Baseline α = 0.0005, mul = 1
Epoch 1: Loss: 0.9634717347545582 -> 0.48708445806935374
Saving epoch 1
Epoch 2: Loss: 0.5124996676330884 -> 0.22140142596518755
Saving epoch 2
Epoch 3: Loss: 0.24569042568008911 -> 0.12269242728472703
Saving epoch 3
Epoch 4: Loss: 0.13440167007141396 -> 0.08684355795403566
Saving epoch 4
...
```

After training is finished, the network for the next generation is finished, and the training cycle begins again.

## Utilities

### Useful Functions & Types

These functions are the ones most likely to be worth modifying or editing, if you want to change how the beam search or training routine works.

- `emulator.rs`: 
	- `neural_heuristic_loop()`: Calculates the legal children and their associated heuristics from a given state; the loop-prevention rule is used.
	- `quiescent_heuristic()`: Calculates the heuristics of a set of states based on the maximum heuristic values of their descendants reachable by continuous S (or Z) line clears.
- `neural.rs`:
	- `forward_pass()`: Computes a linear layer and `tanh()` activation layer for a set of neurons.  Currently the bottleneck for beam searches; it might be optimizable with Single Instruction Multiple Data (SIMD) or an accumulator.
	- `generate_training_data()`: Splits the total list of states into epochs; useful to modify if you want something other than a raw beam search result to determine the training target for each state.
- `searches.rs`:
	- `beam_search_network_full()`: Performs a beam search with all the bells and whistles; multithreading, loop prevention, and optional quiescence lookahead.  This is the function to use if you want to run a standalone beam search.
	- `complete_search()`: Starts from a given state and does a complete search of the game tree from that state, with no heuristic of any kind.  Useful for near-endgame wells or small well dimensions.  Does **not** currently implement the loop-prevention rule; results for a well width of 4 are not accurate as a result.
	- `print_progress()`: Prints some aggregate statistics about an ongoing beam search every move.
- `types.rs`
	- `SearchConf`: the configuration passed to `beam_search_network_full()` and `neural_heuristic_loop()`, as well as the other heuristic and search functions, to pass in global parameters.  This is where the beam width, depth, print behavior, and quiescent behavior are stored.
	- `WeightT`: describes the creation and initialization of neural network weights.  You need to change this if you want to change the neural architecture used.

### Useful Constants

- **Well Geometry:** Dimensions of the well itself.
	- `WELL_HEIGHT`
	- `WELL_WIDTH`
- **Neural Network:** Dimensions of the neural network.
	- `CONVOLUTIONS`: A list of the (x,y) rectangles used for convolutions.  If you want different shapes, or want to add more shapes, use this array.
	- `ALL_CONV`: The total count of all convolutions in the network.
	- `HIDDEN`: Number of hidden nodes in the network.  More hidden nodes means a potentially better network, but will increase evaluation time.
	- `WEIGHT_COUNT`: The total number of `f64` weights in the neural network.
- **Training:** Parameters which influence the training cycle.
	- `MINIBATCH`: How many wells are used for each epoch of the neural network training.
	- `MAX_EPOCHS`: How many epochs the neural network is trained for.
	- `MASTER_BEAM_WIDTH`: The size of the master beam search for a full generation.  This beam search is done once per generation.
	- `TRAINING_BEAM_WIDTH`: The size of the small beams used to evaluate each position within an epoch.
	- `TRAINING_BEAM_DEPTH`: The number of timesteps to run the small beams for.
	- `AEON`: Bookkeeping; used to separate multiple training runs from each other.  Useful when modifying hyperparameters or training routines.
- **Multithreading:**
	- `THREAD_NUMBER`: Number of threads available.  Change based on your CPU and computational availability.  By default, this **only** affects non-[`rayon`](https://docs.rs/rayon/latest/rayon/) processes; `rayon` processes will use all available cores.  There are two ways of restricting `rayon` threads:
		- Set the environmental variable `RAYON_NUM_THREADS` to the desired thread count.
		- Add `rayon::ThreadPoolBuilder::new().num_threads(THREAD_NUMBER).build_global().unwrap();` to the top of `fn main()` in `main.rs`.
	- `THREAD_BATCH`: Default batch size of threads.  Again, this **only** affects non-`rayon` threads.

## Further Background Reading

- threepipes
	- [Create an AI to conquer HATETRIS](https://www.slideshare.net/slideshow/hatetrisai-190245892/190245892) [JAPANESE]
		- First known HATETRIS beam search.
	- [hatetris](https://github.com/threepipes/hatetris) (GitHub Repository)
		- No emulator; calls the original JavaScript functions directly.
		- Uses a manual heuristic use [Optuna](https://optuna.org/) to optimize the heuristic's hyperparameters.
		- Highest score: 18.
- knewjade
	- [What I did to get 66L in HATETRIS](https://gist.github.com/knewjade/586c9d82bd53f13afa8bcb7a65f8bd5a) [JAPANESE]
		- Covers the world records between 32 and 66 points.
		- 32 points: an exhaustive search on the ending of the current 31 point world record.
		- 34, 41 points: a transposition table to store favorable 'terrains' and a manual heuristic approximating the 'quality' of each well.  Beam width of 1 million.
		- 45 points: an improved heuristic and a wider beam (12 million).
		- 66 points: further heuristic improvements, and a wider beam (25 million).
	- [Reached 289 lines in HATETRIS using Neural Networks](https://gist.github.com/knewjade/24fd3a655e5321c8ebac8b93fa497ed9) [JAPANESE]
		- First use of NNUE for HATETRIS, as well as using the quiescence lookahead.
		- Introduction of the master beam search / training beam search technique for (respectively) training samples and training targets.
		- Beam width: 200,000.
		- First discovered loop.
- Dave & Felipe (Us!)
	- [Getting the world record in HATETRIS](https://hallofdreams.org/posts/hatetris/)
		- A detailed breakdown of the techniques that ultimately culminated in the 86 point world record .
		- Details about various thought processes that led to the current approach.
		- Covers AlphaGo approaches, MCTS etc. 
		- A good overview of well structure and heuristic approaches. 
	- [Impractical Rust: The HATETRIS World Record](https://youtu.be/UgQUvD9gyMk)
		- Technical detail on the MCTS approaches, Rust-specific difficulties, etc.
		- Overview of hyperparameter optimization (and why it's not easy).
		- Eerie tales of mutex errors and ghost nodes.
	- [Losing the world record in HATETRIS](https://hallofdreams.org/posts/hatetris-2/)
		- Introduces the 'bird in hand' heuristic, later refined into the quiescent lookahead.
		- A detailed overview of various emulator improvements (most still in use).
		- Covers the discovery of loops. 
		- Concludes with a detailed run-down of NNUE and how it is used. 
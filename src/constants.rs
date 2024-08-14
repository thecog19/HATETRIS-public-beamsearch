use crate::types::{RowT, WaveT};

// WELL GEOMETRY

// The RowT and WaveT types from types.rs MUST match the well width w:
//		RowT: Needs w + 1 bits.
//		WaveT: Needs 4*(w + 2*3 - 4) = 4*w + 8 bits. 
// For instance, a well of standard width 10 would need:
//		RowT: 10 + 1 bits -> u16
//		WaveT: 4*10 + 4 = 44 -> u64.

pub const WELL_HEIGHT: usize = 20;
pub const WELL_LINE: usize = 4;

pub const WIDTH: usize = 10;
pub const WAVE_SIZE: usize = 4*WIDTH + 4;

pub const MAX_ROW: RowT = (2 as RowT).pow(WIDTH as u32)-1;
pub const EFF_HEIGHT: usize = WELL_HEIGHT - WELL_LINE;

// ROTATE_LEFT:   0b000100010001...
// ROTATE_RIGHT:  0b111011101110...

// WAVE_OFFSET used to be (16^(w+1) - 1)/15, but the w+1 limited well width too much.
// Now it is 16 * (16^w - 1)/15 + 1

pub const WAVE_OFFSET: WaveT = 1 + 16 * ((16 as WaveT).pow((WIDTH) as u32) - 1) / 15;
pub const ROTATE_LEFT: WaveT = WAVE_OFFSET * 1;
pub const ROTATE_RIGHT: WaveT = WAVE_OFFSET * 14;


// NEURAL NET PARAMETERS

pub const HIDDEN: usize = 48;
pub const CONVOLUTIONS: [(usize, usize); 2] = [(1,10), (4,3)]; // The order of arguments matters!

// I don't like hardcoding the count, but I don't see a way around it.
pub const CONV_POW: [usize; 2] = [
	2_usize.pow((CONVOLUTIONS[0].0 * CONVOLUTIONS[0].1) as u32),
	2_usize.pow((CONVOLUTIONS[1].0 * CONVOLUTIONS[1].1) as u32)
	];
pub const CONV_COUNT: usize = 
	(EFF_HEIGHT - CONVOLUTIONS[0].0 + 1) * (WIDTH - CONVOLUTIONS[0].1 + 1) + 
	(EFF_HEIGHT - CONVOLUTIONS[1].0 + 1) * (WIDTH - CONVOLUTIONS[1].1 + 1);
pub const ALL_CONV: usize = 
	CONV_POW[0] * (EFF_HEIGHT - CONVOLUTIONS[0].0 + 1) * (WIDTH - CONVOLUTIONS[0].1 + 1) + 
	CONV_POW[1] * (EFF_HEIGHT - CONVOLUTIONS[1].0 + 1) * (WIDTH - CONVOLUTIONS[1].1 + 1);
pub const WEIGHT_COUNT: usize = HIDDEN * ALL_CONV;

pub const ALPHA: f64 = 0.0001;
pub const EPS: f64 = 0.00000001;
pub const RHO: f64 = 0.999;
pub const RHO_F: f64 = 0.9;

pub const CHUNK: usize = 4;

// TRAINING LOOP PARAMETERS

pub const MINIBATCH: usize = 4096;
pub const MAX_EPOCHS: isize = 256;

pub const MASTER_BEAM_WIDTH: usize = 1_000_000;
pub const TRAINING_BEAM_WIDTH: usize = 512;
pub const TRAINING_BEAM_DEPTH: usize = 10;

pub const AEON: usize = 0;

// FILE NAMING AND VERSIONING

pub const BEAM_WIDTH: usize = 10_000;

pub const CHECKPOINTS: &str = "Training";

pub const RUN_TYPE: &str = "recursive_heuristic_2/"; // Only for non-neural-network runs.
pub const SAVE_RUN: bool = false;

pub const VERSION: u32 = (WELL_HEIGHT << 16 + WIDTH << 8 + 0) as u32; // Implicitly limits wells to 65536 x 256.
pub const NET_VERSION: u32 = 1;

// COMPUTATIONAL PARAMETERS

pub const THREAD_NUMBER: usize = 4;
pub const THREAD_BATCH: usize = 1024;
pub const MULTIPLIER: f64 = 1_000_000.0;

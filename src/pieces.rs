// Piece list and base waveform generation

use lazy_static::lazy_static;

use crate::constants::{WAVE_SIZE, WIDTH};
use crate::types::{RowT};

pub const PIECE_COUNT: usize = 7;

// All pieces must have their bounding boxes aligned with the right well edge.
pub const BASE_PIECES: [[[RowT; 4]; 4]; PIECE_COUNT] = [
	[[0,3,6,0], [0,4,6,2], [0,6,12,0], [4,6,2,0]],  // S
	[[0,6,3,0], [0,2,6,4], [0,12,6,0], [2,6,4,0]],  // Z
 	[[0,6,6,0], [0,6,6,0], [0,6,6,0], [0,6,6,0]],   // O
	[[0,15,0,0], [2,2,2,2],[0,0,15,0], [4,4,4,4]],  // I
	[[0,7,4,0], [0,6,2,2], [0,2,14,0], [4,4,6,0]],  // L
	[[0,6,4,4], [0,14,2,0], [2,2,6,0], [0,4,7,0]],  // J
	[[0,7,2,0], [0,2,6,2], [0,4,14,0], [4,6,4,0]],  // T
];

lazy_static! {
	pub static ref PIECE_LIST: [[[RowT; 4]; WAVE_SIZE]; PIECE_COUNT] = {
		let mut piece_list = [[[0; 4]; WAVE_SIZE]; PIECE_COUNT];
		for p in 0..PIECE_COUNT {
			for x in 0..WIDTH+1 {
				for rot in 0..4 {
					for row in 0..4 {
						if x < WIDTH - 2 {
							piece_list[p][rot + 4*x][row] = BASE_PIECES[p][rot][row] << (WIDTH - 2 - x);
						} else {
							piece_list[p][rot + 4*x][row] = BASE_PIECES[p][rot][row] >> (x - (WIDTH - 2));
						}
					}
				}
			}
		};
		piece_list
	};
}
use lazy_static::lazy_static;

use crate::constants::{MAX_ROW, WAVE_SIZE};
use crate::pieces::{PIECE_COUNT, PIECE_LIST};
use crate::types::WaveT;

// Masks
// All masks have as many significant bits as waveforms do.
// All masks are 0, for invalid positions within a waveform and 1 for valid positions.

lazy_static!{
	pub static ref EMPTY_MASKS: [WaveT; PIECE_COUNT] = {
		let mut empty_mask: [WaveT; PIECE_COUNT] = [0; PIECE_COUNT];
		for p in 0..PIECE_COUNT {
			let mut right_edge = [false; 4];
			let mut mask: WaveT = 0;

			for x in 0..WAVE_SIZE {
				let mut exceeds_max = false;
				for row in 0..4 {
					if PIECE_LIST[p][x][row] > MAX_ROW {
						exceeds_max = true;
						break;
					}
				}
				mask <<= 1;
				if !exceeds_max && !right_edge[x % 4] {
					mask += 1;
				}

				for row in 0..4 {
					if PIECE_LIST[p][x][row] % 2 == 1 {
						right_edge[x % 4] = true;
						break;
					}
				}
			}
			empty_mask[p] = mask;
		};
		empty_mask
	};
}

// The individual row masks will sometimes be 1 for pieces past the left edge of the well.
// This is fine; we never use the individual masks without also using the empty well mask.

lazy_static!{
	pub static ref ROW_MASKS: [[[WaveT; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT] = {
		let mut row_masks: [[[WaveT; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT] 
		= [[[0; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT];
	
		for p in 0..PIECE_COUNT {
			for conf in 0..=MAX_ROW {
				for row in 0..4 {
					let mut mask = 0;
		
					for w in 0..WAVE_SIZE {
						mask <<= 1;
						if PIECE_LIST[p][w][row] & conf == 0 {
							mask += 1;
						}
					}
					row_masks[p][conf as usize][row] = mask;
				}
			}
		};
		row_masks
	};
}

lazy_static!{
	pub static ref HEIGHT_MASKS: [[WaveT; 4]; PIECE_COUNT] = {
		let mut height_masks: [[WaveT; 4]; PIECE_COUNT] = [[0; 4]; PIECE_COUNT];
		for p in 0..PIECE_COUNT {
			let mut mask = 0;
			for row in 0..4 {
				let mut tmp_mask = 0;

				for w in 0..WAVE_SIZE {
					tmp_mask <<= 1;
					if PIECE_LIST[p][w][row] != 0 {
						tmp_mask += 1;
					}
				}
				mask |= tmp_mask;

				height_masks[p][row] = mask;
			}
		};
		height_masks
	};
}

lazy_static!{
	pub static ref SCORE_MASKS: [[[WaveT; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT] = {
		let mut score_masks: [[[WaveT; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT] 
			= [[[0; 4]; (MAX_ROW + 1) as usize]; PIECE_COUNT];
	
		for p in 0..PIECE_COUNT {
			for conf in 0..=MAX_ROW {
				for row in 0..4 {
					let mut mask = 0;
		
					for w in 0..WAVE_SIZE {
						mask <<= 1;
						if PIECE_LIST[p][w][row] + conf == MAX_ROW && conf != MAX_ROW {
							mask += 1;
						}
					}
					score_masks[p][conf as usize][row] = mask;
				}
			}
		};
		score_masks
	};
}

lazy_static! {
	pub static ref SURFACE_LINE_ARRAY: Vec<Vec<u16>> = {
		let mut surface_line_array = vec![vec![0; (MAX_ROW + 1) as usize]; (MAX_ROW + 1) as usize];
		for i in 0..=MAX_ROW {
			for j in 0..=MAX_ROW {
				// We want a new surface consisting of 0s and 1s.
				// A bit within a new surface will be 1 if both the corresponding bit on the previous surface and the new line are 1.
				// A bit within a new surface will also be 1 if the corresponding bit on the new line is 1 and a bit to either side is 1.
				// Otherwise, the bit will be 0.

				let previous_surface = i;
				let current_line = !j;
				let mut new_surface = previous_surface & current_line;
				new_surface &= MAX_ROW;

				let mut old_surface = 0;
				while old_surface != new_surface {
					old_surface = new_surface;
					new_surface = ((new_surface << 1) | (new_surface >> 1) | new_surface) & current_line;
					new_surface &= MAX_ROW;
				}

				surface_line_array[i as usize][j as usize] = new_surface & MAX_ROW;
			}
		}
		surface_line_array
	};
}

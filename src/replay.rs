// We use the extremely fast waveform / bitmask emulator during training,
// but that emulator can't simulate arrow keys or track individual moves,
// so we need a weaker emulator to generate replays.

use crate::{
	constants::{ROTATE_LEFT, ROTATE_RIGHT, WAVE_SIZE, WELL_HEIGHT, WIDTH}, emulator::{waveform_to_wells, well_slice}, masks::{EMPTY_MASKS, ROW_MASKS}, pieces::PIECE_COUNT, types::{State, WaveT, WellT}
};

use fnv::FnvHashMap;
use std::hash::{Hash, Hasher};

#[repr(u8)]
#[derive(Clone, Copy)]
enum Path {
	LEFT = 0,
	RIGHT = 1,
	DOWN = 2,
	ROTATE = 3
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Move {
	piece: usize,
	wave: WaveT,
	height: usize
}

impl Hash for Move {
	fn hash<H: Hasher>(&self, state: &mut H) {
		(self.wave as u64 |
		(self.piece as u64) << WAVE_SIZE | 
		(self.height as u64) << (WAVE_SIZE + 3)).hash(state);
    }
}

impl Move {
	fn full_mask(&self, well: &WellT) -> WaveT {
		let well_slice = well_slice(self.height, well);
		let mut mask = EMPTY_MASKS[self.piece];
		for (r, row) in well_slice.iter().enumerate() {
			mask &= ROW_MASKS[self.piece][*row as usize][r];
		}
		return mask
	}

	fn rotate(&self, full_mask: &WaveT) -> Option<Move> {
		let new_wave = 
			((self.wave & ROTATE_LEFT) << 3) | 
			((self.wave & ROTATE_RIGHT) >> 1);

		if new_wave & full_mask == 0 {
			return None
		} else {
			return Some(Move{piece: self.piece, wave: new_wave, height: self.height})
		}
	}

	fn right(&self, full_mask: &WaveT) -> Option<Move> {
		let new_wave = self.wave >> 4;
			
		if new_wave & full_mask == 0 {
			return None
		} else {
			return Some(Move{piece: self.piece, wave: new_wave, height: self.height})
		}
	}

	fn left(&self, full_mask: &WaveT) -> Option<Move> {
		let new_wave = self.wave << 4;
			
		if new_wave & full_mask == 0 {
			return None
		} else {
			return Some(Move{piece: self.piece, wave: new_wave, height: self.height})
		}
	}

	fn down(&self, well: &WellT) -> Option<Move> {
		if self.height == WELL_HEIGHT + 2 {return None}

		let new_move = Move{piece: self.piece, wave: self.wave, height: self.height + 1};
			
		let mask = new_move.full_mask(well);
		if new_move.wave & mask == 0 {
			return None
		} else {
			return Some(new_move)
		}
	}
}

fn get_states(state: &State, piece: usize) -> FnvHashMap<State, Vec<Path>> {
	// TODO: Double-check the starting position for narrower wells.
	let start = Move{
		piece: piece, 
		wave: 1 << 4 * (WIDTH / 2 + 1) - 1, 
		height: 0};

	let mut paths: FnvHashMap<Move, Vec<Path>> = FnvHashMap::default();
	paths.insert(start, vec![]);

	let mut to_return = FnvHashMap::default();

	let mut to_check = vec![start];
	while to_check.len() > 0 {
		let mut new_to_check = vec![];
		for t in &to_check {
			let mask = t.full_mask(&state.well);
			let old_path = paths.get(&t).unwrap().clone();

			// The order these are checked in determine how ties are broken when two paths reach the same move in the same number of steps.
			// Different ordering results in different-looking replays.
			// Our personal preference has been Rotate, Left, Right, Down.
			match t.rotate(&mask) {
				Some(m) => {
					if !paths.contains_key(&m) {
						let mut new_path = old_path.clone();
						new_path.push(Path::ROTATE);
						paths.insert(m, new_path);
						new_to_check.push(m.clone());
					}
				},
				None => ()
			}
			match t.left(&mask) {
				Some(m) => {
					if !paths.contains_key(&m) {
						let mut new_path = old_path.clone();
						new_path.push(Path::LEFT);
						paths.insert(m, new_path);
						new_to_check.push(m.clone());
					}
				},
				None => ()
			}
			match t.right(&mask) {
				Some(m) => {
					if !paths.contains_key(&m) {
						let mut new_path = old_path.clone();
						new_path.push(Path::RIGHT);
						paths.insert(m, new_path);
						new_to_check.push(m.clone());
					}
				},
				None => ()
			}
			match t.down(&state.well) {
				Some(m) => {
					if !paths.contains_key(&m) {
						let mut new_path = old_path.clone();
						new_path.push(Path::DOWN);
						paths.insert(m, new_path);
						new_to_check.push(m.clone());
					}
				},
				None => {
					let new_state = &waveform_to_wells(t.wave, t.height, t.piece, &state)[0];
					if !to_return.contains_key(new_state) {
						let mut new_path = old_path.clone();
						new_path.push(Path::DOWN);
						to_return.insert(new_state.clone(), new_path);
					}
				}
			}
		}

		to_check = new_to_check;
	}

	return to_return
}

// Prints moves as LRDU.
fn vec_to_2bit(path: &Vec<Path>) -> String {
	path
		.iter()
		.map(|p| match p {
			Path::LEFT => "L",
			Path::RIGHT => "R",
			Path::DOWN => "D",
			Path::ROTATE => "U",
		})
		.collect::<String>()
} 

pub fn reconstruct_game(game: &Vec<State>, is_full_game: bool) -> String {
	let mut game_path = vec![];
	for g in 0..game.len()-1 {
		for p in 0..PIECE_COUNT {
			let piece_paths = get_states(&game[g], p);
			match piece_paths.get(&game[g+1]) {
				Some(path) => {
					game_path.push(vec_to_2bit(path));
					break
				},
				None => continue
			}
		}
	};

	// If it's a full game, we need to pad out with downs to make absolutely sure the replay ends.
	// I don't know the maximum theoretical number of downs needed to end the game, so 6 ought to do it.
	// We don't want to do this if it's not a full game - e.g. if we're extracting loop segments.

	if is_full_game {
		game_path.push("DDDDDD".to_string());
	}

	return game_path.join("")
}
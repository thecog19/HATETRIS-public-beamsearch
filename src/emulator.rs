use crate::constants::{EFF_HEIGHT, MAX_ROW, MIN_LOOP, ROTATE_LEFT, ROTATE_RIGHT, WAVE_SIZE, WELL_HEIGHT, WELL_LINE};
use crate::neural::{decompose_well, forward_pass};
use crate::pieces::{PIECE_COUNT, PIECE_LIST};
use crate::masks::{EMPTY_MASKS, ROW_MASKS, HEIGHT_MASKS, SCORE_MASKS};
use crate::types::{ScoreT, State, RowT, WaveT, WeightT, WellT, SearchConf, StateP};

use std::cmp::{max, min};
use std::collections::VecDeque;

use fnv::FnvHashMap;

// The 'height' of a waveform is the height of the row *below* the bottommost row of the waveform.

pub fn well_slice(height: usize, well: &WellT) -> [RowT; 4] {
	let well_slice = [
		if (height <= 3) {0} else if (height - 4 >= EFF_HEIGHT) {MAX_ROW} else {well[height - 4]},
		if (height <= 2) {0} else if (height - 3 >= EFF_HEIGHT) {MAX_ROW} else {well[height - 3]},
		if (height <= 1) {0} else if (height - 2 >= EFF_HEIGHT) {MAX_ROW} else {well[height - 2]},
		if (height <= 0) {0} else if (height - 1 >= EFF_HEIGHT) {MAX_ROW} else {well[height - 1]}
	];
	return well_slice
}

pub fn waveform_to_wells(wave: WaveT, height: usize, p: usize, state: &State) -> Vec<State> {
	let well = state.well;
	let old_score = state.score;

	let mut wells = vec![];
	let mut w = wave;
	for i in (0..WAVE_SIZE).rev() {
		if (w % 2 == 1) {
			let slice = PIECE_LIST[p][i];
			let mut new_well = [0; EFF_HEIGHT];

			let mut score = 0;
			for row in (0..EFF_HEIGHT).rev() {
				let mut new_val = well[row];
				if row <= height - 1 && row + 4 > height - 1 {
					new_val |= slice[3 - (height - 1 - row)];
				}
				if new_val == MAX_ROW {
					score += 1;
				} else {
					new_well[row + score] = new_val;
				}
			}

			wells.push(State{well: new_well, score: old_score + score as ScoreT});
		};
		w >>= 1;
	}

	wells.sort();
	wells.dedup();

	return wells

}

pub fn waveform_step(w_old: WaveT, p: usize, height: usize, well: &WellT) -> WaveT {
	let well_slice = well_slice(height, well);

	let mut mask = EMPTY_MASKS[p];
	for (r, row) in well_slice.iter().enumerate() {
		mask &= ROW_MASKS[p][*row as usize][r];
	}

	let mut w = w_old & mask;
	let mut w_new = w;
	let mut w_seen = w;
	while w_new > 0 {
		let w_right = w >> 4;
		let w_left = w << 4;
		let w_rotate = ((w & ROTATE_LEFT) << 3) | 
							((w & ROTATE_RIGHT) >> 1);
		w |= w_right;
		w |= w_left;
		w |= w_rotate;
		w &= mask;
		w_new = w & !w_seen;
		w_seen |= w;

		// TODO: See if there's a way of doing this with only 1 intermediate variable instead of 2.
	}

	return w
}

pub fn get_well_height(well: &WellT) -> usize {
	let mut height = 0;
	while height < EFF_HEIGHT {
		if well[height] != 0 {
			break
		};
		height += 1;
	}
	return height
}

pub fn resting_waveforms(p: usize, well: &WellT) -> Vec<(WaveT, usize)> {
	let mut height = get_well_height(well);

	let mut waves = Vec::with_capacity(EFF_HEIGHT - height + 4);
	let mut w = EMPTY_MASKS[p];

	while w > 0 && height + 1 < WELL_HEIGHT {
		w = waveform_step(w, p, height, &well);
		let h_mask = 
			match height {
				0 => HEIGHT_MASKS[p][3],
				1 => HEIGHT_MASKS[p][2],
				2 => HEIGHT_MASKS[p][1],
				3 => HEIGHT_MASKS[p][0],
				_ => 0
			};

		waves.push((w & !h_mask, height));
		height += 1;

	}

	// TODO: Incorporate this into the wave list generation, to minimize .push() operations and 0-value waveforms.
	for i in 0..(waves.len() - 1) {
		waves[i].0 &= !waves[i+1].0;
	}

	return waves
}

pub fn score_slice(wave: WaveT, height: usize, p: usize, well: &WellT) -> [WaveT; 4] {
	let well_slice = well_slice(height, well);
	let mut score_slice = [0; 4]; 
	for i in 0..4 {
		score_slice[i] = SCORE_MASKS[p][well_slice[i] as usize][i] & wave;
	};

	return score_slice
}

pub fn scores(wave: WaveT, height: usize, p: usize, well: &WellT) -> [WaveT; 5] {
	let score_slice = score_slice(wave, height, p, well);

	let mut score = [0; 5];

	score[0] = (!score_slice[0] & 
				!score_slice[1] & 
				!score_slice[2] & 
				!score_slice[3]) & wave;
	if score[0] == wave {return score}

	score[1] = (score_slice[0] ^ 
				score_slice[1] ^ 
				score_slice[2] ^ 
				score_slice[3]) & wave;
	if score[1] | score[0] == wave {return score}

	score[2] = ((score_slice[0] & score_slice[1]) ^ 
				(score_slice[0] & score_slice[2]) ^ 
				(score_slice[0] & score_slice[3]) ^ 
				(score_slice[1] & score_slice[2]) ^ 
				(score_slice[1] & score_slice[3]) ^ 
				(score_slice[2] & score_slice[3])) & wave;
	if score[2] | score[1] | score[0] == wave {return score}

	score[3] = ((score_slice[0] & score_slice[1] & score_slice[2]) ^
				(score_slice[0] & score_slice[1] & score_slice[3]) ^
				(score_slice[0] & score_slice[2] & score_slice[3]) ^
				(score_slice[1] & score_slice[2] & score_slice[3])) & wave;
	if score[3] | score[2] | score[1] | score[0] == wave {return score}

	score[4] = (score_slice[0] & score_slice[1] & score_slice[2] & score_slice[3]) & wave;

	return score
}

pub fn get_wave_height(wave: WaveT, wave_height: usize, p: usize, well: &WellT) -> isize {
	// We only care about the lowest possible height of all the pieces.

	let well_height = get_well_height(well) as isize;
	if wave == 0 {return -1 * (WELL_LINE as isize)}

	let scores = scores(wave, wave_height, p, well);

	let mut max_height = -1 * (WELL_LINE as isize);

	let mut wsc = [0; 5];
	let mut total = 0;
	for s in 0..5 {
		wsc[s] = scores[s] & wave;
		
		for (row, h) in HEIGHT_MASKS[p].iter().enumerate() {
			if h & wsc[s] != 0 {
				let tmp_height = min(well_height, (wave_height + row) as isize - 4) + s as isize;
				max_height = max(max_height, tmp_height);
			}
			if h & wsc[s] == wsc[s] {
				break;
			}
		}

		total |= wsc[s];
		if total == wave {break};
	}

	return max_height
}

pub fn get_legal(state: &State) -> (usize, Vec<Vec<(WaveT, usize)>>) {
	let all_waves: Vec<Vec<(WaveT, usize)>> = (0..PIECE_COUNT)
	.map(|p| resting_waveforms(p, &state.well))
	.collect();

	let mut legal_p = 0;
	let mut lowest_height = WELL_HEIGHT as isize;

	for p in 0..PIECE_COUNT {
		let mut piece_height = -1 * (WELL_LINE as isize);
		for wave in &all_waves[p] {
			let new_height = get_wave_height(wave.0, wave.1, p, &state.well);
			if new_height > piece_height {
				piece_height = new_height;
			}
		}
		if piece_height < lowest_height {
			legal_p = p;
			lowest_height = piece_height;
		}
	}

	return (legal_p, all_waves)
}

pub fn single_move(state: &State) -> Vec<State> {
	let (piece, all_waves) = get_legal(&state);

	let mut to_return = vec![];
	for (w, h) in &all_waves[piece] {
		let mut w_list = waveform_to_wells(*w, *h, piece, state);
		to_return.append(&mut w_list);
	}

	return to_return
}

fn quiescent_heuristic(heuristics: &mut Vec<(State, f64)>, weight: &WeightT) -> () {
	let tmp_conf = SearchConf::single();
	
	// Has key (well, depth) and value Vec<ancestor_index>.
	let mut wells_to_evaluate = FnvHashMap::default();
	let mut total_wells = vec![0];

	for i in 0..heuristics.len() {
		if !wells_to_evaluate.contains_key(&(heuristics[i].0.clone(), 0)) {
			wells_to_evaluate.insert((heuristics[i].0.clone(), 0), vec![i]);
			total_wells[0] += 1;
		} else {
			// Multiple wells can lead to the same end state.
			// We want to update all ancestors of that end state without duplication.

			let mut common_ancestors = wells_to_evaluate.get(&(heuristics[i].0.clone(), 0)).unwrap().clone();
			common_ancestors.push(i);
			wells_to_evaluate.insert((heuristics[i].0.clone(), 0), common_ancestors);
		}
	}

	let mut heuristic_map = FnvHashMap::default();
	for i in 0..heuristics.len() {
		heuristic_map.insert(heuristics[i].0.clone(), heuristics[i].1.clone());
	}

	while wells_to_evaluate.len() > 0 {
		total_wells.push(0);
		let tmp_depth = total_wells.len()-1;
		let mut queued_wells = FnvHashMap::default();
		for wev in wells_to_evaluate.iter() {
			total_wells[tmp_depth] += 1;
			let prev_score = heuristics[wev.1[0]].0.score;
			
			'piece: for p in 0..PIECE_COUNT {
				let mut tmp_queue = vec![];
	
				let waves = resting_waveforms(p, &wev.0.0.well);
				for wave in waves {
					let slice = score_slice(wave.0, wave.1, p, &wev.0.0.well);
					let mut new_w = (0, wave.1);
					for s in slice {
						if s > 0 {
							new_w.0 |= wave.0 & s;
						}
					}
					if new_w.0 > 0 {
						let new_wells = waveform_to_wells(new_w.0, new_w.1, p, &wev.0.0);
						for w in new_wells {
							if w.score - prev_score != wev.0.1 + 1 {
								continue 'piece // If the piece can be used to clear more than 1 line, skip the entire piece.
							} else {
								tmp_queue.push(w);
							}
						}
					}
				}
	
				for well in tmp_queue {
					if !queued_wells.contains_key(&(well.clone(), wev.0.1 + 1)) {
						queued_wells.insert((well.clone(), wev.0.1 + 1), wev.1.clone());
	
						let h = network_heuristic_individual(&well, weight, &tmp_conf);
						for &id in wev.1 {
							heuristics[id].1 = heuristics[id].1.max(h);
						}
						heuristic_map.insert(well, h);
					} else {
						let mut common_ancestors = queued_wells.get(&(well.clone(), wev.0.1 + 1)).unwrap().clone();
						let mut to_update = wev.1.clone();
						common_ancestors.append(&mut to_update);
						queued_wells.insert((well.clone(), wev.0.1 + 1), common_ancestors);
						let h = *heuristic_map.get(&well).unwrap();
						for &id in wev.1 {
							heuristics[id].1 = heuristics[id].1.max(h);
						}
					}
				};
	
				// We only care about the first available piece in the ordering.
				// Once we have one that doesn't clear two lines, we're done.
				// For the standard HATETRIS algorithm, this means either S or Z.

				break 'piece
			}
		};
		wells_to_evaluate.clear();
		wells_to_evaluate = queued_wells;
	}
	
	// We don't return anything; quiescent_heuristic() modifies `heuristics` in-place.
}

// Global quiescent lookahead caching.
// It turned out to actually be slower, but we'll keep it in the code just in case there's some major speedup possible.
pub fn quiescent_heuristic_2(heuristics: &mut Vec<(State, f64)>, seen_states: &VecDeque<FnvHashMap<State, f64>>, weight: &WeightT) -> Vec<((State, f64), usize)> {
	let tmp_conf = SearchConf::single();
	let h_len = heuristics.len();
	let mut tmp_depth = 0;

	// Tuple elements are (State, heuristic, ancestors, depth, already_evaluated)
	let mut new_states: Vec<(State, f64, Vec<usize>, usize, bool)> = vec![];

	// Has key `State` and value Vec<ancestor_index>.
	let mut states_to_evaluate = FnvHashMap::default();
	
	// We assume that `heuristics` has no duplicates.
	// If it does have duplicates, none but the last duplicate will update.
	for i in 0..heuristics.len() {
		states_to_evaluate.insert(heuristics[i].0.clone(), vec![i]);
		new_states.push((heuristics[i].0.clone(), heuristics[i].1, vec![i], tmp_depth, false));
	}

	while states_to_evaluate.len() > 0 {
		let mut queued_states = FnvHashMap::default();
		for (sev, ancestors) in &states_to_evaluate {
			let mut new_ancestors = ancestors.clone();
			new_ancestors.push(new_states.len());

			if tmp_depth < seen_states.len() {
				match seen_states[tmp_depth].get(&sev) {
					Some(h) => {
						new_states.push((sev.clone(), *h, new_ancestors.clone(), tmp_depth, true));
						continue	
					},
					None => ()
				}
			}

			new_states.push((sev.clone(), -1.0, new_ancestors.clone(), tmp_depth, false));
			
			'piece: for p in 0..PIECE_COUNT {
				let mut tmp_queue = vec![];
				let waves = resting_waveforms(p, &sev.well);
				for (wave, height) in waves {
					let score_wave = score_slice(wave, height, p, &sev.well).iter().fold(0, |acc, x| acc | x);
					if score_wave > 0 {
						let new_wells = waveform_to_wells(score_wave, height, p, &sev);
						for w in new_wells {
							// We only care about the first available piece in the ordering.
							// Once we have one that doesn't clear two lines, we're done.
							// For the standard HATETRIS algorithm, this means either S or Z.
							if w.score != sev.score + 1 {
								continue 'piece
							} else {
								tmp_queue.push(w);
							}
						}
					}
				}

				for tmp_state in tmp_queue {
					if !queued_states.contains_key(&tmp_state) {
						queued_states.insert(tmp_state, new_ancestors.clone());
					} else {
						let previous_ancestors = queued_states.get_mut(&tmp_state).unwrap();
						let mut old_ancestors = new_ancestors.clone();
						previous_ancestors.append(&mut old_ancestors);
						previous_ancestors.sort();
						previous_ancestors.dedup();
					}
				}

				break 'piece
			}
		}

		states_to_evaluate.clear();
		states_to_evaluate = queued_states;
		tmp_depth += 1;
	}

	for i in (0..new_states.len()).rev() {
		let new_h = 
			if !new_states[i].4 {
				network_heuristic_individual(&new_states[i].0, weight, &tmp_conf)
			} else {
				new_states[i].1
			};

		let ancestors = new_states[i].2.clone();

		for ancestor in &ancestors {
			if !new_states[*ancestor].4 {
				new_states[*ancestor].1 = new_states[*ancestor].1.max(new_h);
			}
			if *ancestor < h_len {
				heuristics[*ancestor].1 = heuristics[*ancestor].1.max(new_h);
			}
		}
	}

	let to_return = new_states
		.iter()
		.map(|(s, h, _, d, _)| ((s.clone(), *h), *d))
		.collect();

	return to_return
}

// Gets heuristic for individual well.
// Only to be used when batching is not appropriate.

pub fn network_heuristic_individual(state: &State, weight: &WeightT, conf: &SearchConf) -> f64 {
	let conv_list = decompose_well(&state.well);
	let heuristic = forward_pass(conv_list, weight);
	let quiescent = conf.quiescent;
	
	if !quiescent {
		return heuristic
	}

	let mut heuristics = vec![(state.clone(), heuristic)];

	quiescent_heuristic(&mut heuristics, weight);

	return heuristics[0].1
}

// Used for batches; gets the children and their heuristics.

pub fn network_heuristic(state: &State, weight: &WeightT, conf: &SearchConf) -> Vec<(State, f64)> {
	let legal = single_move(state);
	let quiescent = conf.quiescent;

	let mut heuristics: Vec<(State, f64)> = legal.iter().map(|s| (s.clone(), -1.0)).collect();
	for i in 0..legal.len() {
		let conv_list = decompose_well(&heuristics[i].0.well);
		heuristics[i].1 = forward_pass(conv_list, weight);
	}

	if quiescent {
		quiescent_heuristic(&mut heuristics, weight);
	}

	return heuristics;
}


pub fn network_heuristic_loop(state: &State, parent_index: usize, parents: &Vec<StateP>, weight: &WeightT, conf: &SearchConf) -> 
	(Vec<(State, f64)>, Vec<Vec<State>>) {

	let all_waves: Vec<Vec<(WaveT, usize)>> = (0..PIECE_COUNT)
		.map(|p| resting_waveforms(p, &state.well))
		.collect();

	let mut piece_order = vec![];

	for p in 0..PIECE_COUNT {
		let mut piece_height = -1 * (WELL_LINE as isize);
		for wave in &all_waves[p] {
			let new_height = get_wave_height(wave.0, wave.1, p, &state.well);
			if new_height > piece_height {
				piece_height = new_height;
			}
		}
		piece_order.push((piece_height, p));
	};
	piece_order.sort();

	let mut loop_list = vec![];
	
	for (_, legal_p) in piece_order {
		let mut legal = vec![];
		for (w, h) in &all_waves[legal_p] {
			let mut w_list = waveform_to_wells(*w, *h, legal_p, state);
			legal.append(&mut w_list);
		}

		let mut heuristics: Vec<(State, f64)> = legal.iter().map(|s| (s.clone(), -1.0)).collect();
		for i in 0..legal.len() {
			let conv_list = decompose_well(&heuristics[i].0.well);
			heuristics[i].1 = forward_pass(conv_list, weight);
		}

		if conf.quiescent {
			quiescent_heuristic(&mut heuristics, weight);
		}

		// LOOP DETECTION

		let mut max_heuristic: f64 = -1.0;
		for (_s, h) in &heuristics {
			max_heuristic = max_heuristic.max(*h);
		};

		let mut j = parent_index;
		let mut d = parents[j].depth;
		
		let mut loop_ends = vec![];
		let original_d = d + 1; // The current well is at a depth one greater than its parent.

		'outer: while d > 0 {
			d = parents[j].depth;
			
			let min_prev = parents[j].min_prev_heuristic;
			let curr = parents[j].heuristic;
			let parent_state = parents[j].convert_state();

			j = parents[j].parent_index;

			if min_prev > max_heuristic {
				break 'outer
			} else if curr > max_heuristic {
				continue 'outer
			} else {
				// It's impossible for a loop length to not be a multiple of MIN_LOOP.
				if (original_d - d) % MIN_LOOP == 0 {
					for (s, _h) in heuristics.iter() {
						if parent_state.well == s.well {
							// Get actual repeat well, since we won't save it otherwise.

							loop_ends.push(s.clone());

							continue 'outer
							// This accounts for an edge case:
							//	- Well W has children A and B. 
							//	- Child A closes loop back to Parent P_A.
							//	- Child B closes loop back to Parent P_B.
						}
					}
				}
			}
		}

		if loop_ends.len() == 0 {
			return (heuristics, loop_list)
		}
		
		for end in loop_ends {
			let mut j = parent_index;
			let mut d = parents[j].depth;

			let mut tmp_loop_list = Vec::with_capacity(d+2);
			tmp_loop_list.push(end);

			while d > 0 {
				d = parents[j].depth;
				tmp_loop_list.push(parents[j].convert_state());
				j = parents[j].parent_index;
			}
			loop_list.push(tmp_loop_list);
		}
	}

	// TODO: Account for the (potentially impossible) case of all pieces allowing a repeat well.

	return (vec![], vec![])
}
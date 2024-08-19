use crate::constants::{EFF_HEIGHT, MAX_ROW, VERSION, MULTIPLIER};
use crate::emulator::{single_move, network_heuristic_individual, network_heuristic};
use crate::types::{SearchConf, State, StateH, WeightT, StateP};

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

use fnv::{FnvHashMap, FnvHashSet};

use savefile::prelude::*;

pub fn complete_search(starting_state: &State) -> () {
	let start = Instant::now();
		
	let mut move_list = HashSet::new();
	move_list.insert(starting_state.clone());

	let mut depth = 0;
	let mut max_score = starting_state.score;
	let mut max_width = 1;

	while move_list.len() > 0 {
		let mut new_list = HashSet::new();
		let mut children_count = 0;
		for m in &move_list {
			let new_wells = single_move(&m);
			for s in new_wells {
				new_list.insert(s);
				children_count += 1;
			}
		};
		
		move_list = new_list.clone();
		if move_list.len() == 0 {
			break;
		}
		let mut best_by_score = move_list.iter().next().unwrap().clone();

		let mut families: Vec<usize> = vec![0; MAX_ROW as usize + 1];
		let mut scores = vec![0; best_by_score.score as usize + 1];
		for w in move_list.iter() {
			families[w.well[EFF_HEIGHT - 1 as usize] as usize] += 1;
			while scores.len() <= w.score as usize {
				scores.push(0);
				best_by_score = w.clone();
			}
			scores[w.score as usize] += 1;
		}
		families.sort();
		families.reverse();

		depth += 1;
		max_score = max_score.max(best_by_score.score);
		max_width = max_width.max(move_list.len());
		
		println!("");
		println!("Depth {}", depth);
		println!("Time: {} seconds", start.elapsed().as_secs());
		println!("Total children: {}", children_count);
		println!("New well count: {}", move_list.len());
		println!("Maximum score: {:?}", best_by_score);
		println!("Family distribution: {:?}", &families[0..10]);
		println!("Score distribution: {:?}", scores);
	}

	println!("");
	println!("COMPLETE SEARCH SUMMARY: Width: {}; Score: {}; Depth: {}", max_width, max_score, depth);
}

pub fn get_keyframes_from_parents(parent_array: &Vec<StateP>) -> Vec<StateH> {
	let mut j = parent_array.len()-1;
	let mut d = parent_array[j].depth;
	let mut keyframes = Vec::with_capacity(d+1);
	while d > 0 {
		d = parent_array[j].depth;
		keyframes.push(parent_array[j].convert_state_h());
		j = parent_array[j].parent_index;
	}

	return keyframes
}

pub fn insert_state(node: &State, h: f64, new_wells: &mut BTreeSet<StateH>, conf: &SearchConf) -> bool {
	let weighted_h = (h * MULTIPLIER) as i64;

	let to_insert = StateH {
						well: node.well, 
						score: node.score, 
						heuristic: weighted_h};

	let length = new_wells.len();
	let first_entry_heuristic = 
		if length == 0 {
			i64::MIN
		} else {
			new_wells.iter().next().unwrap().heuristic
		};

	if ((length == conf.beam_width && to_insert.heuristic > first_entry_heuristic) || length < conf.beam_width) && !new_wells.contains(&to_insert) {
		new_wells.insert(to_insert.clone());

		if length == conf.beam_width {
			let first_entry = new_wells.iter().next().unwrap().clone();
			new_wells.remove(&first_entry);
		}
		return true
	} else {
		return false
	}
}

fn init_from_save(conf: &SearchConf, wells: &mut Vec<State>, parents: &mut Vec<StateP>, starting_state: &State, starting_parent: StateP, depth: &mut usize) {
	if conf.save {
		let mut file_name = conf.move_path(*depth);
		if !Path::new(&file_name).exists() {
			wells.push(starting_state.clone());
			if conf.parent {
				parents.push(starting_parent);
			}
			save_file(&file_name, VERSION, wells).unwrap();
		} else {
			while Path::new(&file_name).exists() {
				*depth += 1;
				file_name = conf.move_path(*depth);
			}
			*depth -= 1;
			file_name = conf.move_path(*depth);
			*wells = load_file(&file_name, VERSION).unwrap();
			println!("Loaded {} positions from depth {}", wells.len(), depth);
			if conf.parent {
				let parent_file_name = conf.parent_path(*depth);
				*parents = load_file(&parent_file_name, VERSION).unwrap();
				println!("Loaded {} parents from depth {}", parents.len(), depth);
			}
		}
	} else {
		wells.push(starting_state.clone());
		if conf.parent {
			parents.push(starting_parent);
		}
	}
}

fn print_progress(conf: &SearchConf, wells: &Vec<State>, depth: usize, children_count: usize, start: &Instant, weight: &WeightT) {
    let mut families: Vec<usize> = vec![0; MAX_ROW as usize + 1];
    let mut scores = vec![0; wells[0].score as usize + 1];
    let mut best_by_score = wells[0].clone();
    
    for w in wells.iter() {
        families[w.well[(EFF_HEIGHT-1) as usize] as usize] += 1;
        if w.score >= best_by_score.score {
            best_by_score = w.clone();
            while scores.len() <= w.score as usize {
                scores.push(0);
            }
        }
        scores[w.score as usize] += 1;
    }
    families.sort();
    families.reverse();

    let score_best_h = StateH {
        well: best_by_score.well.clone(),
        score: best_by_score.score.clone(),
        heuristic: (network_heuristic_individual(&best_by_score, weight, conf) * 1_000_000.0) as i64
    };

    let worst = wells[0].clone();
    let worst_h = StateH {
        well: worst.well.clone(),
        score: worst.score.clone(),
        heuristic: (network_heuristic_individual(&worst, weight, conf) * 1_000_000.0) as i64
    };

    let best = wells[wells.len() - 1].clone();
    let best_h = StateH {
        well: best.well.clone(),
        score: best.score.clone(),
        heuristic: (network_heuristic_individual(&best, weight, conf) * 1_000_000.0) as i64
    };

    println!("");
    println!("Depth {}", depth);
    println!("Time: {} seconds", start.elapsed().as_secs());
    println!("Total children: {}", children_count);
    println!("New well count: {}", wells.len());
    println!("Maximum score: {:?}", score_best_h);
    println!("Worst heuristic: {:?}", worst_h);
    println!("Best heuristic: {:?}", best_h);
    println!("Family distribution: {:?}", &families[0..10]);
    println!("Score distribution: {:?}", scores);
}

pub fn beam_search_network(starting_state: &State, weight: &WeightT, conf: &SearchConf) -> f64  {
	let beam_width = conf.beam_width;
	let beam_depth = conf.beam_depth;
	if conf.save {
		fs::create_dir_all(conf.replay_path()).expect("Could not create replay folder.");
	}

	let mut wells = Vec::with_capacity(beam_width + 1);
	let mut depth: usize = 0;
	let mut parents = if conf.parent {
		Vec::with_capacity(3 * beam_width)
	} else {
		Vec::with_capacity(1)
	};

	let starting_parent = StateP {
		well: starting_state.well.clone(),
		score: starting_state.score,
		heuristic: network_heuristic_individual(starting_state, weight, conf),
		min_prev_heuristic: f64::MAX,
		depth: 0,
		parent_index: usize::MAX, // This would cause a panic were it ever accessed.
	};

	init_from_save(conf, &mut wells, &mut parents, starting_state, starting_parent, &mut depth);
	
	let start = Instant::now();

	while wells.len() > 0 && depth < beam_depth {
		depth += 1;
		let mut new_wells: BTreeSet<StateH> = BTreeSet::new();

		let mut new_parents: Vec<StateP> = parents.clone();

		let mut children_count = 0;

		if conf.parent {		
			for (p, parent) in parents.iter().enumerate() {
				if parent.depth != depth - 1{
					continue
				}

				let well = &parent.convert_state();
				let full_legal = network_heuristic(well, &weight, &conf);
				children_count += full_legal.len();
					
				for (node, h) in full_legal {
					let new_parent = StateP { 
						well: node.well.clone(), 
						score: node.score,
						heuristic: h,
						min_prev_heuristic: h.min(parent.min_prev_heuristic),
						depth: parent.depth + 1,
						parent_index: p
					};

					let did_insert = insert_state(&node, h, &mut new_wells, &conf);
					if did_insert {
						new_parents.push(new_parent);
					}
				};
			}
			
		} else {
			for well in wells.iter() {
				let full_legal = network_heuristic(well, weight, conf);
				children_count += full_legal.len();

				for (node, h) in full_legal {
					insert_state(&node, h, &mut new_wells, &conf);
				}
			}
		}

		if new_wells.len() == 0 {
			break;
		}

		if conf.parent {
			let mut index_hash_set = FnvHashSet::with_capacity_and_hasher(2 * new_wells.len(), Default::default());
			for i in 0..new_parents.len() {
				if new_parents[i].depth == depth && new_wells.contains(&new_parents[i].convert_state_h()) {
					let mut j = i;
					let mut d = depth;
					while d > 0 {
						d = new_parents[j].depth;
						index_hash_set.insert(j);
						j = new_parents[j].parent_index;
					}
				}
			}
			let mut indices: Vec<usize> = index_hash_set.drain().collect();
			indices.sort();

			let mut index_hash_map = FnvHashMap::with_capacity_and_hasher(indices.len(), Default::default());
			index_hash_map.insert(usize::MAX, usize::MAX);

			parents.clear();
			for (new_i, old_i) in indices.iter().enumerate() {
				let mut new_parent = new_parents[*old_i].clone();
				new_parent.parent_index = *index_hash_map.get(&new_parent.parent_index).unwrap();
				parents.push(new_parent);
				index_hash_map.insert(*old_i, new_i);
			}
			parents.dedup();
		}

		wells.clear();
		for w in new_wells {
			wells.push(State::convert(w));
		}

		if conf.save {
			let file_name = conf.move_path(depth);
			save_file(&file_name, VERSION, &wells).unwrap();

			if conf.parent {
				let parent_file_name = conf.parent_path(depth);
				save_file(&parent_file_name, VERSION, &parents).unwrap();
			}
		}

		if conf.print {
			print_progress(conf, &wells, depth, children_count, &start, weight);
		}
	}

	if conf.print && conf.parent {
		println!("");
		let keyframes = get_keyframes_from_parents(&parents);
		for k in keyframes {
			println!("{:?}", k);
		}
	}

	if conf.print {
		let max_score = parents.iter().map(|s| s.score).max().unwrap();

		println!("");
		println!("GENERATION {} SUMMARY: Width: {}; Score: {}; Depth: {}", conf.generation, conf.beam_width, max_score, depth);
	}

	if depth <= beam_depth || beam_depth == 0 {
		return -1.0
	} else {
		let mut best_heuristic = -1.0;
		for p in parents {
			if p.depth != depth {
				continue
			} else if p.heuristic > best_heuristic {
				// Annoying that this isn't a simple one-liner.
				best_heuristic = p.heuristic
			}
		}

		return best_heuristic
	}
}
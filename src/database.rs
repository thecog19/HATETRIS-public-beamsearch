use crate::constants::VERSION;
use crate::types::State;
use crate::types::SearchConf;

use std::path::Path;
use std::fs;
use std::time::Instant;

use rand_distr::{WeightedIndex, Distribution};
use rand::{thread_rng, seq::SliceRandom};

use savefile::prelude::*;

pub fn extract_data_points(count: usize, conf: &SearchConf) -> Vec<State> {
	// Note that this random weighting assumes the files contain only Vec<State>.
	// Vec<State; N> takes up N*(ScoreT + 2 * EFF_HEIGHT) + 85 = 34*N + 85 bytes. 
	// It may give wrong results otherwise.
	let start = Instant::now();

	let mut byte_counts: Vec<u64> = vec![];
	let mut depth = 0;

	let mut file_name = conf.move_path(depth);
	while Path::new(&file_name).exists() {
		let metadata = fs::metadata(&file_name).unwrap();
		byte_counts.push(metadata.len());
		depth += 1;
		file_name = conf.move_path(depth);
	}
	depth -= 1;
	let state_count: Vec<usize> = byte_counts.iter().map(|&x| ((x - 85)/34) as usize).collect();
	let len = state_count.len();
	let sum = state_count.iter().sum::<usize>();
	
	println!("{} states identified from {} timesteps in {} seconds.", sum, len, start.elapsed().as_secs());

	let dist = WeightedIndex::new(state_count).unwrap();
	let mut rng = thread_rng();
	let mut chosen = vec![0; len];

	for _ in 0..count {
		chosen[dist.sample(&mut rng)] += 1;
	}

	println!("{} selections allocated in {} seconds.", count, start.elapsed().as_secs());

	let mut to_return = Vec::with_capacity(count);
	for d in 0..=depth {
		let file_name = conf.move_path(d);
		let wells: Vec<State> = load_file(&file_name, VERSION).unwrap();
		for _ in 0..chosen[d] {
			// We're choosing one at a time to allow repeats.
			to_return.push(wells.choose(&mut rng).unwrap().clone());
		}
		println!("{} states extracted from timestep {} in {} seconds.", chosen[d], d, start.elapsed().as_secs());
	}
	to_return.shuffle(&mut rng);

	return to_return
}



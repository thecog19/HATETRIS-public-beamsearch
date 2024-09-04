#![allow(unused_parens)]

pub mod constants;
pub mod database;
pub mod emulator;
pub mod masks;
pub mod neural;
pub mod pieces;
pub mod replay;
pub mod searches;
pub mod types;

// use neural::training_cycle;

use constants::NET_VERSION;
use savefile::load_file;
use searches::beam_search_network_full;
use types::{SearchConf, State};

fn main() {
	// training_cycle();

	let mut conf = SearchConf::testing();
	conf.quiescent = true;
	let gens = [0];

	for i in 10..=10 {
		for gen in gens.iter() {
			conf.generation = *gen;
			let neural_network_path = conf.neural_network_path(); 
			let weight = load_file(&neural_network_path, NET_VERSION).unwrap();
			let starting_state = State::new();

			conf.beam_width = 2_usize.pow(i);
			println!("");
			println!("Starting beam with width {}", conf.beam_width);

			beam_search_network_full(&starting_state, &weight, &conf);
		}
	}
}

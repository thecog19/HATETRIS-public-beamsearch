use crate::constants::{AEON, ALL_CONV, ALPHA, CONV_COUNT, CONVOLUTIONS, EFF_HEIGHT, EPS, HIDDEN, RHO, RHO_F, WIDTH, MINIBATCH, MAX_EPOCHS, THREAD_NUMBER, VERSION, NET_VERSION};
use crate::database::{extract_data_points};
use crate::searches::beam_search_network;
use crate::types::{State, RowT, WeightT, WellT, SearchConf};

// use std::arch::x86_64::{__m256d, _mm256_add_pd};
// use std::simd::f64x4;
use std::thread;
use std::thread::JoinHandle;

use std::fs;
use std::path::Path;
use std::time::{Instant};

use savefile::prelude::*;

// Decomposes well into indices of MxN convolutions:
//	M is the height of the convolution.
//	N is the width of the convolution.

// It's worrying that this takes almost 1 μs for a single pass.
// If the timing persists even for warm cache, investigate this as possible bottleneck.
// Since the time is mostly vector memory allocation, maybe have this return an iterator?

pub fn decompose_well(well: &WellT) -> [usize; CONV_COUNT] {
	let mut to_return = [0; CONV_COUNT];
	let mut offset = 0;
	let mut count = 0;
	for (m, n) in CONVOLUTIONS {
		let mut mask = ((1 << n) - 1) as RowT;
		let mut masked_well = well.clone();
		for r in 0..EFF_HEIGHT {
			masked_well[r] &= (mask as RowT);
		};
		for col in 0..=(WIDTH - n) {
			for r in 0..EFF_HEIGHT {
				masked_well[r] = (well[r] & mask) >> col;
			};
			for row in 0..=(EFF_HEIGHT - m) {
				let mut conv: usize = 0;
				for i in 0..m {
					conv <<= n;
					conv += (masked_well[row + i] as usize);
				}
				to_return[count] = conv + offset;
				count += 1;
				offset += 1 << (m*n);
			}
			mask <<= 1;
		}
	}

	return to_return
}


// Takes convolution list and weights, and returns total loss.
// This does not keep or return internal neuron values.

pub fn forward_pass(
		conv_list: [usize; CONV_COUNT], 
		weight: &WeightT
	) -> f64 {

	let mut output = 0.0;
	let mut hidden = [0.0; HIDDEN];

	for c in conv_list {
		for h in 0..HIDDEN {
			hidden[h] += weight.conv[c][h];
		}
	}

	for h in 0..HIDDEN {
		hidden[h] = hidden[h].tanh();
		output += hidden[h] * weight.hidden[h];
	}
	output = output.tanh();

	return output
}

// Takes convolution list and weights, and returns total loss.
// This does not keep or return internal neuron values.

// pub unsafe fn forward_pass_chunk(
// 	conv_list: [usize; CONV_COUNT], 
// 	weight: &WeightChunkT
// ) -> f64 {

// 	let mut output = 0.0;
// 	let mut hidden = vec![];
// 	for _ in (0..HIDDEN).step_by(CHUNK) {
// 		hidden.push(__m256d::from(f64x4::from_array([0.0; CHUNK])));
// 	}

// 	for c in conv_list {
// 		for h in 0..hidden.len() {
// 			let tmp = _mm256_add_pd(hidden[h], weight.conv[c][h]);
// 			hidden[h] = tmp;
// 		}
// 	}

// 	let mut unpacked_hidden = Vec::with_capacity(HIDDEN);
// 	for h in 0..hidden.len() {
// 		let unpack = f64x4::from(hidden[h]).as_array().clone();
// 		for u in unpack {
// 			unpacked_hidden.push(u);
// 		}
// 	}

// 	let mut unpacked_weight = Vec::with_capacity(HIDDEN);
// 	for h in 0..weight.hidden.len() {
// 		let unpack = f64x4::from(weight.hidden[h]).as_array().clone();
// 		for u in unpack {
// 			unpacked_weight.push(u);
// 		}
// 	}

// 	for h in 0..HIDDEN {
// 		unpacked_hidden[h] = unpacked_hidden[h].tanh();
// 		output += unpacked_hidden[h] * unpacked_weight[h];
// 	}
// 	output = output.tanh();

// return output
// }

// Takes convolution list and weights, and returns a tuple (inputs, hidden, loss).
// Used for training and backpropagation.

pub fn forward_pass_memory(
		conv_list: [usize; CONV_COUNT], 
		weight: &WeightT
	) -> ([f64; HIDDEN], f64) {

	//let mut inputs = [0.0; ALL_CONV];
	let mut pre_hidden = [0.0; HIDDEN];
	let mut hidden = [0.0; HIDDEN];
	let mut pre_output = 0.0;
	let output: f64;

	// for c in conv_list {
	// 	inputs[c] += 1.0;
	// }

	for c in conv_list {
		for h in 0..HIDDEN {
			pre_hidden[h] += weight.conv[c][h];
		}
	}
	for h in 0..HIDDEN {
		hidden[h] = pre_hidden[h].tanh();
		pre_output += hidden[h] * weight.hidden[h];
	}
	output = pre_output.tanh();

	return (hidden, output)
}

pub fn tanh_discrete(x: i16, s: u8) -> i16 {
	let n = x >> s;
	let num = n.abs().min(127) - 127;
	return (x.signum() * (126 - ((num * num) >> 7))) as i16
}

// Takes training data and backpropagates until either:
//	The neural network loss goes up 3 times in a row.
//	The training data is exhausted.
// Training data is evaluated in sets of MINIBATCH.
// Implements Adam optimizer.

pub fn train_network(
		raw_training: &Vec<(WellT, f64)>,
		weight: &WeightT,
		_conf: &SearchConf)
	 -> WeightT {
	
	let mut new_weights = weight.clone();

	let mut exp_ave_weights = WeightT::zero();
	let mut smooth_grad_weights = WeightT::zero();

	let mut best_weights = weight.clone();
	let mut consecutive = 0;

	let mut rho = RHO;
	let mut rho_f = RHO_F;

	let mut epoch = 0;

	// We need to normalize the training data so that the maximum goal is 1.0 and the minimum goal is -1.0.
	let mut min: f64 = 1.0;
	let mut max: f64 = -1.0;
	for (_, g) in raw_training {
		min = min.min(*g);
		max = max.max(*g);
	}
	// The normalization for the specific case of -1 to 1 works out to (2g - a) / b.
	let a = max + min;
	let b = max - min;
	let training = raw_training
						.iter()
						.map(|(w, g)| (*w, (2.0*g - a)/b))
						.collect::<Vec<(WellT, f64)>>();
						
	let gen_mul = 1.0;
	println!("Baseline α = {}, mul = {}", ALPHA, gen_mul);

	while (epoch + 1) * MINIBATCH < training.len() {
		rho *= RHO;
		rho_f *= RHO_F;
		let alpha_t = gen_mul * ALPHA * (1.0 - rho).sqrt() / (1.0 - rho_f);

		let mut inc = WeightT::zero();
		let mut ave_loss = 0.0;
		let mut ave_pre_loss = 0.0;
		
		let mut goals = vec![0.0; MINIBATCH];

		for t in epoch*MINIBATCH..(epoch + 1)*MINIBATCH {
			let conv_list = decompose_well(&training[t].0);
			let (hidden, output) = forward_pass_memory(conv_list, &new_weights);
			let goal = training[t].1;
			goals[t - epoch*MINIBATCH] = goal;
			
			let loss = (goal - output) * (goal - output);
			ave_pre_loss += loss;
			
			// println!("{}	{}	{}	{}", epoch, t - epoch*MINIBATCH, goal, output);

			// Backpropagation
			//	We treat this three-layer neural network as a five-layer neural network.
			//		Layer 1: Inputs * WI -> pre_hidden		[All -> All]
			//		Layer 2: Tanh(pre_hidden) -> hidden		[One -> One]
			//		Layer 3: hidden * WH -> pre_output		[All -> One]
			//		Layer 4: Tanh(pre_output) -> output		[One -> One]
			//		Layer 5: (goal - output)^2 -> Loss		[One -> One]
			// 	With this decomposition, we start from the end and work backwards.
			
			//	From page 119: g_i = g_(i+1) · (1 - v_(i+1) · v_(i+1))

			// g_LAYER -> gradient of LAYER, elementwise
			// v_LAYER -> value of LAYER, elementwise

			// Layer 5
			//	Loss -> MYSTERY GRADIENT g_output
			//	g_output = d((goal - output)^2, output)
			//	g_output = d(output^2 - 2*output*goal + goal^2, output)
			//			 = 2 * (goal - output) ???
			// Layer 4
			//	g_pre_output = d(tanh(pre_output), pre_output)
			//	g_pre_output = g_output * (1 - v_output^2)
			// Layers 4 + 5
			//	g_pre_output = 2 * (output - goal) * (1 - v_output^2)

			let g_pre_output = 2.0 * (output - goal) * (1.0 - output*output) / (MINIBATCH as f64);

			// Layer 3
			//	g_hidden = Transpose(WH) * g_pre_output
			//	WH_inc = g_pre_output * WH 		[Elementwise]

			// Layer 2
			//	g_pre_hidden = g_hidden * (1 - v_hidden^2)

			let mut g_pre_hidden = [0.0; HIDDEN];
			for h in 0..HIDDEN {
				g_pre_hidden[h] = new_weights.hidden[h] * g_pre_output * (1.0 - hidden[h] * hidden[h]);
				inc.hidden[h] += new_weights.hidden[h] * g_pre_output;
			}

			// Layer 1
			//	g_inputs = Transpose(WI) * g_pre_hidden
			//	WI_inc = Σ_(h in pre_hidden)_(h * WI)		[Elementwise]

			for c in conv_list { // All other nodes c are incremented by zero, by definition.
				for h in 0..HIDDEN {
					inc.conv[c][h] += g_pre_hidden[h];
				}
			}
		}

		// Adam Optimizer
		for c in 0..ALL_CONV {
			// First layer weights
			// WI_inc = Σ_(h in pre_hidden)_(h * WI)		[Elementwise]
			for h in 0..HIDDEN {
				let d_loss = inc.conv[c][h];

				exp_ave_weights.conv[c][h] = 
					rho * exp_ave_weights.conv[c][h] + 
					(1.0 - rho) * d_loss * d_loss;
				smooth_grad_weights.conv[c][h] = 
					rho_f * smooth_grad_weights.conv[c][h] + 
					(1.0 - rho_f) * d_loss;

				new_weights.conv[c][h] -= alpha_t * smooth_grad_weights.conv[c][h] / (exp_ave_weights.conv[c][h].sqrt() + EPS);
			}
		}

		// Second layer weights
		for h in 0..HIDDEN {
			let d_loss = inc.hidden[h];
			exp_ave_weights.hidden[h] = 
				rho * exp_ave_weights.hidden[h] + 
				(1.0 - rho) * d_loss * d_loss;
			smooth_grad_weights.hidden[h] = 
				rho_f * smooth_grad_weights.hidden[h] + 
				(1.0 - rho_f) * d_loss;

			new_weights.hidden[h] -= alpha_t * smooth_grad_weights.hidden[h] / (exp_ave_weights.hidden[h].sqrt() + EPS);
		}
		
		// Second loss calculation pass
		
		for t in epoch*MINIBATCH..(epoch + 1)*MINIBATCH {
			let conv_list = decompose_well(&training[t].0);
			let output = forward_pass(conv_list, &new_weights);
			let goal = goals[t - epoch*MINIBATCH];
			let loss = (goal - output) * (goal - output);
			
			ave_loss += loss;
		}

		ave_pre_loss = ave_pre_loss / (MINIBATCH as f64);
		ave_loss = ave_loss / (MINIBATCH as f64);
		epoch += 1;

		println!("Epoch {}: Loss: {} -> {}", epoch, ave_pre_loss, ave_loss);

		if ave_loss < ave_pre_loss {
			consecutive = 0;
			best_weights = new_weights.clone();
			println!("Saving epoch {}", epoch);
		} else {
			consecutive += 1;
			if consecutive == 3 {
				println!("Loss has gone up for three consecutive epochs, halting training.");
				break;
			}
		}
	};

	return best_weights
}

pub fn generate_training_data(states: Vec<State>, epoch: isize, weight: WeightT, conf: SearchConf) -> JoinHandle<()> {
	let thread = thread::spawn(move || {
		let start = Instant::now();
		let mut training_data = vec![];
		let generation = conf.generation;
		let training_conf = SearchConf::training(generation);

		for well in states {
			let best_result = beam_search_network(&well, &weight, &training_conf) as f64;
			let goal_heuristic = best_result;
			
			training_data.push((well.well.clone(), goal_heuristic));
		}
		
		let epoch_file_name = conf.epoch_path(epoch);
		save_file(&epoch_file_name, 0, &training_data).unwrap();
		println!("Training data generated for epoch {} out of {} in {} seconds.", epoch, MAX_EPOCHS, start.elapsed().as_secs());
	});

	return thread
}

pub fn training_cycle() -> () {
	let mut master_conf = SearchConf::master(0);
	let mut generation = 0;

	// Determine if we are continuing an existing aeon or starting a new one.
	let aeon_folder_name = master_conf.aeon_path();
	if !Path::new(&aeon_folder_name).exists() {
		fs::create_dir_all(aeon_folder_name)
			.expect("Could not create aeon folder.");
		println!("Creating new training cycle.");
	} else {
		println!("Continuing training cycle {}.", AEON);
	}

	let mut weight = WeightT::new();
	let mut gen_folder_name = master_conf.generation_path();
	if !Path::new(&gen_folder_name).exists() {
		fs::create_dir_all(gen_folder_name.clone())
			.expect("Could not create generation folder.");
		println!("No previous generation files found, starting from generation {}", generation);
		
		let neural_network_path = master_conf.neural_network_path();
		save_file(&neural_network_path, NET_VERSION, &weight).unwrap();
		
	} else {
		while Path::new(&gen_folder_name).exists() {
			generation += 1;
			master_conf.generation = generation;
			gen_folder_name = master_conf.generation_path();
		}

		generation -= 1;
		master_conf.generation = generation;

		// See if there is a neural network saved for this generation.

		let neural_network_path = master_conf.neural_network_path(); 
		if Path::new(&neural_network_path).exists() {
			weight = load_file(&neural_network_path, NET_VERSION).unwrap();
			println!("Loading neural network from generation {}", generation);
		} else {
			panic!("No neural network found at generation {}!", generation);
		}
	}

	// Main training loop.
	// We now know what aeon and generation we are in.
	
	loop {
		let training_path = master_conf.training_path();
		if Path::new(&training_path).exists() {
			let mut epoch = 0;
			let mut epoch_file_name = master_conf.epoch_path(epoch);
			while Path::new(&epoch_file_name).exists() {
				epoch += 1;
				epoch_file_name = master_conf.epoch_path(epoch);
			}
			
			epoch -= 1;

			println!("Training path found, training data populated up to {} out of {}.",epoch, MAX_EPOCHS);

			let starting_epoch = (epoch + 1) as usize;
			let mut thread_list: Vec<JoinHandle<()>> = vec![];

			let training_data: Vec<State> = load_file(&master_conf.data_path(), VERSION).unwrap();
			println!("{} training wells loaded.", training_data.len());

			while epoch < MAX_EPOCHS {
				epoch += 1;
				if epoch as usize >= THREAD_NUMBER && 
					thread_list.len() > ((epoch as usize - starting_epoch) % THREAD_NUMBER) {
					thread_list.remove((epoch as usize - starting_epoch) % THREAD_NUMBER).join().unwrap();
				}

				let mut slice = Vec::with_capacity(MINIBATCH);
				for i in (epoch as usize)*MINIBATCH..((epoch + 1) as usize)*MINIBATCH {
					if i >= training_data.len() {break}
					slice.push(training_data[i].clone());
				}

				let t = generate_training_data(slice, epoch, weight.clone(), master_conf.clone());
				
				thread_list.insert((epoch as usize - starting_epoch) % THREAD_NUMBER, t);
				
			};
			for thread in thread_list {
				thread.join().unwrap();
			}

			println!("Backpropagating network from generation {}", generation);

			let mut all_training = vec![];
			for e in 0..MAX_EPOCHS {
				let tmp_file_name = master_conf.epoch_path(e);
				// let tmp_file_name = master_conf.epoch_path(0);
				let mut tmp_training: Vec<(WellT, f64)> = load_file(&tmp_file_name, 0).unwrap();

				all_training.append(&mut tmp_training);
			}

			let new_weight = train_network(&all_training, &weight, &master_conf);
			weight = new_weight.clone();
			
			generation += 1;
			master_conf.generation = generation;

			gen_folder_name = master_conf.generation_path();
			fs::create_dir_all(gen_folder_name)
				.expect("Could not create generation folder.");

			let neural_network_path = master_conf.neural_network_path();
			save_file(&neural_network_path, 0, &weight).unwrap();

			println!("Neural network created for generation {}", generation);
		} else {
			// If the training data folder does not exist, we need to run the beam search.
			// The beam search handles the folder creation etc. internally.

			let conf = SearchConf::master(generation);
			let starting_state = State::new();

			beam_search_network(&starting_state, &weight, &conf);
			let training = extract_data_points(MINIBATCH * (MAX_EPOCHS as usize), &conf);

			// When done, create the training folder for the next loop.
			fs::create_dir_all(training_path.clone())
				.expect("Could not create training data folder.");
				
			// Save the extracted data points in the newly created training data folder.
			save_file(&conf.data_path(), VERSION, &training).unwrap();
		}
	}
}

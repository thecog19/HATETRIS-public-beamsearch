use crate::constants::{VERSION};
use crate::types::{State};
use crate::types::{SearchConf};

use std::path::Path;
use std::fs;
use std::time::Instant;

// use postgres::{Client, NoTls};
use rand_distr::{WeightedIndex, Distribution};
use rand::{thread_rng, seq::SliceRandom};

use savefile::prelude::*;

/* pub fn connect() -> Client {
	let url = "postgresql://felipe:hunter2@localhost:5432";
	let mut client = Client::connect(url, NoTls).unwrap();
	client.batch_execute(&create_well_table()).unwrap();
	client.batch_execute(&create_run_table()).unwrap();
	client.batch_execute(&disable_wal()).unwrap();
	client
}

pub fn create_run_table() -> String {
	// creates a table for runs
	// run_id is the primary key
	// date is the date the run was created
	// name is a user-defined name for the run

	return "CREATE TABLE IF NOT EXISTS runs (
			run_id SERIAL PRIMARY KEY,
			date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			name VARCHAR(100) UNIQUE NOT NULL
		);".to_string();
}

pub fn create_well_table() -> String {
	// Creates a table for wells
	// with a well_state,run_id, future_score, depth and a ranking

	return "CREATE TABLE IF NOT EXISTS wells (
		id SERIAL PRIMARY KEY,
		well_state TEXT NOT NULL,
		run_id INT NOT NULL,
		future_score INT DEFAULT NULL,
		score INT DEFAULT 0,
		depth INT,
		ranking BIGINT
	);".to_string();
}

pub fn disable_wal() -> String {
	return "ALTER TABLE wells SET UNLOGGED;".to_string();
}

// Only put this in an automated script if you want to drop tables
// you will lose all our data 
// I will be sad 
// tears will be shed. 
pub fn drop_tables(client: &mut Client) -> () {
	client.batch_execute("DROP TABLE IF EXISTS wells;").unwrap();
	client.batch_execute("DROP TABLE IF EXISTS runs;").unwrap();
}

pub fn get_random_wells(client: &mut Client, num_wells: usize) -> Vec<postgres::Row> {
	let mut wells = Vec::new();
	let query = format!("SELECT * FROM wells ORDER BY RANDOM() LIMIT {};", num_wells);
	for row in client.query(query.as_str(), &[]).unwrap() {
		let well_state = row;
		wells.push(well_state);
	}
	wells
}

pub fn get_well_state_from_well(row: postgres::Row) -> State {
	let mut new_state = State::new();
	let raw_string: String = row.get("well_state");
	let tokens: Vec<&str> = raw_string.split(|c: char| !c.is_numeric()).filter(|c| !c.is_empty()).collect();
	for (i, token) in tokens.iter().enumerate() {
		let num = token.parse::<u16>().unwrap();
		new_state.well[i] = num;
	}

	return new_state
}

pub fn get_random_wells_for_run(client: &mut Client, num_wells: usize, run_id: i32) -> Vec<postgres::Row> {
	let mut wells = Vec::new();
	let query = format!("SELECT * FROM wells WHERE run_id = {} ORDER BY RANDOM() LIMIT {};", run_id, num_wells);
	for row in client.query(query.as_str(), &[]).unwrap() {
		let well_state = row;
		wells.push(well_state);
	}
	wells
}

pub fn get_random_wells_for_depth(client: &mut Client, num_wells: usize, depth: i32, run_id: i32) -> Vec<postgres::Row> {
	let mut wells = Vec::new();
	let query = format!("SELECT * FROM wells WHERE depth = {} AND where run_id = {} ORDER BY RANDOM() LIMIT {};", depth, run_id, num_wells);
	for row in client.query(query.as_str(), &[]).unwrap() {
		let well_state = row;
		wells.push(well_state);
	}
	wells
}

pub fn add_well_to_table(well_state: &StateD, client: &mut Client) {
	let query = well_state.get_insert_query();
	//println!("{}", query);
	client.batch_execute(
		query.as_str()
	).unwrap();
}

pub fn initialize_run(name: String, client: &mut Client) -> i32 {
	// creates a new run in the database
	// returns the run_id

	client.execute(
		"INSERT INTO runs (name) VALUES ($1)",
		&[&name]
	).unwrap();

	let run_id: i32 = client.query(
		"SELECT run_id FROM runs WHERE name = $1",
		&[&name]
	).unwrap().get(0).unwrap().get(0);

	return run_id;
}

pub fn get_run_id_from_name(name: String, client: &mut Client) -> i32 {
	let run_id: i32 = client.query(
		"SELECT run_id FROM runs WHERE name = $1",
		&[&name]
	).unwrap().get(0).unwrap().get(0);

	return run_id;
}

pub fn populate_database_with_wells(settings: &SearchConf, client: &mut Client) -> () {
	let run_id = get_run_id_from_name(settings.run_name(), client);

	let mut depth = 1;
	let mut file_name = settings.move_path(depth);

	if !Path::new(&file_name).exists() {
		panic!("No moves found in folder {}",settings.move_path(depth));
	} else {
		while Path::new(&file_name).exists() {
			let start = Instant::now();
			
			let wells: Vec<State> = load_file(&file_name, VERSION).unwrap();
			
			for w in &wells {
				let k = StateD::convert(&w, depth as i32, run_id);
				add_well_to_table(&k, client);
			}
			
			println!("Depth {} wells processed in {} ms.", depth, start.elapsed().as_millis());
			
			depth += 1;
			file_name = settings.move_path(depth);
		}
	}
}

pub fn process_files(settings: &SearchConf) -> () {
	// populates the database with the initial wells
	// returns the run_id

	// Path must be specified in the configuration block and passed to this function
	// Multiple path support should be a thing, but isn't until we standarize our file format
	let run_name = settings.run_name();
	let mut client = connect();
	initialize_run(run_name.to_string(), &mut client);
	// if this becomes too slow, switch to a batch insert using COPY
	// https://www.postgresql.org/docs/current/sql-copy.html
	populate_database_with_wells(settings, &mut client);
} */

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



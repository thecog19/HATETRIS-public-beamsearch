#![allow(unused_parens)]

pub mod constants;
pub mod database;
pub mod emulator;
pub mod masks;
pub mod neural;
pub mod pieces;
pub mod searches;
pub mod types;

use crate::constants::{ALL_CONV, EFF_HEIGHT, HIDDEN, WIDTH, WEIGHT_COUNT};
use crate::masks::{EMPTY_MASKS, ROW_MASKS, HEIGHT_MASKS, SCORE_MASKS, SURFACE_LINE_ARRAY};
use crate::neural::{training_cycle};
use crate::pieces::{PIECE_LIST};

extern crate savefile;

fn main() {
	println!("EMPTY_MASKS created with {} elements.", EMPTY_MASKS.len());
	println!("ROW_MASKS created with {} elements.", ROW_MASKS.len());
	println!("HEIGHT_MASKS created with {} elements.", HEIGHT_MASKS.len());
	println!("SCORE_MASKS created with {} elements.", SCORE_MASKS.len());
	println!("PIECE_LIST created with {} elements.", PIECE_LIST.len());
	println!("SURFACE_LINE_ARRAY created with {} elements.", SURFACE_LINE_ARRAY.len());

	println!("");
	println!("Well height: {}, well width: {}", EFF_HEIGHT, WIDTH);
	println!("Neural network contains {} convolutional nodes and {} hidden nodes for {} total weights.", ALL_CONV, HIDDEN, WEIGHT_COUNT);
	println!("");

	training_cycle();
}

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

use neural::training_cycle;

fn main() {
	training_cycle();
}

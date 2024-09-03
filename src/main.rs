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

use crate::{
	masks::load_lazy_statics,
	neural::training_cycle
};

fn main() {
	load_lazy_statics();

	training_cycle();
}

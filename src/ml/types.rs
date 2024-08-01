
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::Array3;
use ndarray::prelude::*;
use std::collections::HashMap;
use std::io::Result as IoResult;



//________________________________________________________________
pub trait Layer {
    // sometimes self has to be muttable
    fn forward(&mut self, training: &bool, input: &Array3<f32>) -> Array3<f32>; 
    fn backward(&mut self, d_out: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32>;
    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>);
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>);
}
//________________________________________________________________
pub trait Opt {
    fn param1(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32>;
    fn param2(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32>;
}
//________________________________________________________________
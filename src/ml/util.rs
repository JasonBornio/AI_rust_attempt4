
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array2, Array3, Array4, stack};
use ndarray::prelude::*;
use rand::Rng;

use std::fs::File;
use std::io::{self,BufRead};
 
use std::path::Path;
use std::sync::Arc;


//________________________________________________________________
pub fn clip(grad: &Array3<f32>, threshold: f32) -> Array3<f32>{
    // Calculate the 2-norm of the gradients
    let norm: f32 = grad.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
    
    // Clip the gradients if their norm exceeds the threshold
    let mut output:Array3<f32> = grad.clone(); 
    if norm > threshold {
        let scale = threshold / norm;
        output = grad * scale;
    }

    output
}
//________________________________________________________________
pub fn clipd(grad: &ArrayD<f32>, threshold: f32) -> ArrayD<f32>{
    // Calculate the 2-norm of the gradients
    let norm: f32 = grad.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
    
    // Clip the gradients if their norm exceeds the threshold
    let mut output:ArrayD<f32> = grad.clone(); 
    if norm > threshold {
        let scale = threshold / norm;
        output = grad * scale;
    }

    output
}
//________________________________________________________________
//can maybe try lanes?
/*
// Assuming `array1` and `array2` are 4D arrays
let lanes1 = array1.lanes(Axis(0));
let lanes2 = array2.lanes(Axis(0));

// Perform dot product on each pair of lanes
let result_lanes: Vec<Array3<f32>> = lanes1
    .iter()
    .zip(lanes2.iter())
    .map(|(lane1, lane2)| lane1.dot(lane2.t()))
    .collect();

// Stack them back together into a 4D array
let result = ndarray::stack(Axis(0), &result_lanes).unwrap();
*/
//________________________________________________________________
pub fn dot4(array1: &Array4<f32>, array2: &Array4<f32>) -> Array4<f32> {
    let (a1_dim_1, a1_dim_2, a1_dim_3, a1_dim_4) = array1.dim();
    let (_, _, a2_dim_3, a2_dim_4) = array2.dim();

    let array1 = array1.as_standard_layout().to_owned();
    let array2 = array2.as_standard_layout().to_owned();
    
    // Reshape arrays
    let arr1_reshaped = array1.view().into_shape((a1_dim_1 * a1_dim_2, a1_dim_3, a1_dim_4)).unwrap();
    //println!("arr1_reshaped: {:?}", arr1_reshaped.shape());
    let arr2_reshaped = array2.view().into_shape((a1_dim_1 * a1_dim_2, a2_dim_3, a2_dim_4)).unwrap();
    //println!("arr2_reshaped: {:?}", arr2_reshaped.shape());

    // Compute dot products for each slice and collect the results
    let mut array3_vec = Vec::new();
    for (slice1, slice2) in arr1_reshaped.axis_iter(Axis(0)).zip(arr2_reshaped.axis_iter(Axis(0))) {
        let result = slice1.dot(&slice2);
        array3_vec.push(result);
    }

    // Convert to views and stack along Axis(0)
    let views: Vec<_> = array3_vec.iter().map(|arr| arr.view()).collect();
    let array3_stack = stack(Axis(0), &views).unwrap();

    // Reshape back to 4D array
    //println!("array3_stack: {:?}", array3_stack.shape());

    let arr3_reshaped = array3_stack.into_shape((a1_dim_1, a1_dim_2, a1_dim_3, a2_dim_4)).unwrap();

    //println!("array3: {:?}", arr3_reshaped.shape());

    arr3_reshaped
}
//________________________________________________________________
pub fn load_embeddings(file: &str) -> (Array1<String>, Array2<f32>) {

    let mut token_path = "embeddings\\".to_string();
    token_path.push_str(file);
    token_path.push_str("_tokens.txt");

    let mut vocab_path = "embeddings\\".to_string();
    vocab_path.push_str(file);
    vocab_path.push_str("_vocab.txt");
    // Path to the file
    let path: &Path = Path::new(&token_path);

    // Open the file
    let file = File::open(&path);

    // Create a buffered reader
    let reader = io::BufReader::new(file.unwrap());

    // Temporary vector to store read lines
    let mut vocab: Vec<String> = Vec::new();
    let mut numbers: Vec<f32> = Vec::new();

    for line in reader.lines() {
        let line = line;
        let line = Arc::new(line.expect("IM FINISHED"));

        vocab.push((*line).clone());
    }

    let path = Path::new(&vocab_path);
    let file = File::open(&path);
    let reader = io::BufReader::new(file.unwrap());

    let mut _init = true;
    let mut range: usize = 0;

    for line in reader.lines() {
        if _init == true {
            let line = line;
            let line = Arc::new(line.expect("IM FINISHED"));
            range = (*line).parse().expect("can't convert string to i32!");
            _init = false;
        }
        else{
            let line = line;
            let line = Arc::new(line.expect("IM FINISHED"));
            let num: f32 = (*line).parse().expect("can't convert string to f32!");
            numbers.push(num);
        }
    }

    let cols = vocab.len();

    let vocabulary: Array1<String> = Array1::from_shape_vec(cols, vocab).unwrap();
    let vectors: Array2<f32> = Array2::from_shape_vec((cols, range), numbers).unwrap();

    (vocabulary, vectors)

}
//________________________________________________________________
pub fn argmax(arr: Array3<f32>, axis: Axis) -> Array2<usize> {

    let shape = arr.shape();
    let axis_to_reduce = axis;

    let mut max_indices = Array2::<usize>::zeros((shape[0], shape[1]));

    // Iterate through the remaining axes
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let mut max_value = f32::NEG_INFINITY;
            let mut max_index = 0;

            // Search for the maximum value along axis 0
            for k in 0..shape[2] {
                let value = arr[(i, j, k)];
                if value > max_value {
                    max_value = value;
                    max_index = k;
                }
            }

            max_indices[(i, j)] = max_index;
        }
    }

    //println!("hiddenstate: {:?}", max_indices);

    max_indices

}
//________________________________________________________________
pub fn argmax2(arr: Array3<f32>, axis: Axis) -> Array2<usize> {

    let shape = arr.shape();
    let axis_to_reduce = axis;

    let mut max_indices = Array2::<usize>::zeros((shape[0], shape[1]));

    // Iterate through the remaining axes
    for i in 0..shape[0] {
        for j in 0..shape[1] {

            let array1: Array1<f32> = arr.slice(s![i, j, ..]).to_owned();

            let max_index = multinomial(&array1);

            max_indices[(i, j)] = max_index;
        }
    }

    //println!("hiddenstate: {:?}", max_indices);

    max_indices

}
//________________________________________________________________
pub fn argmax3(arr: Array3<f32>, axis: Axis) -> Array2<usize> {

    let shape = arr.shape();
    let axis_to_reduce = axis;

    let mut max_indices = Array2::<usize>::zeros((shape[0], shape[1]));

    // Iterate through the remaining axes
    for i in 0..shape[0] {
        for j in 0..shape[1] {

            let array1: Array1<f32> = arr.slice(s![i, j, ..]).to_owned();

            let max_index = nucleus_sampling(&array1, 0.8);

            max_indices[(i, j)] = max_index;
        }
    }

    //println!("hiddenstate: {:?}", max_indices);

    max_indices

}
//________________________________________________________________
pub fn multinomial(probabilities: &Array1<f32>) -> usize {
    let mut rng = rand::thread_rng();
    let mut cdf = Array1::zeros(probabilities.len());
    
    // Compute the Cumulative Distribution Function (CDF)
    cdf[0] = probabilities[0];
    for i in 1..probabilities.len() {
        cdf[i] = cdf[i - 1] + probabilities[i];
    }

    // Sample a random float between 0 and 1
    let sample = rng.gen::<f32>();

    // Find the index where the sample falls under in the CDF
    let mut sampled_idx = 0;
    for (idx, &prob) in cdf.iter().enumerate() {
        if sample < prob {
            //println!("prob: {:?}, sample {:?}, idx {:?}", prob, sample, idx);
            sampled_idx = idx;
            break;
        }
    }
    
    sampled_idx
}
//________________________________________________________________
pub fn nucleus_sampling(probabilities: &Array1<f32>, p: f32) -> usize {
    let mut rng = rand::thread_rng();

    // Sort probabilities and keep track of original indices
    let mut sorted_probs_with_indices: Vec<(f32, usize)> = probabilities.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
    sorted_probs_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute the CDF (Cumulative Distribution Function)
    let mut cdf = 0.0;
    let mut last_idx_to_consider = 0;

    for &(prob, _) in &sorted_probs_with_indices {
        cdf += prob;
        if cdf > p {
            break;
        }
        last_idx_to_consider += 1;
    }

    // Slice to get the nucleus
    let nucleus: Vec<usize> = sorted_probs_with_indices[0..=last_idx_to_consider]
        .iter()
        .map(|&(_, idx)| idx)
        .collect();

    // Sample from the nucleus
    let sampled_idx = nucleus[rng.gen_range(0..=last_idx_to_consider)];

    sampled_idx
}
//________________________________________________________________
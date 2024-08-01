
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3};
use ndarray::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

use std::path::Path;
use std::sync::Arc;
use crate::ml::util::*;

//________________________________________________________________
pub fn load_data1() -> Array2<String> {
    let padding = String::from("<pad>");
    // Path to the file
    let path = Path::new("data\\text.txt");

    // Open the file
    let file = File::open(&path);

    // Create a buffered reader
    let reader = io::BufReader::new(file.unwrap());

    // Temporary vector to store read lines
    let mut full_data: Vec<Array1<String>> = Vec::new();
    let max_seq_size = 70;
    // Read the file line by line
    let mut cols: usize = 0;
    for line in reader.lines() {
        cols += 1;
        let line = line;
        let line = Arc::new(line.expect("IM FINISHED"));

        let mut words: Vec<String> = (*line).clone()
        .split_whitespace()
        .map(|word| word.to_string())
        .collect();

        while words.len() < max_seq_size {
            words.push(padding.clone());
        }

        let arr: Array1<String> = Array::from(words);

        full_data.push(arr);
    }

    let mut flat_data = Vec::new();

    for array in full_data.iter() {
        flat_data.extend(array.iter().cloned());
    }

    let array2: Array2<String> = Array2::from_shape_vec((cols, max_seq_size), flat_data).unwrap();

    println!("ARRY: {:?}", array2);

    array2
}
//________________________________________________________________
pub fn load_data_label_pairs(max: usize, limit: usize) -> (Array2<String>,Array2<String>) {
    let padding = String::from("<pad>");
    let start = String::from("<start>");
    let end = String::from("<end>");
    // Path to the file
    let path = Path::new("data\\text.txt");

    // Open the file
    let file = File::open(&path);

    // Create a buffered reader
    let reader = io::BufReader::new(file.unwrap());

    // Temporary vector to store read lines
    let mut full_data: Vec<Array1<String>> = Vec::new();
    let mut full_labs: Vec<Array1<String>> = Vec::new();

    // Read the file line by line
    let mut cols: usize = 0;
    let mut counter: usize = 0;

    for line in reader.lines() {
        if counter == limit {
            break;
        }
        counter += 1;
        let line = line;
        let line = Arc::new(line.expect("IM FINISHED"));

        let mut words: Vec<String> = (*line).clone()
        .split_whitespace()
        .map(|word| word.to_string())
        .collect();

        words.insert(0, start.clone());

        while words.len() < max {
            words.push(padding.clone());
        }

        let mut labels: Vec<String> = (*line).clone()
        .split_whitespace()
        .map(|word| word.to_string())
        .collect();

        labels.push(end.clone());

        while labels.len() < max {
            labels.push(padding.clone());
        }

        let range: usize = words.len() - max;

        for i in 0..range {  
            cols += 1;
            let arr1: Array1<String> = Array::from(words[i..max + i].to_vec());
            full_data.push(arr1);
            let arr2: Array1<String> = Array::from(labels[i..max + i].to_vec());
            full_labs.push(arr2);

        }
    }
    

    let mut flat_data: Vec<String> = Vec::new();
    let mut flat_labels: Vec<String> = Vec::new();


    for (array1, array2) in full_data.iter().zip(full_labs.iter()) {
        flat_data.extend(array1.iter().cloned());
        flat_labels.extend(array2.iter().cloned());
    }

    let data_array2: Array2<String> = Array2::from_shape_vec((cols, max), flat_data).unwrap();
    let label_array2: Array2<String> = Array2::from_shape_vec((cols, max), flat_labels).unwrap();

    (data_array2, label_array2)
}
//________________________________________________________________
pub fn create_data_label_indices(file: &str, num_examples: usize) -> (Array2<i32>, Array3<f32>) {

    let (data_points, data_labels) = load_data_label_pairs(15,num_examples);
    //println!("{:?}", data_points);
    //println!("{:?}", data_labels);
    let (vocabulary, _) = load_embeddings(file);

    let (batch, max_seq) = data_points.dim(); 
    let mut temp_data: Vec<i32> = Vec::new();
    let mut temp_labs: Vec<Array1<f32>> = Vec::new();

    let mut my_map: HashMap<String, i32> = HashMap::new();
    let vec: Vec<i32> = (0..=(vocabulary.len()-1) as i32).collect();
    let arr: Array1<i32> = Array::from_shape_vec(vocabulary.len(), vec).unwrap();
    let vocab_size = vocabulary.len();

    for (key, value) in vocabulary.iter().zip(arr.iter()) {
        my_map.insert(key.to_owned(), value.to_owned());
    }

    for (array1, array2) in data_points.axis_iter(Axis(0))
        .zip(data_labels.axis_iter(Axis(0))) {
        for (word1, word2) in array1.iter().zip(array2.iter()) {
            //println!("{:?}", word);
            temp_data.push(*my_map.get(&word1.clone()).expect("Cannot find element in hashmap!"));
            let mut temp: Array1<f32> = Array1::zeros(vocab_size);
            temp[*my_map.get(&word2.clone()).expect("Cannot find element in hashmap!") as usize] = 1.0;
            temp_labs.push(temp);

        }
    }

    //println!("LABELS: {:?}",temp_labs);

    let mut flat_data = Vec::new();

    for array in temp_labs.iter() {
        flat_data.extend(array.iter().cloned());
    }

    let data_arr: Array2<i32> = Array2::from_shape_vec((batch, max_seq), temp_data).unwrap();
    let labs_arr: Array3<f32> = Array3::from_shape_vec((batch, max_seq, vocab_size), flat_data).unwrap();

    (data_arr, labs_arr)
}
//________________________________________________________________
pub fn create_data_label_indices2(file: &str, d_model: usize) -> (Array3<f32>, Array3<f32>) {

    let (data_points, data_labels) = load_data_label_pairs(15, 3);
    //println!("{:?}", data_points);
    //println!("{:?}", data_labels);
    let (vocabulary, _) = load_embeddings(file);

    let (batch, max_seq) = data_points.dim(); 
    let mut temp_data: Vec<Array1<f32>> = Vec::new();
    let mut temp_labs: Vec<Array1<f32>> = Vec::new();

    let mut my_map: HashMap<String, i32> = HashMap::new();
    let vec: Vec<i32> = (0..=(vocabulary.len()-1) as i32).collect();
    let arr: Array1<i32> = Array::from_shape_vec(vocabulary.len(), vec).unwrap();
    let vocab_size = vocabulary.len();

    for (key, value) in vocabulary.iter().zip(arr.iter()) {
        my_map.insert(key.to_owned(), value.to_owned());
    }

    for (array1, array2) in data_points.axis_iter(Axis(0))
        .zip(data_labels.axis_iter(Axis(0))) {
        for (word1, word2) in array1.iter().zip(array2.iter()) {
            //println!("{:?}", word);
            let mut temp: Array1<f32> = Array1::zeros(d_model);
            let num = *my_map.get(&word1.clone()).expect("Cannot find element in hashmap!") as f32;
            temp = temp.mapv(|val| val + num);
            temp_data.push(temp);
            let mut temp: Array1<f32> = Array1::zeros(vocab_size);
            temp[*my_map.get(&word2.clone()).expect("Cannot find element in hashmap!") as usize] = 1.0;
            temp_labs.push(temp);


        }
    }

    let mut flat_data = Vec::new();

    for array in temp_labs.iter() {
        flat_data.extend(array.iter().cloned());
    }

    let labs_arr: Array3<f32> = Array3::from_shape_vec((batch, max_seq, vocab_size), flat_data).unwrap();

    let mut flat_data = Vec::new();

    for array in temp_data.iter() {
        flat_data.extend(array.iter().cloned());
    }

    let data_arr: Array3<f32> = Array3::from_shape_vec((batch, max_seq, d_model), flat_data).unwrap();

    (data_arr, labs_arr)
}
//________________________________________________________________
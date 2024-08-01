
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3};
use ndarray::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, BufReader, BufRead, Result as IoResult};

mod ml;
use crate::ml::types::*;
use crate::ml::util::*;
use crate::ml::data_util::create_data_label_indices;

//________________________________________________________________
struct Transformer {
    decoder_embeddings: ml::transformers::Embeddings,
    decoder: ml::transformers::Decoder,
    parameters: HashMap<String, f32>,
    d_model: usize,
}

impl Transformer {
    pub fn new(emb_file: &str, num_layers: usize, d_model: usize, num_heads: usize, d_ffn: usize, vocab_size: usize, drop_prob: f32, optimiser: f32, optim_lr1: f32, optim_lr2: f32, lr: f32) -> Self {

        let decoder = ml::transformers::Decoder::new(num_layers, d_model, num_heads, d_ffn, vocab_size, drop_prob);
        let mut parameters = HashMap::new();
        let decoder_embeddings = ml::transformers::Embeddings::new(emb_file);

        parameters.insert(String::from("optimiser"), optimiser);
        parameters.insert(String::from("optim_lr1"), optim_lr1);
        parameters.insert(String::from("optim_lr2"), optim_lr2);
        parameters.insert(String::from("lr"), lr);
        
        Self {
            decoder_embeddings,
            decoder,
            parameters,
            d_model,
        }
    }

    // sometimes self has to be muttable
    pub fn forward(&mut self, training: &bool, input: &Array2<i32>) -> Array3<f32>{

        let output_emb = self.decoder_embeddings.get_vector_array(input, self.d_model);
        let output = self.decoder.forward(training,&output_emb);

        output
    }

    pub fn backward(&mut self, d_out: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32>{
        let gradient = self.decoder.backward(d_out, &Some(&self.parameters));

        gradient
    }

    pub fn save_parameters_to_file(&mut self, file_path: &str) -> IoResult<()> {
        
        let mut full_path = "models//".to_string();
        full_path.push_str(file_path);
        full_path.push_str(".txt");
        let mut file = File::create(full_path)?;
        let mut file_lines: Vec<String> = Vec::new();

        self.decoder.save_parameters_to_file(&mut file_lines);

        for line in file_lines.iter(){
            writeln!(file, "{}", line)?;
        }
        Ok(())

    }

    pub fn load_parameters_from_file(file_path: &str, emb_file: &str, num_layers: usize, d_model: usize, num_heads: usize, d_ffn: usize, vocab_size: usize, drop_prob: f32, optimiser: f32, optim_lr1: f32, optim_lr2: f32, lr: f32) -> Self {
        let mut decoder = ml::transformers::Decoder::new(num_layers, d_model, num_heads, d_ffn, vocab_size, drop_prob);
        //decoder.linear_classifier.load_parameters_from_file(file_path).expect("msg");
        let mut parameters = HashMap::new();
        let decoder_embeddings = ml::transformers::Embeddings::new(emb_file);

        parameters.insert(String::from("optimiser"), optimiser);
        parameters.insert(String::from("optim_lr1"), optim_lr1);
        parameters.insert(String::from("optim_lr2"), optim_lr2);
        parameters.insert(String::from("lr"), lr);

        let mut full_path = "models//".to_string();
        full_path.push_str(file_path);
        full_path.push_str(".txt");
        let file = File::open(full_path).expect("cannot load file");
        let reader = BufReader::new(file);

        let mut file_lines: Vec<String> = reader.lines()
            .filter_map(|result| result.ok())  // Filter out any errors
            .collect();

        decoder.load_parameters_from_file(&mut file_lines);

        Self {
            decoder_embeddings,
            decoder,
            parameters,
            d_model,
        }

    }
    
}
//________________________________________________________________
fn generate( model: &mut Transformer, mut start_token: Vec<i32>, max_seq: usize, num: usize){
    
    let padding = 1;

    while start_token.len() < max_seq {
        start_token.push(padding);
    }

    for i in 0..max_seq-1{

        //println!("START TOKEN: {:?}", start_token);

        let mut data_arr: Array2<i32> = Array2::from_shape_vec((1, max_seq), start_token.clone()).unwrap();
        let result = model.forward(&false, &data_arr);
        //println!("PREDS: {:?}", result);
        //println!("SUMMM: {:?}", result.sum_axis(Axis(2)));
        
        let max_vals = argmax(result, Axis(2));
        let prediction = max_vals[(0, max_seq-1)];
        //println!("PREDICTION: {:?}", prediction);

        start_token.remove(0);
        start_token.push(prediction as i32);
        

    }

    let mut data_arr: Array2<i32> = Array2::from_shape_vec((1, max_seq), start_token.clone()).unwrap();
    
    let sentence = model.decoder_embeddings.get_string_array(&data_arr);
    let arr_str = format!("{}", sentence);
    println!("\n\nSENTENCE: {:?}\n", arr_str);
    
}
//________________________________________________________________
fn generate2( model: &mut Transformer, mut start_token: Vec<i32>, max_seq: usize, num: usize){
    
    let padding = 1;

    while start_token.len() < max_seq {
        start_token.push(padding);
    }

    let mut _sentence: Vec<i32> = start_token.clone();
    _sentence.push(0);
    
    
    for i in 0..num{

        //println!("START TOKEN: {:?}", start_token);

        let mut data_arr: Array2<i32> = Array2::from_shape_vec((1, max_seq), start_token.clone()).unwrap();
        let result = model.forward(&false, &data_arr);
        //println!("PREDS: {:?}", result);
        //println!("SUMMM: {:?}", result.sum_axis(Axis(2)));
        
        let max_vals = argmax2(result, Axis(2));
        let prediction = max_vals[(0, max_seq-1)];
        //println!("PREDICTION: {:?}", prediction);

        start_token.remove(0);
        start_token.push(prediction as i32);
        _sentence.push(prediction as i32);

        

    }

    let data_arr: Array2<i32> = Array2::from_shape_vec((1, _sentence.len()), _sentence.clone()).unwrap();
    
    let sentence = model.decoder_embeddings.get_string_array(&data_arr);
    let arr_str = format!("{}", sentence);
    println!("\n\nSENTENCE222: {:?}\n", arr_str);
    
}
//________________________________________________________________
fn generate3( model: &mut Transformer, mut start_token: Vec<i32>, max_seq: usize, num: usize){
    
    let padding = 1;

    while start_token.len() < max_seq {
        start_token.push(padding);
    }

    let mut _sentence: Vec<i32> = start_token.clone();
    _sentence.push(0);
    
    for i in 0..num{

        //println!("START TOKEN: {:?}", start_token);

        let mut data_arr: Array2<i32> = Array2::from_shape_vec((1, max_seq), start_token.clone()).unwrap();
        let result = model.forward(&false, &data_arr);
        //println!("PREDS: {:?}", result);
        //println!("SUMMM: {:?}", result.sum_axis(Axis(2)));
        
        let max_vals = argmax3(result, Axis(2));
        let prediction = max_vals[(0, max_seq-1)];
        //println!("PREDICTION: {:?}", prediction);

        start_token.remove(0);
        start_token.push(prediction as i32);
        _sentence.push(prediction as i32);

    }

    let data_arr: Array2<i32> = Array2::from_shape_vec((1, _sentence.len()), _sentence.clone()).unwrap();
    
    let sentence = model.decoder_embeddings.get_string_array(&data_arr);
    let arr_str = format!("{}", sentence);
    println!("\n\nSENTENCE333: {:?}\n", arr_str);
    
}
//________________________________________________________________
//________________________________________________________________
//________________________________________________________________
fn main() {
     
    let num_layers = 1;
    let d_model = 24;
    let d_ffn = 240;
    let num_heads = 3;
    let vocab_size = 419;
    let drop_prob = 0.1;
    let optimiser = 3.0;
    let optim_lr1 = 0.9;
    let optim_lr2 = 0.999;
    let lr = 5e-3;
    let total_itr = 1;
    let num_examples = 15;

    let file = "embeddings7";
    let save_file = "transformer_2_layers";
    let load = true;

    let mut loss = ml::functional::CrossEntropyLoss::new();
    let mut model:Transformer;

    if load == false {
        model = Transformer::new(file, 
            num_layers, 
            d_model, 
            num_heads, 
            d_ffn, 
            vocab_size, 
            drop_prob,
            optimiser,
            optim_lr1, 
            optim_lr2,
            lr);
    }
    else{
        model = Transformer::load_parameters_from_file(save_file,
        file, 
        num_layers, 
        d_model, 
        num_heads, 
        d_ffn, 
        vocab_size, 
        drop_prob,
        optimiser,
        optim_lr1, 
        optim_lr2, 
        lr);

    }

    /*optimisers:
    0.0 GD,
    1.0 Momentum,
    2.0 RMS,
    3.0 Adam
    */

    let (data_batch, label_batch) = create_data_label_indices("embeddings7", num_examples);

    for i in 0..total_itr {

        let result = model.forward(&true, &data_batch);
        let error = loss.calculate(result, label_batch.clone());

        println!("[ Epoch: {epoch:0>3} ]  [ Loss:  {error:0>2.18} ]", error=error, epoch=i);
    
        model.backward(&loss.backward(), &None);
    }

    model.save_parameters_to_file(save_file);

    generate(&mut model, vec![0],15, 15);
    //tanjiros attack completely severs enmus neck causing the lower rank to unleash a terrible scream 

    let mut hash: HashMap<String, i32> = HashMap::new();

    let (vocabulary, _) = load_embeddings(file);

    let vec: Vec<i32> = (0..=(vocabulary.len()-1) as i32).collect();
    let vocab: Array1<i32> = Array::from_shape_vec(vocabulary.len(), vec).unwrap();

    for (key, value) in vocabulary.iter().zip(vocab.iter()) {
        hash.insert(key.to_owned(), value.to_owned());
    }

    let vector = vec!["<start>", "tanjiros", "attack", "completely", "severs", "enmus", "neck", "causing", "the", "lower", "rank", "to", "unleash", "a", "terrible"];
    let mut emb_vec: Vec<i32> = Vec::new();

    for elem in vector.iter(){
        emb_vec.push(*hash.get(&String::from(*elem)).unwrap());
    }
    let num = 10;
    generate(&mut model, emb_vec.clone(),15, 15);
    generate2(&mut model, emb_vec.clone(),15, num);

    //the piece of the demonic train that tanjiro and inosuke are on begins to fall sideways with the formers earlier

    let vector = vec!["<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<start>"];
    let mut emb_vec: Vec<i32> = Vec::new();

    for elem in vector.iter(){
        emb_vec.push(*hash.get(&String::from(*elem)).unwrap());
    }

    let num = 50;

    generate(&mut model, emb_vec.clone(),15, 15);
    generate2(&mut model, emb_vec.clone(),15, num);

}
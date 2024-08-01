
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3};
use ndarray::prelude::*;
use std::collections::HashMap;

use crate::ml::types::Layer;
use crate::ml::nn::*;
use crate::ml::util::*;


//________________________________________________________________
pub struct Embeddings {
    look_up_table: HashMap<i32, Array1<f32>>,
    vocabulary: Array1<String>,
}

impl Embeddings {
    pub fn new(file: &str) -> Self {

    let mut my_map: HashMap<i32, Array1<f32>> = HashMap::new();
    let (vocabulary, vectors) = load_embeddings(file);

    let vec: Vec<i32> = (0..=(vocabulary.len()-1) as i32).collect();
    let vocab: Array1<i32> = Array::from_shape_vec(vocabulary.len(), vec).unwrap();

    for (key, value) in vocab.iter().zip(vectors.axis_iter(Axis(0))) {
        my_map.insert(key.to_owned(), value.to_owned());
    }

        Self { look_up_table: my_map, vocabulary}
    }

    pub fn get_vector_from_token(&self, token: &i32) -> Array1<f32> {
        self.look_up_table.get(token).expect("Cannot find element in hashmap!").clone()
    }

    pub fn get_string_from_token(&self, token: &i32) -> String {
        self.vocabulary[*token as usize].clone()
    }

    pub fn get_string_array(&self, token_array: &Array2<i32>) -> Array2<String> {
        let (batch, max_seq) = token_array.dim(); 
        let mut temp: Vec<String> = Vec::new();

        for array in token_array.axis_iter(Axis(0)){
            for token in array.iter() {
                temp.push(self.get_string_from_token(token));
            }
        }
    
        let array2: Array2<String> = Array2::from_shape_vec((batch, max_seq), temp).unwrap();
        array2
    }

    pub fn get_vector_array(&self, token_array: &Array2<i32>, d_model: usize) -> Array3<f32> {
        
        let (batch, max_seq) = token_array.dim(); 
        let mut full_data: Array3<f32>;
        let mut temp: Vec<Array1<f32>> = Vec::new();
        for array in token_array.axis_iter(Axis(0)){
            for token in array.iter() {
                //println!("{:?}", word);
                temp.push(self.get_vector_from_token(token));
            }
        }

        let mut flat_data = Vec::new();

        for array in temp.iter() {
            flat_data.extend(array.iter().cloned());
        }
    
        let array2: Array3<f32> = Array3::from_shape_vec((batch, max_seq, d_model), flat_data).unwrap();
        array2
    }
}
//________________________________________________________________
pub struct PositionalEncoding {

}

impl PositionalEncoding {
    pub fn new() -> Self{
        Self {  }
    }

    pub fn encode(&self, input_vector: &Array3<f32>) -> Array3<f32> {

        let (_, max_seq_length, d_model) = input_vector.dim();
        let indices: Array1<f32> = Array1::linspace(0.0, (d_model-1) as f32, d_model);

        let even_indices= indices.slice(s![0..;2]);
        let odd_indices= indices.slice(s![1..;2]);

        let positions: Array1<f32> = Array1::linspace(0.0, (max_seq_length-1) as f32, max_seq_length);
        let positions = positions.broadcast(((d_model/2) as usize, max_seq_length)).expect("cannot broacast!");

        let even_exp = even_indices.mapv(|val| 2.0 * (val/d_model as f32));
        let even_exp = even_exp.broadcast((max_seq_length, (d_model/2) as usize)).expect("cannot broacast!");
        let pe_even = (&positions.t()/ &even_exp.mapv(|val| 10000f32.powf(val))).mapv(|val| val.sin());

        let odd_exp = odd_indices.mapv(|val| 2.0 * (val/d_model as f32));
        let odd_exp = odd_exp.broadcast((max_seq_length, (d_model/2) as usize)).expect("cannot broacast!");
        let pe_odd = (&positions.t()/ &odd_exp.mapv(|val| 10000f32.powf(val))).mapv(|val| val.sin());

        let mut full: Array2<f32> = Array2::zeros((max_seq_length, d_model));

        full.slice_mut(s![.., 0..;2]).assign(&pe_even);
        full.slice_mut(s![.., 1..;2]).assign(&pe_odd);

        let out = input_vector + &full;

        out
    }
}
//________________________________________________________________
pub struct PointwiseFeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    relu: ReLU,
}

impl PointwiseFeedForward {
    pub fn new(d_model: usize, hidden: usize, drop_prob: f32) -> Self {
        let linear1 = Linear::new(d_model, hidden);
        let linear2 = Linear::new(hidden, d_model);
        let relu = ReLU::new(true, None);
        let dropout = Dropout::new(drop_prob);

        Self { linear1, linear2, dropout, relu }
    }
}

impl Layer for PointwiseFeedForward {
    fn forward(&mut self, training: &bool, input_vector: &Array3<f32>) -> Array3<f32> {
        // First Linear Layer
        let output = self.linear1.forward(training, input_vector);
        
        // ReLU activation
        let output = self.relu.forward(training, &output);
        
        // Dropout
        let output = self.dropout.forward(training, &output);
        
        // Second Linear Layer
        let output = self.linear2.forward(training, &output);
        
        output
    }

    fn backward(&mut self, gradient: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        // Backprop through second Linear Layer
        let d_output = self.linear2.backward(gradient, param);
        
        // Backprop through Dropout
        let d_output = self.dropout.backward(&d_output, param);
        
        // Backprop through ReLU 
        let d_output = self.relu.backward(&d_output, param);
        
        // Backprop through first Linear Layer
        let d_output = self.linear1.backward(&d_output, param);
        
        d_output
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        self.linear1.save_parameters_to_file(file_lines);
        self.linear2.save_parameters_to_file(file_lines);
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        self.linear1.load_parameters_from_file(file_lines);
        self.linear2.load_parameters_from_file(file_lines);
    }
}
//________________________________________________________________
pub struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    qkv_layer: Linear,
    linear_layer: Linear,
    softmax_layer: Softmax,
    queries: Option<Array4<f32>>,
    keys: Option<Array4<f32>>,
    values: Option<Array4<f32>>,
    scores: Option<Array4<f32>>,
    //mask: Option<Array4<f32>>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let head_dim = d_model / num_heads;
        let qkv_layer = Linear::new(d_model, 3 * d_model);
        let linear_layer = Linear::new(d_model, d_model);
        let softmax_layer = Softmax::new();
        //EXAMPLE: batch size = 30, max_seq = 10, d_model = 24, num_heads = 8, head_dim = 3
        
        Self {
            d_model,
            num_heads,
            head_dim,
            qkv_layer,
            linear_layer,
            softmax_layer,
            queries: None,
            keys: None,
            values: None,
            scores: None,
            //mask: None,
        }
    }
    
    fn scaled_dot_product(&mut self, q: Array4<f32>, k: Array4<f32>, v: Array4<f32>, mask: Option<Array4<f32>>) -> (Array4<f32>, Array4<f32>) {
        let d_k = self.head_dim as f32;

        self.queries = Some(q.clone());
        self.keys = Some(k.clone());
        self.values = Some(v.clone());

        //30 x 8 x 10 x 10
        let mut scaled = dot4(&q, &k.to_owned().permuted_axes([0,1,3,2])) / f32::sqrt(d_k);

        if let Some(m) = mask {
            scaled = scaled + m;
        }

        // Convert to ArrayD before passing to softmax
        let scaled_d: ArrayD<f32> = scaled.clone().into_dyn();
        let attention_d = self.softmax_layer.forward(&scaled_d);

        // Convert back to Array4
        let attention: Array4<f32> = attention_d.into_dimensionality().expect("Error converting ArrayD to Array4");

        //30 x 8 x 10 x 10
        self.scores = Some(attention.clone());

        let values = dot4(self.scores.as_ref().unwrap(), &v);

        //30 x 8 x 10 x 3
        (values, attention)
    }

    fn look_ahead_mask(&self, batch_size: usize, num_heads: usize, seq_len: usize) -> Array4<f32> {
        // Create mask of shape [batch_size, num_heads, seq_len, seq_len]

        // Create an upper triangular matrix for a single batch
        let mut single_batch_mask = Array3::zeros((num_heads, seq_len, seq_len));
        for i in 0..seq_len {
            for j in (i+1)..seq_len {
                single_batch_mask.slice_mut(s![.., i, j]).fill(f32::NEG_INFINITY);
            }
        }

        // Replicate the mask for all batches
        let mask = Array4::from_shape_fn((batch_size, num_heads, seq_len, seq_len), |(_, h, i, j)| {
            single_batch_mask[(h, i, j)]
        });

        mask

    }
}

impl Layer for MultiHeadAttention {
    fn forward(&mut self, training: &bool, input_vector: &Array3<f32>) -> Array3<f32> {

        let (batch_size, sequence_length, d_model) = input_vector.dim();

        // Create the look-ahead mask
        let mask = self.look_ahead_mask(batch_size, self.num_heads, sequence_length);
        
        //30 x 10 x 72
        let qkv = self.qkv_layer.forward(training, input_vector);
        
        //30 x 10 x 8 x 9
        let qkv = qkv.into_shape((batch_size, sequence_length, self.num_heads, 3 * self.head_dim)).unwrap();
        
        //30 x 8 x 10 x 9
        let qkv = qkv.permuted_axes([0, 2, 1, 3]);
        
        //30 x 8 x 10 x 3
        let chunks: Vec<_> = qkv.axis_chunks_iter(Axis(3), self.head_dim).collect();
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        //30 x 8 x 10 x 3 
        //could change to take in refernces instead?
        let (values, _) = self.scaled_dot_product(q.to_owned(), k.to_owned(), v.to_owned(), Some(mask));

        //30 x 10 x 8 x 3

        let values_perm = values.permuted_axes([0, 2, 1, 3]);
        let values_shaped = values_perm.as_standard_layout().to_owned();

        //30 x 10 x 24
        let output = values_shaped.into_shape((batch_size, sequence_length, d_model)).unwrap();

        let out = self.linear_layer.forward(training, &output);

        out
    }

    fn backward(&mut self, gradient: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        let (batch_size, sequence_length, _) = gradient.dim();
        let scores = self.scores.to_owned().unwrap();
        let q = self.queries.to_owned().unwrap();
        let k = self.keys.to_owned().unwrap();
        let v = self.values.to_owned().unwrap();

        //30 x 10 x 24
        let gradient = self.linear_layer.backward(gradient, param);
        
        // Reshape and permute gradient to match forward computation
        //30 x 10 x 8 x 3
        let gradient = gradient.into_shape((batch_size, sequence_length, self.num_heads, self.head_dim)).unwrap();

        //30 x 8 x 10 x 3
        let gradient = gradient.permuted_axes([0, 2, 1, 3]);
        let gradient = gradient.as_standard_layout().to_owned();

        // Derivative w.r.t. V
        //30 x 8 x 10 x 3
        //let d_v = self.scores.t().dot(&gradient);
        let d_v = dot4(&scores.permuted_axes([0,1,3,2]), &gradient);
        
        // Derivative w.r.t. Scores
        //30 x 8 x 10 x 10
        //gradient = gradient.dot(&self.values.t());
        let gradient = dot4(&gradient, &v.permuted_axes([0,1,3,2]));

        // Derivative w.r.t. Softmax outputs
        //30 x 8 x 10 x 10
        let gradient_d: ArrayD<f32> = gradient.clone().into_dyn();
    
        let gradient_d = self.softmax_layer.backward(&gradient_d, param);
        let gradient: Array4<f32> = gradient_d.into_dimensionality().unwrap();
    
        // Derivative w.r.t. Q
        //30 x 8 x 10 x 3
        //let d_q = gradient.dot(&self.keys) / f32::sqrt(self.head_dim as f32);
        let d_q = dot4(&gradient, &k) / f32::sqrt(self.head_dim as f32);
        
        // Derivative w.r.t. K
        //30 x 8 x 10 x 3
        //let d_k = gradient.t().dot(&self.queries) / f32::sqrt(self.head_dim as f32);
        let d_k = dot4(&gradient.permuted_axes([0,1,3,2]), &q) / f32::sqrt(self.head_dim as f32);
        
        // Concatenate gradients for Q, K, V
        //30 x 8 x 10 x 9
        let d_qkv = ndarray::concatenate(Axis(3), &[d_q.view(), d_k.view(), d_v.view()]).unwrap();

        //30 x 10 x 8 x 9
        let d_qkv = d_qkv.permuted_axes([0, 2, 1, 3]);
        let d_qkv = d_qkv.as_standard_layout().to_owned();

        //30 x 10 x 72
        let d_qkv = d_qkv.into_shape((batch_size, sequence_length, 3 * self.d_model)).unwrap();

        // Backward pass through qkv linear layer
        let gradient = self.qkv_layer.backward(&d_qkv, param);

        gradient
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        self.qkv_layer.save_parameters_to_file(file_lines);
        self.linear_layer.save_parameters_to_file(file_lines);
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        self.qkv_layer.load_parameters_from_file(file_lines);
        self.linear_layer.load_parameters_from_file(file_lines);
    }
}
//________________________________________________________________
pub struct MultiHeadAttentionQKV {
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_layer: Linear,
    softmax_layer: Softmax,
    queries: Option<Array4<f32>>,
    keys: Option<Array4<f32>>,
    values: Option<Array4<f32>>,
    scores: Option<Array4<f32>>,
    //mask: Option<Array4<f32>>,
}

impl MultiHeadAttentionQKV {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let head_dim = d_model / num_heads;
        let linear_layer = Linear::new(d_model, d_model);
        let linear_q = Linear::new(d_model, d_model);
        let linear_k = Linear::new(d_model, d_model);
        let linear_v = Linear::new(d_model, d_model);
        let softmax_layer = Softmax::new();
        //EXAMPLE: batch size = 30, max_seq = 10, d_model = 24, num_heads = 8, head_dim = 3
        
        Self {
            d_model,
            num_heads,
            head_dim,
            linear_q,
            linear_k,
            linear_v,
            linear_layer,
            softmax_layer,
            queries: None,
            keys: None,
            values: None,
            scores: None,
            //mask: None,
        }
    }
    
    fn scaled_dot_product(&mut self, q: Array4<f32>, k: Array4<f32>, v: Array4<f32>, mask: Option<Array4<f32>>) -> (Array4<f32>, Array4<f32>) {
        let d_k = self.head_dim as f32;

        self.queries = Some(q.clone());
        self.keys = Some(k.clone());
        self.values = Some(v.clone());

        //30 x 8 x 10 x 10
        let mut scaled = dot4(&q, &k.to_owned().permuted_axes([0,1,3,2])) / f32::sqrt(d_k);

        if let Some(m) = mask {
            scaled = scaled + m;
        }

        // Convert to ArrayD before passing to softmax
        let scaled_d: ArrayD<f32> = scaled.clone().into_dyn();
        let attention_d = self.softmax_layer.forward(&scaled_d);

        // Convert back to Array4
        let attention: Array4<f32> = attention_d.into_dimensionality().expect("Error converting ArrayD to Array4");

        //30 x 8 x 10 x 10
        self.scores = Some(attention.clone());

        let values = dot4(self.scores.as_ref().unwrap(), &v);

        //30 x 8 x 10 x 3
        (values, attention)
    }

    fn look_ahead_mask(&self, batch_size: usize, num_heads: usize, seq_len: usize) -> Array4<f32> {
        // Create mask of shape [batch_size, num_heads, seq_len, seq_len]

        // Create an upper triangular matrix for a single batch
        let mut single_batch_mask = Array3::zeros((num_heads, seq_len, seq_len));
        for i in 0..seq_len {
            for j in (i+1)..seq_len {
                single_batch_mask.slice_mut(s![.., i, j]).fill(f32::NEG_INFINITY);
            }
        }

        // Replicate the mask for all batches
        let mask = Array4::from_shape_fn((batch_size, num_heads, seq_len, seq_len), |(_, h, i, j)| {
            single_batch_mask[(h, i, j)]
        });

        mask

    }
}

impl Layer for MultiHeadAttentionQKV {
    fn forward(&mut self, training: &bool, input_vector: &Array3<f32>) -> Array3<f32> {

        let (batch_size, sequence_length, d_model) = input_vector.dim();

        // Create the look-ahead mask
        let mask = self.look_ahead_mask(batch_size, self.num_heads, sequence_length);
        
        //30 x 10 x 24
        let q = self.linear_q.forward(training, input_vector);
        let k = self.linear_k.forward(training, input_vector);
        let v = self.linear_v.forward(training, input_vector);
        
        //30 x 10 x 8 x 3
        let q = q.into_shape((batch_size, sequence_length, self.num_heads, self.head_dim)).unwrap();
        let k = k.into_shape((batch_size, sequence_length, self.num_heads, self.head_dim)).unwrap();
        let v = v.into_shape((batch_size, sequence_length, self.num_heads, self.head_dim)).unwrap();

        //30 x 8 x 10 x 3
        let q = q.permuted_axes([0, 2, 1, 3]);
        let k = k.permuted_axes([0, 2, 1, 3]);
        let v = v.permuted_axes([0, 2, 1, 3]);

        //30 x 8 x 10 x 3 
        //could change to take in refernces instead?
        let (values, _) = self.scaled_dot_product(q.to_owned(), k.to_owned(), v.to_owned(), Some(mask));

        //30 x 10 x 8 x 3
        let values_perm = values.permuted_axes([0, 2, 1, 3]);
        let values_shaped = values_perm.as_standard_layout().to_owned();

        //30 x 10 x 24
        let output = values_shaped.into_shape((batch_size, sequence_length, d_model)).unwrap();

        let out = self.linear_layer.forward(training, &output);

        out
    }

    fn backward(&mut self, gradient: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        let (batch_size, sequence_length, _) = gradient.dim();
        let scores = self.scores.to_owned().unwrap();
        let q = self.queries.to_owned().unwrap();
        let k = self.keys.to_owned().unwrap();
        let v = self.values.to_owned().unwrap();

        //30 x 10 x 24
        let gradient = self.linear_layer.backward(gradient, param);
        
        // Reshape and permute gradient to match forward computation
        //30 x 10 x 8 x 3
        let gradient = gradient.into_shape((batch_size, sequence_length, self.num_heads, self.head_dim)).unwrap();

        //30 x 8 x 10 x 3
        let gradient = gradient.permuted_axes([0, 2, 1, 3]);
        let gradient = gradient.as_standard_layout().to_owned();

        // Derivative w.r.t. V
        //30 x 8 x 10 x 3
        //let d_v = self.scores.t().dot(&gradient);
        let d_v = dot4(&scores.permuted_axes([0,1,3,2]), &gradient);
        
        // Derivative w.r.t. Scores
        //30 x 8 x 10 x 10
        //gradient = gradient.dot(&self.values.t());
        let gradient = dot4(&gradient, &v.permuted_axes([0,1,3,2]));

        // Derivative w.r.t. Softmax outputs
        //30 x 8 x 10 x 10
        let gradient_d: ArrayD<f32> = gradient.clone().into_dyn();
    
        let gradient_d = self.softmax_layer.backward(&gradient_d, param);
        let gradient: Array4<f32> = gradient_d.into_dimensionality().unwrap();
    
        // Derivative w.r.t. Q
        //30 x 8 x 10 x 3
        //let d_q = gradient.dot(&self.keys) / f32::sqrt(self.head_dim as f32);
        let d_q = dot4(&gradient, &k) / f32::sqrt(self.head_dim as f32);
        
        // Derivative w.r.t. K
        //30 x 8 x 10 x 3
        //let d_k = gradient.t().dot(&self.queries) / f32::sqrt(self.head_dim as f32);
        let d_k = dot4(&gradient.permuted_axes([0,1,3,2]), &q) / f32::sqrt(self.head_dim as f32);
        
        // Concatenate gradients for Q, K, V

        //30 x 10 x 8 x 3
        let d_q = d_q.permuted_axes([0, 2, 1, 3]);
        let d_k = d_k.permuted_axes([0, 2, 1, 3]);
        let d_v = d_v.permuted_axes([0, 2, 1, 3]);

        let d_q = d_q.as_standard_layout().to_owned();
        let d_k = d_k.as_standard_layout().to_owned();
        let d_v = d_v.as_standard_layout().to_owned();

        //30 x 10 x 24
        let d_q = d_q.into_shape((batch_size, sequence_length, self.d_model)).unwrap();
        let d_k = d_k.into_shape((batch_size, sequence_length, self.d_model)).unwrap();
        let d_v = d_v.into_shape((batch_size, sequence_length, self.d_model)).unwrap();

        // Backward pass through qkv linear layer
        let grad_q = self.linear_q.backward(&d_q, param);
        let grad_k = self.linear_k.backward(&d_k, param);
        let grad_v = self.linear_v.backward(&d_v, param);

        let gradient = grad_q + grad_k + grad_v;

        gradient
    }
    
    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        self.linear_q.save_parameters_to_file(file_lines);
        self.linear_k.save_parameters_to_file(file_lines);
        self.linear_v.save_parameters_to_file(file_lines);
        self.linear_layer.save_parameters_to_file(file_lines);
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        self.linear_q.load_parameters_from_file(file_lines);
        self.linear_k.load_parameters_from_file(file_lines);
        self.linear_v.load_parameters_from_file(file_lines);
        self.linear_layer.load_parameters_from_file(file_lines);
    }
}
//________________________________________________________________
pub struct DecoderLayer {
    self_attention: MultiHeadAttention,
    dropout1: Dropout,
    norm1: LayerNorm,
    feed_forward_net: PointwiseFeedForward,
    dropout2: Dropout,
    norm2: LayerNorm,
}

impl DecoderLayer {
    
    pub fn new(d_model: usize, num_heads: usize, d_ffn: usize, drop_prob: f32) -> Self {
        let self_attention = MultiHeadAttention::new(d_model, num_heads);
        let dropout1 = Dropout::new(drop_prob);
        let norm1 = LayerNorm::new(d_model);
        let feed_forward_net = PointwiseFeedForward::new(d_model, d_ffn, drop_prob);
        let dropout2 = Dropout::new(drop_prob);
        let norm2 = LayerNorm::new(d_model);

        Self {
            self_attention,
            dropout1,
            norm1,
            feed_forward_net,
            dropout2,
            norm2,
        }
    }
}

impl Layer for DecoderLayer {
    // sometimes self has to be muttable
    fn forward(&mut self, training: &bool, input: &Array3<f32>) -> Array3<f32>{

        let residual = input.to_owned();
        let output = self.self_attention.forward(training, &residual);
        let output = self.dropout1.forward(training,&output);
        let output = self.norm1.forward(training,&(output + residual));
        //let output = self.norm1.forward(training,&output);

        //let residual = input.to_owned();
        //let output = self.feed_forward_net.forward(training,&residual);
        //let output = self.dropout2.forward(training,&output);
        //let output = self.norm2.forward(training,&(output + residual));
        //let output = self.norm2.forward(training,&output);


        output
    }

    fn backward(&mut self, d_out: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32> {

        //let gradient = self.norm2.backward(d_out, param);
        //let residual = gradient.to_owned();
        //let gradient = self.dropout2.backward(&residual, param);
        //let gradient = self.feed_forward_net.backward(&gradient, param);
        //let gradient = gradient + residual;

        let gradient = self.norm1.backward(d_out, param);
        let residual = gradient.to_owned();
        let gradient = self.dropout1.backward(&residual, param);
        let gradient = self.self_attention.backward(&gradient, param);
        let gradient = gradient + residual;

        gradient
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        self.self_attention.save_parameters_to_file(file_lines);
        self.norm1.save_parameters_to_file(file_lines);
        //self.feed_forward_net.save_parameters_to_file(file_lines);
        //self.norm2.save_parameters_to_file(file_lines);
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        self.self_attention.load_parameters_from_file(file_lines);
        self.norm1.load_parameters_from_file(file_lines);
        //self.feed_forward_net.load_parameters_from_file(file_lines);
        //self.norm2.load_parameters_from_file(file_lines);
    }
}
//________________________________________________________________
pub struct SequentialDecoder {
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialDecoder {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        SequentialDecoder { layers }
    }
}

impl Layer for SequentialDecoder {

    fn forward(&mut self, training: &bool, input: &Array3<f32>) -> Array3<f32> {

        let mut output = input.to_owned();
        
        for layer in self.layers.iter_mut() {
            output = layer.forward(training,&output);
        }
        
        output
    }

    fn backward(&mut self, d_out: &Array3<f32>,  param: &Option<&HashMap<String, f32>>) -> Array3<f32> {

        let mut gradient = d_out.to_owned();

        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient, param);
        }
        
        gradient
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        for layer in self.layers.iter_mut() {
            layer.save_parameters_to_file(file_lines);
        }
    }
    
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        for layer in self.layers.iter_mut() {
            layer.load_parameters_from_file(file_lines);
        }
    }
}
//________________________________________________________________
pub struct Decoder {
    positional_ecoding: PositionalEncoding,
    sequential_decoder: SequentialDecoder,
    linear_classifier: Linear,
    softmax: Softmax,
}

impl Decoder {
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ffn: usize, vocab_size: usize, drop_prob: f32) -> Self {

        let positional_ecoding = PositionalEncoding::new();
        
        let decoder_layers: Vec<Box<dyn Layer>> = (0..num_layers)
        .map(|_| Box::new(DecoderLayer::new(d_model, num_heads, d_ffn, drop_prob)) as Box<dyn Layer>)
        .collect();

        let sequential_decoder = SequentialDecoder::new(decoder_layers);
        let linear_classifier = Linear::new(d_model, vocab_size);
        let softmax = Softmax::new();

        Self {
            positional_ecoding,
            sequential_decoder,
            linear_classifier,
            softmax,
        }
    }
}

impl Layer for Decoder {
    // sometimes self has to be muttable
    fn forward(&mut self, training: &bool, input: &Array3<f32>) -> Array3<f32>{

        let output = self.positional_ecoding.encode(input);
        let mut output = self.sequential_decoder.forward(training,&output);
        output = self.linear_classifier.forward(training,&output);
        
        // Convert the output to ArrayD for the softmax operation
        let output_d: ArrayD<f32> = output.into_dyn();
        let softmax_output_d = self.softmax.forward(&output_d);
        
        // Convert the softmax output back to Array3
        let softmax_output = softmax_output_d.into_dimensionality::<Ix3>().expect("Expected 3D output from softmax");
        
        softmax_output
    }

    fn backward(&mut self, d_out: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32>{

        let d_out_d: ArrayD<f32> = d_out.clone().into_dyn();
    
        // Softmax backward pass with ArrayD
        let gradient_d = self.softmax.backward(&d_out_d, param);
    
        // Convert ArrayD back to Array3
        let mut gradient: Array3<f32> = match gradient_d.ndim() {
            3 => gradient_d.into_dimensionality::<Ix3>().unwrap(),
            _ => panic!("Unsupported dimensionality!"),
        };
    
        // Continue with the rest of the backward passes
        gradient = self.linear_classifier.backward(&gradient, param);
        gradient = self.sequential_decoder.backward(&gradient, param);
    
        gradient
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>) {
        self.sequential_decoder.save_parameters_to_file(file_lines);
        self.linear_classifier.save_parameters_to_file(file_lines);
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
        self.sequential_decoder.load_parameters_from_file(file_lines);
        self.linear_classifier.load_parameters_from_file(file_lines);
    }
}
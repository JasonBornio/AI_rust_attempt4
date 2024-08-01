
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufRead, Result as IoResult};

use crate::ml::types::*;
use crate::ml::optimisers::*;
use crate::ml::util::*;

//________________________________________________________________
pub struct Linear {
    input_size: usize,
    output_size: usize,
    weights: Array2<f32>,
    bias: Array1<f32>,
    input: Option<Array3<f32>>,
    optim_available: bool,
    optim: Optimiser,
    //norm: Option<String>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Array::random((input_size, output_size), Uniform::new(-0.05, 0.05));
        let bias = Array::zeros(output_size);
        let gd = GradientDescent::new();

        Self {
            input_size,
            output_size,
            weights: weights,
            bias: bias,
            input: None,
            optim_available: true,
            optim: Optimiser::GradientDescent(gd),
            //norm,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, training: &bool, input: &Array3<f32>) -> Array3<f32> {
        self.input = Some(input.to_owned());
        let _input = self.input.as_ref().unwrap();

        // ... normalise inputs as per self.norm
        let (batch_size, sequence_length, _) = _input.dim();

        let reshaped_input = _input.view().into_shape((batch_size * sequence_length, self.input_size)).unwrap();
    
        // Compute the dot product
        let weighted_sum = reshaped_input.dot(&self.weights);
        let bias_added = &weighted_sum + &self.bias;

        let output = bias_added.view().into_shape((batch_size, sequence_length, self.output_size)).unwrap().to_owned();

        output
    }

    fn backward(&mut self, d_out: &Array3<f32>,  param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        // ... initialise optimiser
        let param = param.as_ref().expect("Linear: parameters uninitialised!");
        if self.optim_available {
            if let Some(optim_type) = param.get("optimiser") {
                self.optim_available = false;
                let momentum = Momentum::new(*param.get("optim_lr1").unwrap());
                let rms: RMSprop = RMSprop::new(*param.get("optim_lr1").unwrap());
                let adam = Adam::new(*param.get("optim_lr1").unwrap(), *param.get("optim_lr2").unwrap());
                let gd = GradientDescent::new();
                self.optim = match optim_type {
                    1.0 => Optimiser::Momentum(momentum),
                    2.0 => Optimiser::RMSprop(rms),
                    3.0 => Optimiser::Adam(adam),
                    _ => Optimiser::GradientDescent(gd),
                };   
            }
        }

        let learning_rate = *param.get("lr").unwrap_or(&0.001);

        let (batch_size, sequence_length, grad_size) = d_out.dim();

        let combined_size = batch_size * sequence_length;

        let input = self.input.as_ref().unwrap();

        let reshaped_input = input.view().into_shape((combined_size, self.input_size)).unwrap();
        let reshaped_grad = d_out.view().into_shape((combined_size, grad_size)).unwrap();
    
        // Compute the dot product
        let d_weights = reshaped_input.t().dot(&reshaped_grad);
        let d_bias = d_out.sum_axis(Axis(0)).sum_axis(Axis(0));

        let w_grad: Array2<f32> = d_weights.map(|x| (x/combined_size as f32));
        let b_grad: Array1<f32> = d_bias.map(|x| (x/combined_size as f32));

        match &mut self.optim {
            Optimiser::Momentum(optim) => {
                let d_weights : &Array2<f32> = &optim.param1(&w_grad.into_dyn()).into_dimensionality().unwrap();
                let d_bias : &Array1<f32> = &optim.param2(&b_grad.into_dyn()).into_dimensionality().unwrap();
                let d_weights = d_weights.map(|x| x * learning_rate);
                let d_bias = d_bias.map(|x| x * learning_rate);
                self.weights = &self.weights - &d_weights;
                self.bias = &self.bias - &d_bias;
            }, 
            Optimiser::RMSprop(optim)  => {
                let d_weights : &Array2<f32> = &optim.param1(&w_grad.into_dyn()).into_dimensionality().unwrap();
                let d_bias : &Array1<f32> = &optim.param2(&b_grad.into_dyn()).into_dimensionality().unwrap();
                let d_weights = d_weights.map(|x| x * learning_rate);
                let d_bias = d_bias.map(|x| x * learning_rate);
                self.weights = &self.weights - &d_weights;
                self.bias = &self.bias - &d_bias;
            }, 
            Optimiser::Adam(optim) => {
                let d_weights : &Array2<f32> = &optim.param1(&w_grad.into_dyn()).into_dimensionality().unwrap();
                let d_bias : &Array1<f32> = &optim.param2(&b_grad.into_dyn()).into_dimensionality().unwrap();
                let d_weights = d_weights.map(|x| x * learning_rate);
                let d_bias = d_bias.map(|x| x * learning_rate);
                self.weights = &self.weights - &d_weights;
                self.bias = &self.bias - &d_bias;
            }, 
            Optimiser::GradientDescent(optim) => {
                let d_weights : &Array2<f32> = &optim.param1(&w_grad.into_dyn()).into_dimensionality().unwrap();
                let d_bias : &Array1<f32> = &optim.param2(&b_grad.into_dyn()).into_dimensionality().unwrap();
                let d_weights = d_weights.map(|x| x * learning_rate);
                let d_bias = d_bias.map(|x| x * learning_rate);
                self.weights = &self.weights - &d_weights;
                self.bias = &self.bias - &d_bias;
            }, 
        }

        // Compute the dot product
        let delta_x = reshaped_grad.dot(&self.weights.t());
        let out_grad = delta_x.view().into_shape((batch_size, sequence_length, self.input_size)).unwrap().to_owned();
        clip(&out_grad, 1.0)
    }

    fn save_parameters_to_file(&mut self, mut file_lines: &mut Vec<String>){

        // Write weights
        file_lines.push("Weights:".to_string());
        for row in self.weights.genrows() {
            let row_str = row.iter()
                .map(|&x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ");
            file_lines.push(row_str.to_string());
        }

        // Write bias
        file_lines.push("Bias:".to_string());
        let bias_str = self.bias.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        file_lines.push(bias_str.to_string());

    }

    fn load_parameters_from_file(&mut self, file_lines: &mut Vec<String>){

        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut bias: Vec<f32> = Vec::new();
        let mut read_weights = false;
        let mut read_bias = false;
        let mut read = true;

        while read {
            let line = file_lines[0].clone();
            if line == "Weights:" {
                read_weights = true;
                read_bias = false;
                file_lines.remove(0);
                continue;
            } else if line == "Bias:" {
                read_weights = false;
                read_bias = true;
                file_lines.remove(0);
                continue;
            }

            if read_weights {
                let row: Vec<f32> = line.split(',')
                    .map(|x| x.trim().parse().unwrap())
                    .collect();
                weights.push(row);
            } else if read_bias {
                bias = line.split(',')
                    .map(|x| x.trim().parse().unwrap())
                    .collect();

                read = false;
            }

            file_lines.remove(0);
            //println!("removed");
        }

        let weights: Array2<f32> = Array::from_shape_vec((weights.len(), weights[0].len()), weights.concat()).unwrap();
        let bias: Array1<f32> = Array::from(bias);

        println!("{:?}", weights.dim());

        self.weights = weights;
        self.bias = bias;

    }
}
//________________________________________________________________
pub struct ReLU {
    threshold: Option<f32>,
    gradient: Option<Array3<f32>>,
    leaky: bool,
}

impl ReLU {
    pub fn new(leaky: bool, thres: Option<f32>) -> Self {
        let threshold = thres;
        let leaky = leaky;

        Self { threshold, gradient: None, leaky}
    }
}

impl Layer for ReLU {
  
    fn forward(&mut self, training: &bool, input_vector: &Array3<f32>) -> Array3<f32> {
        // Forward computation code here
        let mut min_mul: f32 = 0.0;

        if self.leaky{
            min_mul = 0.1;
        }

        self.gradient = Some(input_vector.mapv(|a| if a <= 0.0 { min_mul } else { 1.0 }));

        if self.threshold == None {
            return input_vector.mapv(|a| a.max(min_mul * a));
        }
        else{
            return input_vector.mapv(|a| a.max(min_mul * a).min(self.threshold.unwrap()));
        }
    }
    
    fn backward(&mut self, gradient: &Array3<f32>,  param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        // Backward computation code here
        let out = self.gradient.as_ref().unwrap() * gradient;
        clip(&out, 1.0)
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>){
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
    }
}
//________________________________________________________________
pub struct Sigmoid {
    sigmoid: Option<ArrayD<f32>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { sigmoid: None }
    }

    pub fn forward(&mut self, input_vector: &ArrayD<f32>) -> ArrayD<f32> {
        self.sigmoid = Some(input_vector.mapv(|a| 1.0/(1.0+f32::exp(-a))));
        self.sigmoid.clone().unwrap()
    }
    
    pub fn backward(&mut self, gradient: &ArrayD<f32>,param: &Option<&HashMap<String, f32>>) -> ArrayD<f32> {
        let sig = self.sigmoid.as_ref().unwrap();
        let grad = sig.mapv(|a| a * (1.0 - a));
        let out = &grad * gradient;
        clipd(&out, 1.0)
    }
}
//________________________________________________________________
pub struct Tanh {
    output: Option<ArrayD<f32>>,
}

impl Tanh {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, input_vector: &ArrayD<f32>) -> ArrayD<f32> {
        self.output = Some(input_vector.mapv(|a| a.tanh()));
        self.output.clone().unwrap()
    }
    
    pub fn backward(&mut self, gradient: &ArrayD<f32>,param: &Option<&HashMap<String, f32>>) -> ArrayD<f32> {
        let grad_tanh = self.output.as_ref().expect("output does not exist").mapv(|x| 1.0 - x.powi(2));
        let out = &grad_tanh * gradient;
        clipd(&out, 1.0)
    }
}
//________________________________________________________________
//CHECK
pub struct Softmax {
    softmax: Option<ArrayD<f32>>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { softmax: None }
    }
  
    pub fn forward(&mut self, input_vector: &ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
        let axis_to_fold = match input_vector.ndim() {
            3 => Axis(2),
            4 => Axis(3),
            _ => panic!("Unsupported dimensionality!"),
        };
    
        let max_vals = input_vector.fold_axis(axis_to_fold, f32::NEG_INFINITY, |&acc, &x| {
            if x > acc { x } else { acc }
        });
    
        let centered_x = input_vector - &max_vals.insert_axis(axis_to_fold);
        let exp_x = centered_x.mapv(|val| val.exp() + 1e-15);
        let sum_exp = exp_x.sum_axis(axis_to_fold);
        let denom = sum_exp.insert_axis(axis_to_fold);
    
        let output = exp_x / denom;
    
        self.softmax = Some(output.clone());

        output
    }
    
    pub fn backward(&mut self, gradient: &ndarray::ArrayD<f32>,param: &Option<&HashMap<String, f32>>) -> ndarray::ArrayD<f32> {
        if let Some(ref softmax) = self.softmax {
            let axis_to_sum = match gradient.ndim() {
                3 => Axis(2),
                4 => Axis(3),
                _ => panic!("Unsupported dimensionality!"),
            };
    
            let sum_dy_times_y = (gradient).sum_axis(axis_to_sum).insert_axis(axis_to_sum);
            let dx = gradient - &(softmax * &sum_dy_times_y);
            clipd(&dx, 1.0)

        } else {
            panic!("Forward pass must be called before backward pass.");
        }
    }
}
//________________________________________________________________
pub struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
    parameters_shape: usize,
    x_cache: Option<Array3<f32>>,
    optim_available: bool,
    optim: Optimiser,
    optimiser: String,  // Cache for input x
}

impl LayerNorm {
    pub fn new(parameters_shape: usize) -> Self {
        let gamma = Array1::ones(parameters_shape);
        let beta = Array1::zeros(parameters_shape);
        let gd = GradientDescent::new();

        Self { gamma, beta, eps: 1e-8, parameters_shape, x_cache: None,
            optim_available: true,
            optim: Optimiser::GradientDescent(gd),
            optimiser: String::from("GD"), }
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, training: &bool, x: &Array3<f32>) -> Array3<f32> {
        // Compute mean and std along the last dimension
        let mean = x.mean_axis(Axis(2)).unwrap();
        let variance = x.mapv(|val| val.powi(2)).mean_axis(Axis(2)).unwrap() - mean.mapv(|val| val.powi(2));
        let std = (variance).mapv(|val| (val + self.eps).sqrt());

        // Normalize
        let y = (x - &mean.insert_axis(Axis(2))) / &std.insert_axis(Axis(2));

        // Scale and shift
        let out = &y * &self.gamma + &self.beta;//.to_owned().insert_axis(Axis(0)).insert_axis(Axis(0));

        // Cache the input x for backward pass
        self.x_cache = Some(x.to_owned());

        out
    }

    fn backward(&mut self, dout: &Array3<f32>,  param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        
        let param = param.as_ref().expect("Linear: parameters uninitialised!");

        if self.optim_available {
            if let Some(optim_type) = param.get("optimiser") {
                self.optim_available = false;
                let momentum = Momentum::new(*param.get("optim_lr1").unwrap());
                let rms: RMSprop = RMSprop::new(*param.get("optim_lr1").unwrap());
                let adam = Adam::new(*param.get("optim_lr1").unwrap(), *param.get("optim_lr2").unwrap());
                let gd = GradientDescent::new();
                self.optim = match optim_type {
                    1.0 => Optimiser::Momentum(momentum),
                    2.0 => Optimiser::RMSprop(rms),
                    3.0 => Optimiser::Adam(adam),
                    _ => Optimiser::GradientDescent(gd),
                };   
            }
            
        }

        let learning_rate = *param.get("lr").unwrap_or(&0.001);


        let x = self.x_cache.as_ref()
            .expect("Forward pass not called");

        // Your code for computing the gradients here.
        // You'll likely need to compute intermediate gradients
        // for mean and variance as well.

        // For example:
        let mean = x.mean_axis(Axis(2))
            .expect("Backward Pass: failed to calculate mean!");

        let variance = x.mapv(|val| val.powi(2)).mean_axis(Axis(2))
            .expect("Backward Pass: failed to calculate!") - mean.mapv(|val| val.powi(2));
        
        let std = (variance).mapv(|val| (val + self.eps).sqrt());
        let mean = mean.insert_axis(Axis(2));
        let std = std.insert_axis(Axis(2));
        let variance = variance.insert_axis(Axis(2));

        let dx_hat = dout * &self.gamma;
        let dvar = &dx_hat * &(x - &mean) * -0.5 * variance.mapv(|val| (val + self.eps).powf(-1.5));
        let dmean = ((&dx_hat * -1.0) / &std) + (&dvar * -2.0 * (x - &mean));
        
        let dx = dout * &self.gamma / &std + &dvar * 2.0 * dout * (x - &mean) + dout * &dmean;
        let dgamma = (dout * &((x - &mean) / std)).mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap();
        let dbeta = dout.mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap();

            // Here, you'll also need to update self.gamma and self.beta using dgamma and dbeta.
        
        let dgamma = dgamma.mapv(|val| val * learning_rate);
        let dbeta = dbeta.mapv(|val| val * learning_rate);

        match &mut self.optim {
            Optimiser::Momentum(optim) => {
                let d_gamma : &Array1<f32> = &optim.param1(&dgamma.into_dyn()).into_dimensionality().unwrap();
                let d_beta : &Array1<f32> = &optim.param2(&dbeta.into_dyn()).into_dimensionality().unwrap();
                let d_gamma = d_gamma.mapv(|val| val * learning_rate);
                let d_beta = d_beta.mapv(|val| val * learning_rate);
                self.gamma = &self.gamma - &d_gamma;
                self.beta = &self.beta - &d_beta;
            }, 
            Optimiser::RMSprop(optim)  => {
                let d_gamma : &Array1<f32> = &optim.param1(&dgamma.into_dyn()).into_dimensionality().unwrap();
                let d_beta : &Array1<f32> = &optim.param2(&dbeta.into_dyn()).into_dimensionality().unwrap();
                let d_gamma = d_gamma.mapv(|val| val * learning_rate);
                let d_beta = d_beta.mapv(|val| val * learning_rate);
                self.gamma = &self.gamma - &d_gamma;
                self.beta = &self.beta - &d_beta;
            }, 
            Optimiser::Adam(optim) => {
                let d_gamma : &Array1<f32> = &optim.param1(&dgamma.into_dyn()).into_dimensionality().unwrap();
                let d_beta : &Array1<f32> = &optim.param2(&dbeta.into_dyn()).into_dimensionality().unwrap();
                let d_gamma = d_gamma.mapv(|val| val * learning_rate);
                let d_beta = d_beta.mapv(|val| val * learning_rate);
                self.gamma = &self.gamma - &d_gamma;
                self.beta = &self.beta - &d_beta;
            }, 
            Optimiser::GradientDescent(optim) => {
                let d_gamma : &Array1<f32> = &optim.param1(&dgamma.into_dyn()).into_dimensionality().unwrap();
                let d_beta : &Array1<f32> = &optim.param2(&dbeta.into_dyn()).into_dimensionality().unwrap();
                let d_gamma = d_gamma.mapv(|val| val * learning_rate);
                let d_beta = d_beta.mapv(|val| val * learning_rate);
                self.gamma = &self.gamma - &d_gamma;
                self.beta = &self.beta - &d_beta;
            }, 
        }
        
        clip(&dx, 1.0) // return the gradient of the loss with respect to the input
    }

    fn save_parameters_to_file(&mut self, mut file_lines: &mut Vec<String>){

        // Write weights
        file_lines.push("Gamma:".to_string());
        let gamma_str = self.gamma.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        file_lines.push(gamma_str.to_string());

        // Write bias
        file_lines.push("Beta:".to_string());
        let beta_str = self.beta.iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        file_lines.push(beta_str.to_string());
    }

    fn load_parameters_from_file(&mut self, file_lines: &mut Vec<String>){

        let mut gamma: Vec<f32> = Vec::new();
        let mut beta: Vec<f32> = Vec::new();
        let mut read_gamma = false;
        let mut read_beta = false;
        let mut read = true;

        while read{
            let line = file_lines[0].clone();
            if line == "Gamma:" {
                read_gamma = true;
                read_beta = false;
                file_lines.remove(0);
                continue;
            } else if line == "Beta:" {
                read_gamma = false;
                read_beta = true;
                file_lines.remove(0);
                continue;
            }

            if read_gamma {
                gamma = line.split(',')
                    .map(|x| x.trim().parse().unwrap())
                    .collect();
            } else if read_beta {
                beta = line.split(',')
                    .map(|x| x.trim().parse().unwrap())
                    .collect();

                read = false
            }

            file_lines.remove(0);
            //println!("removed");
        }

        let gamma: Array1<f32> = Array::from(gamma);
        let beta: Array1<f32> = Array::from(beta);

        println!("{:?}", gamma.dim());

        self.gamma = gamma;
        self.beta = beta;

    }
}
//________________________________________________________________
pub struct Dropout {
    drop_prob: f32,
    mask: Option<Array3<f32>>,
}

impl Dropout {
    pub fn new(prob: f32) -> Self {
        let drop_prob = prob;
        Dropout { drop_prob, mask: None }
    }

}

impl Layer for Dropout {

    fn forward(&mut self, training: &bool, input_vector: &Array3<f32>) -> Array3<f32> {
        // Forward computation code here
        let shape = input_vector.dim();
        let mut rng = rand::thread_rng(); 

        let mask = Array3::from_shape_fn(shape, |_| {
            if rng.gen::<f32>() > self.drop_prob {
                1.0 / (1.0 - self.drop_prob)
            } else {
                0.0
            }
        });

        self.mask = Some(mask);

        if training == &false {
            return input_vector.clone()
        }
        
        input_vector * self.mask.as_ref().unwrap()
    }
    
    fn backward(&mut self, gradient: &Array3<f32>, param: &Option<&HashMap<String, f32>>) -> Array3<f32> {
        // Backward computation code here
        let out = gradient * self.mask.as_ref().unwrap();
        clip(&out, 1.0)
    }

    fn save_parameters_to_file(&mut self, file_lines: &mut Vec<String>){
    }
    fn load_parameters_from_file(&mut self,  file_lines: &mut Vec<String>){
    }
}
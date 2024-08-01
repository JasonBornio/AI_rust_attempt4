
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use crate::ml::types::Opt;

//________________________________________________________________
pub enum Optimiser {
    GradientDescent(GradientDescent),
    Momentum(Momentum),
    RMSprop(RMSprop),
    Adam(Adam),
}
//________________________________________________________________
pub struct GradientDescent {
    
}

impl GradientDescent {
    pub fn new() -> Self{
        Self {

        }
    }
}

impl Opt for GradientDescent {
    fn param1(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {
        gradient.clone()
    }

    fn param2(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {
        gradient.clone()
    }
}
//________________________________________________________________
pub struct Momentum {
    average_param1: Option<ArrayD<f32>>,
    average_param2: Option<ArrayD<f32>>,
    beta_1: f32,
    init_param1: bool,
    init_param2: bool,
}

impl Momentum {
    pub fn new(beta_1: f32) -> Self{
        Self {
            average_param1: None,
            average_param2: None,
            beta_1,
            init_param1: true,
            init_param2: true,
        }
    }
}

impl Opt for Momentum{
    fn param1(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param1== true {
            self.init_param1 = false;
            self.average_param1 = Some(ArrayD::zeros(gradient.dim()));
        }

        let avg2d = self.average_param1.to_owned().expect("Adam: average_param2d not initalised");

        let avg2d = avg2d.mapv(|val| val * &self.beta_1) + gradient.mapv(|val| val * (1.0 - &self.beta_1));
        
        let output = avg2d.clone();

        self.average_param1 = Some(avg2d);
        
        output
    }

    fn param2(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param2 == true {
            self.init_param2 = false;
            self.average_param2 = Some(ArrayD::zeros(gradient.dim()));
        }

        let avg1d = self.average_param2.to_owned().expect("Adam: average_param2d not initalised");

        let avg1d = avg1d.mapv(|val| val * &self.beta_1) + gradient.mapv(|val| val * (1.0 - &self.beta_1));
        
        let output = avg1d.clone();

        self.average_param2 = Some(avg1d);
        
        output
    }
}
//________________________________________________________________
pub struct RMSprop {
    s_dparam1: Option<ArrayD<f32>>,
    s_dparam2: Option<ArrayD<f32>>,
    beta_1: f32,
    init_param1: bool,
    init_param2: bool,
}

impl RMSprop {
    pub fn new(beta_1: f32) -> Self{
        Self {
            s_dparam1: None,
            s_dparam2: None,
            beta_1,
            init_param1: true,
            init_param2: true,
        }
    }
}

impl Opt for RMSprop {
    fn param1(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param1 == true {
            self.init_param1 = false;
            self.s_dparam1 = Some(ArrayD::zeros(gradient.dim()));
        }

        let s_d2d = self.s_dparam1.to_owned().expect("Adam: s_dparam2d not initalised");

        let s_d2d = s_d2d.mapv(|val| val * &self.beta_1) + gradient.mapv(|val| val.powi(2) * (1.0 - &self.beta_1));
        
        let output = gradient / &s_d2d.mapv(|val| val.powf(0.5) + 1e-15);

        self.s_dparam1  = Some(s_d2d);
        
        output
    }

    fn param2(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param2 == true {
            self.init_param2 = false;
            self.s_dparam2 = Some(ArrayD::zeros(gradient.dim()));
        }

        let s_d1d = self.s_dparam2.to_owned().expect("Adam: s_dparam2d not initalised");

        let s_d1d = s_d1d.mapv(|val| val * &self.beta_1) + gradient.mapv(|val| val.powi(2) * (1.0 - &self.beta_1));
        
        let output = gradient / &s_d1d.mapv(|val| val.powf(0.5) + 1e-15);

        self.s_dparam2 = Some(s_d1d);
        
        output
    }
}
//________________________________________________________________
pub struct Adam {
    s_dparam2: Option<ArrayD<f32>>,
    s_dparam1: Option<ArrayD<f32>>,
    average_param2: Option<ArrayD<f32>>,
    average_param1: Option<ArrayD<f32>>,
    beta_1: f32,
    beta_2: f32,
    init_param2: bool,
    init_param1: bool,
}

impl Adam {
    pub fn new(beta_1: f32, beta_2: f32) -> Self{
        Self {
            s_dparam2: None,
            s_dparam1: None,
            average_param1: None,
            average_param2: None,
            beta_1,
            beta_2,
            init_param1: true,
            init_param2: true,
        }
    }
}

impl Opt for Adam{
    fn param1(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param1 == true {
            self.init_param1 = false;
            self.average_param1 = Some(ArrayD::zeros(gradient.dim()));
            self.s_dparam1 = Some(ArrayD::zeros(gradient.dim()));
        }

        //        self.average_w = self.beta_1 * self.average_w + (1-self.beta_1) * gradient
        //self.s_dW = self.beta_2 * self.s_dW + (1 - self.beta_2) * (gradient ** 2)

        let avg2d = self.average_param1.to_owned().expect("Adam: average_param2d not initalised");
        let s_d2d= self.s_dparam1.to_owned().expect("Adam: s_dparam2d not initalised");

        let avg2d = avg2d.mapv(|val| val * &self.beta_1) + &gradient.mapv(|val| val * (1.0 - &self.beta_1));
        let s_d2d = s_d2d.mapv(|val| val * &self.beta_2) + &gradient.mapv(|val| val.powi(2) * (1.0 - &self.beta_2));
        
        let output = &avg2d / &s_d2d.mapv(|val| val.powf(0.5) + 1e-15);

        self.average_param1 = Some(avg2d);
        self.s_dparam1 = Some(s_d2d);
        
        output
    }

    fn param2(&mut self, gradient: &ArrayD<f32>) -> ArrayD<f32> {

        if self.init_param2 == true {
            self.init_param2 = false;
            self.average_param2 = Some(ArrayD::zeros(gradient.dim()));
            self.s_dparam2 = Some(ArrayD::zeros(gradient.dim()));
        }

        let avg1d = self.average_param2.to_owned().expect("Adam: average_param1d not initalised");
        let s_d1d = self.s_dparam2.to_owned().expect("Adam: s_dparam1d not initalised");

        let avg1d = avg1d.mapv(|val| val * &self.beta_1) + &gradient.mapv(|val| val * (1.0 - &self.beta_1));
        let s_d1d = s_d1d.mapv(|val| val * &self.beta_2) + &gradient.mapv(|val| val.powi(2) * (1.0 - &self.beta_2));
        
        let output = &avg1d / &s_d1d.mapv(|val| val.powf(0.5) + 1e-15);

        self.average_param2 = Some(avg1d);
        self.s_dparam2 = Some(s_d1d);
        
        output
    }
}
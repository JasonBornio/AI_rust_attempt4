extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::Array3;
use crate::ml::util::*;

//________________________________________________________________
pub struct MSELoss {
    predictions: Option<Array3<f32>>,
    labels: Option<Array3<f32>>,
}

impl MSELoss { 
    pub fn new() -> Self {
        Self {
            predictions: None,
            labels: None,
        }
    }

    pub fn calculate(&mut self, predictions: Array3<f32>, labels: Array3<f32>) -> f32 {
        let batch_size = predictions.dim().0;

        self.labels = Some(labels);
        self.predictions = Some(predictions);

        let subtraction = self.labels.as_ref().unwrap() - self.predictions.as_ref().unwrap();
        let loss = subtraction.mapv(|x| x.powf(2.0) * 0.5);

        loss.sum() / batch_size as f32
    }

    pub fn backward(&mut self) -> Array3::<f32> {
        let batch_size = self.predictions.as_ref().unwrap().dim().0;
        let mut out = (self.predictions.as_ref().unwrap() - self.labels.as_ref().unwrap()) / batch_size as f32;
        clip(&out, 1.0)
    }
}
//________________________________________________________________
pub struct BinaryCrossEntropyLoss {
    predictions: Option<Array3<f32>>,
    labels: Option<Array3<f32>>,
}

impl BinaryCrossEntropyLoss { 
    pub fn new() -> Self {
        Self {
            predictions: None,
            labels: None,
        }
    }

    pub fn calculate(&mut self, predictions: Array3<f32>, labels: Array3<f32>) -> f32 {
        let (batch_size, _, _) = predictions.dim();

        self.labels = Some(labels);
        self.predictions = Some(predictions);
        //- torch.mean((labels * torch.log(predictions + 1e-8))
        //loss = - torch.mean((labels * torch.log(predictions + 1e-15) + (1-labels)*torch.log(1-(predictions + 1e-15)))* self.bool_tensor)

        let loss = -(self.labels.as_ref().unwrap() * &self.predictions.as_ref().unwrap().mapv(|x|(x +  1e-15).ln()) 
            + self.labels.as_ref().unwrap().mapv(|x| 1.0 - x) 
            *  &self.predictions.as_ref().unwrap().mapv(|x|(1.0 - x +  1e-15).ln()));

        //println!("PREDS: {:?}", self.predictions);
        loss.sum()  / batch_size as f32
    }

    pub fn backward(&mut self) -> Array3::<f32> {
        //-((self.labels/(self.predictions + 1e-15)) - (1-self.labels)/(1-self.predictions + 1e-15))/len(self.labels)
        let batch_size = self.predictions.as_ref().unwrap().dim().0;
        let out = -(self.labels.as_ref().unwrap() / &(self.predictions.as_ref().unwrap().mapv(|val| val + 1e-15))
        - self.labels.as_ref().unwrap().mapv(|val| 1.0 - val + 1e-15) / self.predictions.as_ref().unwrap().mapv(|val| 1.0 - val + 1e-15)) / batch_size as f32;
        clip(&out, 1.0)
    }
}
//________________________________________________________________
pub struct CrossEntropyLoss {
    predictions: Option<Array3<f32>>,
    labels: Option<Array3<f32>>,
}

impl CrossEntropyLoss { 
    pub fn new() -> Self {
        Self {
            predictions: None,
            labels: None,
        }
    }

    pub fn calculate(&mut self, predictions: Array3<f32>, labels: Array3<f32>) -> f32 {
        let (batch_size, _, _) = predictions.dim();

        self.labels = Some(labels);
        self.predictions = Some(predictions);
        //- torch.mean((labels * torch.log(predictions + 1e-8))
        let exp = self.predictions.as_ref().unwrap().mapv(|x|(x +  1e-15).ln());
        let loss = -self.labels.as_ref().unwrap() * exp;
        //println!("PREDS: {:?}", self.predictions);
        loss.sum()  / batch_size as f32
    }

    pub fn backward(&mut self) -> Array3::<f32> {
        let batch_size = self.predictions.as_ref().unwrap().dim().0;
        let out = -(self.labels.as_ref().unwrap() / &(self.predictions.as_ref().unwrap().mapv(|val| val + 1e-15))) / batch_size as f32;
        clip(&out, 1.0)
    }
}
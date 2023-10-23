# RayBNN_Optimizer

Gradient Descent Optimizers and Genetic Algorithms using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI 

* ADAM
* SGD
* Genetic
* Random Search




# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.7.0"
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
RayBNN_Optimizer = "0.1.1"
```

# List of Examples


# Optimizing values for a loss function
```

//Define Starting Point for optimization
let x0_cpu = vec![0.1, 0.4, 0.5,   -1.2, 0.7];
let x0_dims = arrayfire::Dim4::new(&[1, x0_cpu.len() as u64, 1, 1]);
let x0 = arrayfire::Array::new(&x0_cpu, x0_dims);

//Define the loss function
let y_cpu = vec![-1.1, 0.4, 2.0,    2.1, 4.0];
let y = arrayfire::Array::new(&y_cpu, x0_dims);

//Define the loss function
let loss = |yhat: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
    RayBNN_Optimizer::Continuous::Loss::MSE(yhat, &y)
};

//Define the gradient of the loss function
let loss_grad = |yhat: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
    RayBNN_Optimizer::Continuous::Loss::MSE_grad(yhat, &y)
};


let mut point = x0.clone();
let mut direction = -loss_grad(&point);
let mut mt = arrayfire::constant::<f64>(0.0,direction.dims());
let mut vt = arrayfire::constant::<f64>(0.0,direction.dims());

let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
let mut alpha = arrayfire::constant::<f64>(1.0,single_dims);

let alpha_max = arrayfire::constant::<f64>(1.0,single_dims);

let rho = arrayfire::constant::<f64>(0.1,single_dims);

//Create alpha values to sweep through
let v = 30;
let alpha_vec = RayBNN_Optimizer::Continuous::LR::create_alpha_vec::<f64>(v, 1.0, 0.5);


let beta0 = arrayfire::constant::<f64>(0.9,single_dims);
let beta1 = arrayfire::constant::<f64>(0.999,single_dims);

//Optimization Loop
for i in 0..120
{
    alpha = alpha_max.clone();
    //Automatically Determine Optimal Step Size using BTLS
    RayBNN_Optimizer::Continuous::LR::BTLS(
        loss
        ,loss_grad
        ,&point
        ,&direction
        ,&alpha_vec
        ,&rho
        ,&mut alpha
    );

    //Update current point
    point = point.clone()  + alpha*direction.clone();
    direction = -loss_grad(&point);



    //Use ADAM optimizer
    RayBNN_Optimizer::Continuous::GD::adam(
        &beta0
        ,&beta1
        ,&mut direction
        ,&mut mt
        ,&mut vt
    );

}

```


# Types of Loss Functions
```
let mut cross_entropy = RayBNN_Optimizer::Continuous::Loss::softmax_cross_entropy(&Yhat,&Y);
let mut cross_entropy_grad = RayBNN_Optimizer::Continuous::Loss::softmax_cross_entropy_grad(&Yhat,&Y);
let mut cross_entropy = RayBNN_Optimizer::Continuous::Loss::sigmoid_cross_entropy(&Yhat,&Y);
let mut cross_entropy_grad = RayBNN_Optimizer::Continuous::Loss::sigmoid_cross_entropy_grad(&Yhat,&Y);
let mut MAE = RayBNN_Optimizer::Continuous::Loss::MAE(&Yhat,&Y);
let mut MSE = RayBNN_Optimizer::Continuous::Loss::MSE(&Yhat,&Y);
let MSE_grad = RayBNN_Optimizer::Continuous::Loss::MSE_grad(&Yhat,&Y);
let mut RMSE = RayBNN_Optimizer::Continuous::Loss::RMSE(&Yhat,&Y);
```






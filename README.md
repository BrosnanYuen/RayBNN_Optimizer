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
RayBNN_Optimizer = "0.1.0"
```

# List of Examples





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






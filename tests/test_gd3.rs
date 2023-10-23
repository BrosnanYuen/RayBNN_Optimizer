#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use rayon::prelude::*;


#[test]
fn test_gd3() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);







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



    let mut point_act:Vec<f64> = y_cpu.clone();

    let mut point_cpu = vec!(f64::default();point.elements());

    point.host(&mut point_cpu);

    point_cpu = point_cpu.par_iter().map(|x|  (x * 100.0).round() / 100.0 ).collect::<Vec<f64>>();

    point_act = point_act.par_iter().map(|x|  (x * 100.0).round() / 100.0 ).collect::<Vec<f64>>();


    assert_eq!(point_act, point_cpu);





}

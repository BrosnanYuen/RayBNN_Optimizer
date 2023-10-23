#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_gd() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);







	let n: u64 = 2;
	let v = 30;
	let x0_cpu: [f64; 2] = [2.0, 3.0];
	let x0 = arrayfire::Array::new(&x0_cpu, arrayfire::Dim4::new(&[1, n, 1, 1]));

	let loss = |x: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		let mut z = vec!(f64::default();x.elements());
		x.host(&mut z);
		let x1 = z[0];
		let x2 = z[1];
		let ret = vec![ (( (x1*x2) -x1 +1.5).powf(2.0)) + (( (x1*(x2.powf(2.0))) -x1 +2.25 ).powf(2.0)) + (( (x1*(x2.powf(3.0))) -x1 +2.625 ).powf(2.0))];
	
        arrayfire::Array::new(&ret, arrayfire::Dim4::new(&[1, 1, 1, 1]))
    };

	let loss_grad = |x: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		let mut z = vec!(f64::default();x.elements());
		x.host(&mut z);
		let x1 = z[0];
		let x2 = z[1];
		let g1 = 2.0*(x2.powf(2.0) - 1.0)*(x1*x2.powf(2.0) - x1 + 9.0/4.0) + 2.0*(x2.powf(3.0) - 1.0)*(x1*x2.powf(3.0) - x1 + 21.0/8.0) + 2.0*(x2 - 1.0)*(x1*x2 - x1 + 3.0/2.0);
		let g2 = 2.0*x1*(x1*x2 - x1 + 3.0/2.0) + 4.0*x1*x2*(x1*x2.powf(2.0) - x1 + 9.0/4.0) + 6.0*x1*x2.powf(2.0)*(x1*x2.powf(3.0) - x1 + 21.0/8.0);

		z[0] = g1;
		z[1] = g2;
		arrayfire::Array::new(&z, arrayfire::Dim4::new(&[1, n, 1, 1]))
	};


	let mut point = x0.clone();
	let mut loss_val = loss(&point);
	let mut direction = -loss_grad(&point);
	let mut mt = arrayfire::constant::<f64>(0.0,direction.dims());
	let mut vt = arrayfire::constant::<f64>(0.0,direction.dims());

	let mut alpha = arrayfire::constant::<f64>(1.0,arrayfire::Dim4::new(&[1, 1, 1, 1]));
	let mut next_point = point.clone();
	let mut newdirection = direction.clone();



    let alpha_max = arrayfire::constant::<f64>(1.0,arrayfire::Dim4::new(&[1, 1, 1, 1]));

    let rho = arrayfire::constant::<f64>(0.1,arrayfire::Dim4::new(&[1, 1, 1, 1]));

	let alpha_vec = RayBNN_Optimizer::Continuous::LR::create_alpha_vec::<f64>(v, 1.0, 0.5);

	let beta = arrayfire::constant::<f64>(0.9,arrayfire::Dim4::new(&[1, 1, 1, 1]));

	for i in 0..400
	{
        alpha = alpha_max.clone();
		RayBNN_Optimizer::Continuous::LR::BTLS(
			loss
			,loss_grad
			,&point
			,&direction
			,&alpha_vec
			,&rho
			,&mut alpha
        );


		point = point.clone()  + alpha*direction.clone();
		direction = -loss_grad(&point);




		RayBNN_Optimizer::Continuous::GD::momentum(
            &beta
            ,&direction
            ,&mut newdirection
        );
        direction = newdirection.clone();

	}



    let mut point_act:Vec<f64> = vec![3.0, 0.5];

    let mut point_cpu = vec!(f64::default();point.elements());

    point.host(&mut point_cpu);

    point_cpu = point_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    point_act = point_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(point_act, point_cpu);









}

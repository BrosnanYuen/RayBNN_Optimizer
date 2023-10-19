#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_hidden_generate2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);







    let mut positions_cpu:Vec<f32> = vec![ 0.1, -0.7, 7.3,     9.3,0.61,1.21,          1.3, 4.2, 0.9,       7.5, 2.7, 2.1,      9.3,0.6,1.2,      1.3,-1.0,6.1,         9.31,0.6,1.21,     1.0,-1.0,6.1];
    let mut positions = arrayfire::Array::new(&positions_cpu, arrayfire::Dim4::new(&[3, 8, 1, 1]));

    positions = arrayfire::transpose(&positions, false);


	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	
	let neg_idx = RayBNN_Cell::Hidden::Sphere::select_non_overlap(
		&positions, 
		0.1
	);

	let mut neg_idx_cpu = vec!(u32::default();neg_idx.elements());
	neg_idx.host(&mut neg_idx_cpu);

	assert_eq!(neg_idx_cpu,vec![1,4,6]);


	positions = arrayfire::lookup(&positions, &neg_idx, 0);

	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	
















    let mut positions_cpu:Vec<f64> = vec![ 0.1, -0.7, 7.3,     9.3,0.61,1.21,          1.3, 4.2, 0.9,       7.5, 2.7, 2.1,      9.3,0.6,1.2,      1.3,-1.0,6.1,         9.31,0.6,1.21,     1.0,-1.0,6.1];
    let mut positions = arrayfire::Array::new(&positions_cpu, arrayfire::Dim4::new(&[3, 8, 1, 1]));

    positions = arrayfire::transpose(&positions, false);


	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	
	let neg_idx = RayBNN_Cell::Hidden::Sphere::select_non_overlap(
		&positions, 
		0.1
	);

	let mut neg_idx_cpu = vec!(u32::default();neg_idx.elements());
	neg_idx.host(&mut neg_idx_cpu);

	assert_eq!(neg_idx_cpu,vec![1,4,6]);


	positions = arrayfire::lookup(&positions, &neg_idx, 0);

	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	

}

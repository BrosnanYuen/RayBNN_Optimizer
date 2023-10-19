#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_hidden_generate() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);




    let mut positions_cpu:Vec<f32> = vec![ 0.0,-1.0,6.1,     1.0,-2.0,6.1,          1.0,-1.0,6.1,       7.5, 2.7, 2.1,      1.0,-1.0,6.7,      1.0,-1.0,6.1,         1.0,-1.0,5.1,     1.0,-1.0,5.7];
    let mut positions = arrayfire::Array::new(&positions_cpu, arrayfire::Dim4::new(&[3, 8, 1, 1]));

    positions = arrayfire::transpose(&positions, false);


	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	let pivot = vec![0.1, -1.5, 5.5];
	let idx_intersect2 = RayBNN_Cell::Hidden::Sphere::get_inside_idx_cubeV2(
		&positions
		, 1.0
		, &pivot
	);

	let mut idx_intersect2_cpu = vec!(u32::default();idx_intersect2.elements());
	idx_intersect2.host(&mut idx_intersect2_cpu);

	assert_eq!(idx_intersect2_cpu,vec![2, 5, 7]);














    let mut positions_cpu:Vec<f64> = vec![ 0.0,-1.0,6.1,     1.0,-2.0,6.1,          1.0,-1.0,6.1,       7.5, 2.7, 2.1,      1.0,-1.0,6.7,      1.0,-1.0,6.1,         1.0,-1.0,5.1,     1.0,-1.0,5.7];
    let mut positions = arrayfire::Array::new(&positions_cpu, arrayfire::Dim4::new(&[3, 8, 1, 1]));

    positions = arrayfire::transpose(&positions, false);


	//arrayfire::print_gen("positions".to_string(), &positions, Some(6));

	let pivot = vec![0.1, -1.5, 5.5];
	let idx_intersect2 = RayBNN_Cell::Hidden::Sphere::get_inside_idx_cubeV2(
		&positions
		, 1.0
		, &pivot
	);

	let mut idx_intersect2_cpu = vec!(u32::default();idx_intersect2.elements());
	idx_intersect2.host(&mut idx_intersect2_cpu);

	assert_eq!(idx_intersect2_cpu,vec![2, 5, 7]);






}

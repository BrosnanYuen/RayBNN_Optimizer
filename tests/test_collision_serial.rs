#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use std::collections::HashMap;
use std::time::{Duration, Instant};



const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

const TWO_F64: f64 = 2.0;

#[test]
fn test_sphere_cell_collision_serial() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<f32>();


	let neuron_size: u64 = 51000;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 3;
	let active_size: u64 = 500000;
	let space_dims: u64 = 3;
	let mut batch_size: u64 = 105;

	let neuron_rad = 0.1;
    let time_step = 0.3;
    let nratio =  0.5;
    let neuron_std =  0.3;
    let sphere_rad =  30.0;


    let mut modeldata_float: HashMap<String, f64> = HashMap::new();
    let mut modeldata_int: HashMap<String, u64>  = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size.clone());
    modeldata_int.insert("input_size".to_string(), input_size.clone());
    modeldata_int.insert("output_size".to_string(), output_size.clone());
    modeldata_int.insert("proc_num".to_string(), proc_num.clone());
    modeldata_int.insert("active_size".to_string(), active_size.clone());
    modeldata_int.insert("space_dims".to_string(), space_dims.clone());
    modeldata_int.insert("batch_size".to_string(), batch_size.clone());





    modeldata_float.insert("neuron_rad".to_string(), neuron_rad.clone());
    modeldata_float.insert("time_step".to_string(), time_step.clone());
    modeldata_float.insert("nratio".to_string(), nratio.clone());
    modeldata_float.insert("neuron_std".to_string(), neuron_std.clone());
    modeldata_float.insert("sphere_rad".to_string(), sphere_rad.clone());



	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);


    let mut cell_pos: arrayfire::Array<f32>  = RayBNN_Cell::Hidden::Sphere::generate_uniform_sphere_posiiton(&modeldata_float, &modeldata_int);


    println!("cell_pos {}", cell_pos.dims()[0]);

    assert_eq!(cell_pos.dims()[0], active_size*2);
    assert_eq!(cell_pos.dims()[1], space_dims);

    let start = Instant::now();

    let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_serial(
        &modeldata_float, 
        &cell_pos
    );

    let idx = arrayfire::locate(&idx);

	cell_pos = arrayfire::lookup(&cell_pos, &idx, 0);


    RayBNN_Cell::Hidden::Sphere::split_into_glia_neuron(
        &modeldata_float,
    
        &cell_pos,
    
        &mut glia_pos,
        &mut neuron_pos
    );

	let duration = start.elapsed();





    println!("Time elapsed in expensive_function() is: {:?}", duration);

	println!("glia_pos.dims()[0] {}",glia_pos.dims()[0]);
	println!("neuron_pos.dims()[0] {}",neuron_pos.dims()[0]);

	
	assert_eq!(glia_pos.dims()[1],space_dims);
	assert_eq!(neuron_pos.dims()[1],space_dims);

	let total_obj = arrayfire::join(0, &glia_pos, &neuron_pos);
	drop(neuron_pos);
	drop(glia_pos);

	let mut active_size = total_obj.dims()[0];
	assert!(active_size >= 200000);


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let neuron_sq = 4.0*neuron_rad*neuron_rad;

	for i in 0u64..active_size
	{
		let select_pos = arrayfire::row(&total_obj,i as i64);

		let mut dist = arrayfire::sub(&select_pos,&total_obj, true);
		let mut magsq = arrayfire::pow(&dist,&TWO,false);
		let mut magsq = arrayfire::sum(&magsq,1);


		let insert = arrayfire::constant::<f32>(1000000.0,single_dims);

		arrayfire::set_row(&mut magsq, &insert, i as i64);

		let (m0,_) = arrayfire::min_all::<f32>(&magsq);

		//println!("{} dist {}",i, m0);
		assert!((m0 as f64) > neuron_sq);
	}


    let mut magsq = arrayfire::pow(&total_obj,&TWO,false);
    let mut magsq = arrayfire::sum(&magsq,1);

    let (max0,_) = arrayfire::max_all::<f32>(&magsq);

    assert!(sphere_rad*sphere_rad > (max0 as f64));

    println!("max {}", max0);

}

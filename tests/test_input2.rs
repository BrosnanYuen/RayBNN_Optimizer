#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

const TWO_F64: f64 = 2.0;

#[test]
fn test_input2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<f32>();



    let sphere_rad = 14.0;

    let input_size = 71;

    let input_pos: arrayfire::Array<f64> = RayBNN_Cell::Input::Sphere::create_spaced_neurons_1D(
        sphere_rad,
        input_size,
    );

    assert_eq!(input_pos.dims()[0],input_size);
    assert_eq!(input_pos.dims()[1],3);

    let mut magsq = arrayfire::pow(&input_pos,&TWO,false);
    let mut magsq = arrayfire::sum(&magsq,1);

    let (mut max0,_) = arrayfire::max_all::<f64>(&magsq);
    let (mut min0,_) = arrayfire::min_all::<f64>(&magsq);

    max0 = (max0 * 1000000.0).round() / 1000000.0;
    min0 = (min0 * 1000000.0).round() / 1000000.0;

    assert_eq!(max0, min0);
    println!("max {}", max0);
    assert_eq!(max0, sphere_rad*sphere_rad);


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	for i in 0u64..input_pos.dims()[0]
	{
		let select_pos = arrayfire::row(&input_pos,i as i64);

		let mut dist = arrayfire::sub(&select_pos,&input_pos, true);
		let mut magsq = arrayfire::pow(&dist,&TWO,false);
		let mut magsq = arrayfire::sum(&magsq,1);


		let insert = arrayfire::constant::<f64>(1000000.0,single_dims);

		arrayfire::set_row(&mut magsq, &insert, i as i64);

		let (m0,_) = arrayfire::min_all::<f64>(&magsq);

		//println!("{} dist {}",i, m0);
		assert!((m0 as f64) > 5.0);
	}



}

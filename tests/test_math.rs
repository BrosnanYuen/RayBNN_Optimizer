#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_math() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);



    let mut positions_cpu:Vec<f64> = vec![ 0.1, -0.7, 7.3,     -1.0, 0.4, 1.7,          1.3, 4.2, 0.9,       7.5, 2.7, 2.1,      9.3,0.6,1.2  ];
    let mut positions = arrayfire::Array::new(&positions_cpu, arrayfire::Dim4::new(&[3, 5, 1, 1]));

    positions = arrayfire::transpose(&positions, false);

    let mut dist = arrayfire::constant::<f64>(0.0,single_dims);
	let mut magsq = arrayfire::constant::<f64>(0.0,single_dims);

	
    RayBNN_Cell::Util::Math::matrix_dist(
    	&positions,
    	&mut dist,
    	&mut magsq
    );

    assert_eq!(dist.dims()[0], 5);
    assert_eq!(dist.dims()[1], 3);
    assert_eq!(dist.dims()[2], 5);

    assert_eq!(magsq.dims()[0], 5);
    assert_eq!(magsq.dims()[1], 1);
    assert_eq!(magsq.dims()[2], 5);


    let mut magsq_act:Vec<f64> = vec![    0.000000,33.780000,66.410000,93.360000 ,  123.540000 ,   33.780000, 0.000000,20.370000,77.700000 ,  106.380000 ,   66.410000,20.370000, 0.000000,42.130000,77.050000 ,   93.360000,77.700000,42.130000, 0.000000, 8.460000 ,  123.540000 ,  106.380000,77.050000, 8.460000, 0.000000  ];

    let mut magsq_cpu = vec!(f64::default();magsq.elements());
    magsq.host(&mut magsq_cpu);

    magsq_act = magsq_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    magsq_cpu = magsq_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    assert_eq!(magsq_act, magsq_cpu);



    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let val = arrayfire::constant(1000000.0, single_dims);

    RayBNN_Cell::Util::Math::set_diag(
    	&mut magsq,
        &val
    );


    let mut magsq_act:Vec<f64> = vec![    1000000.0,33.780000,66.410000,93.360000 ,  123.540000 ,   33.780000, 1000000.0,20.370000,77.700000 ,  106.380000 ,   66.410000,20.370000, 1000000.0,42.130000,77.050000 ,   93.360000,77.700000,42.130000, 1000000.0, 8.460000 ,  123.540000 ,  106.380000,77.050000, 8.460000, 1000000.0  ];

    let mut magsq_cpu = vec!(f64::default();magsq.elements());
    magsq.host(&mut magsq_cpu);

    magsq_act = magsq_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    magsq_cpu = magsq_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    assert_eq!(magsq_act, magsq_cpu);







}

#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_loss() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);





    let Y_cpu: Vec<f64> = vec![0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0           ,0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0           ,0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0 ];
    let Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[7, 3, 1, 1]));


    let Yhat_cpu: Vec<f64> = vec![0.8,  0.3,  0.5,  0.2,  0.4,  0.1,  0.7        ,0.1,  0.2,  0.4,  0.2,  0.1,  0.6,  0.3           ,0.9,  0.2,  0.4,  0.2,  0.5,  0.6,  0.3 ];
    let Yhat = arrayfire::Array::new(&Yhat_cpu, arrayfire::Dim4::new(&[7, 3, 1, 1]));









    let mut cross_entropy = RayBNN_Optimizer::Continuous::Loss::softmax_cross_entropy(&Yhat,&Y);
    let mut cross_entropy_act:f64 =  0.4;


    cross_entropy = (cross_entropy * 1000000.0).round() / 1000000.0 ;

    cross_entropy_act = (cross_entropy_act * 1000000.0).round() / 1000000.0 ;


    assert_eq!(cross_entropy_act,  cross_entropy);









    let mut cross_entropy_grad = RayBNN_Optimizer::Continuous::Loss::softmax_cross_entropy_grad(&Yhat,&Y);


    let mut cross_entropy_grad_act_cpu:Vec<f64> = vec![0.067097, -0.292637, 0.049707, 0.036824, 0.044977, 0.03332, 0.060712, 0.03954, 0.043698, 0.053373, 0.043698, 0.03954, -0.268143, 0.048294, 0.073105, 0.036303, 0.04434, -0.29703, 0.049004, 0.054158, 0.040121];
    let mut cross_entropy_grad_cpu = vec!(f64::default();cross_entropy_grad.elements());


    cross_entropy_grad.host(&mut cross_entropy_grad_cpu);

    cross_entropy_grad_cpu = cross_entropy_grad_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    cross_entropy_grad_act_cpu = cross_entropy_grad_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();



    assert_eq!(cross_entropy_grad_act_cpu, cross_entropy_grad_cpu);























    


    let Y_cpu: [f64; 5] = [1.0,  0.0,  1.0,  1.0,  0.0];
    let Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[1, 5, 1, 1]));


    let Yhat_cpu: [f64; 5] = [0.2,  -0.5,  0.9,  0.02,  -0.1];
    let Yhat = arrayfire::Array::new(&Yhat_cpu, arrayfire::Dim4::new(&[1, 5, 1, 1]));




    let mut cross_entropy = RayBNN_Optimizer::Continuous::Loss::sigmoid_cross_entropy(&Yhat,&Y);
    let mut cross_entropy_act:f64 =  0.548192713618798;


    cross_entropy = (cross_entropy * 1000000.0).round() / 1000000.0 ;

    cross_entropy_act = (cross_entropy_act * 1000000.0).round() / 1000000.0 ;


    assert_eq!(cross_entropy_act,  cross_entropy);










    let mut cross_entropy_grad = RayBNN_Optimizer::Continuous::Loss::sigmoid_cross_entropy_grad(&Yhat,&Y);


    let mut cross_entropy_grad_act_cpu:Vec<f64> = vec![-0.090033200537504  , 0.075508133759629 , -0.057810099474999 , -0.099000033332000 ,  0.095004162504212];
    let mut cross_entropy_grad_cpu = vec!(f64::default();cross_entropy_grad.elements());


    cross_entropy_grad.host(&mut cross_entropy_grad_cpu);

    cross_entropy_grad_cpu = cross_entropy_grad_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    cross_entropy_grad_act_cpu = cross_entropy_grad_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();



    assert_eq!(cross_entropy_grad_act_cpu, cross_entropy_grad_cpu);





}

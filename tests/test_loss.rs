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

    let test_cpu: Vec<f64> = vec![3.0, -2.0, 5.0, 8.0,       9.0, 5.0, -7.0, -3.0,       1.2, 3.5, -0.6, -0.8,       3.5, -2.3, 5.11, 8.4,        9.4, 5.9, -7.7, -3.40,       1.25, 3.5, -2.65, -3.8,];
    let test = arrayfire::Array::new(&test_cpu, arrayfire::Dim4::new(&[3, 4, 2, 1]));

    let mut mean = RayBNN_Optimizer::Continuous::Loss::mean_all(&test);

    let mut mean_cpu = vec!(f64::default();mean.elements());
    mean.host(&mut mean_cpu);

    assert_eq!((mean_cpu[0] * 10000000.0).round() / 10000000.0,    1.6045833);












    let Y_cpu: Vec<f64> = vec![0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0           ,0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0           ,0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0 ];
    let Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[7, 3, 1, 1]));


    let Yhat_cpu: Vec<f64> = vec![0.8,  0.3,  0.5,  0.2,  0.4,  0.1,  0.7        ,0.1,  0.2,  0.4,  0.2,  0.1,  0.6,  0.3           ,0.9,  0.2,  0.4,  0.2,  0.5,  0.6,  0.3 ];
    let Yhat = arrayfire::Array::new(&Yhat_cpu, arrayfire::Dim4::new(&[7, 3, 1, 1]));









    let mut cross_entropy = RayBNN_Optimizer::Continuous::Loss::softmax_cross_entropy(&Yhat,&Y);
    let (mut cross_entropy,_) = arrayfire::mean_all(&cross_entropy);
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
    let (mut cross_entropy,_) = arrayfire::mean_all(&cross_entropy);
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
































    let Y_cpu: [f64; 5] = [5.3, 4.3 , -3.2, -1.2, 2.1];
    let Y = arrayfire::Array::new(&Y_cpu, arrayfire::Dim4::new(&[1, 5, 1, 1]));


    let Yhat_cpu: [f64; 5] = [2.1, -0.2 , 1.0, -1.2, 9.0];
    let Yhat = arrayfire::Array::new(&Yhat_cpu, arrayfire::Dim4::new(&[1, 5, 1, 1]));











	let mut MAE = RayBNN_Optimizer::Continuous::Loss::MAE(&Yhat,&Y);
    let (mut MAE,_) = arrayfire::mean_all(&MAE);
    let mut MAE_act: f64 =  3.759999999999999;

    MAE = (MAE * 1.0e8).round() / 1.0e8;
    MAE_act = (MAE_act * 1.0e8).round() / 1.0e8;

    assert_eq!(MAE, MAE_act);










    let mut MSE = RayBNN_Optimizer::Continuous::Loss::MSE(&Yhat,&Y);
    let (mut MSE,_) = arrayfire::mean_all(&MSE);
    let mut MSE_act: f64 = 19.148000000000003;

    MSE = (MSE * 1.0e8).round() / 1.0e8;
    MSE_act = (MSE_act * 1.0e8).round() / 1.0e8;


    assert_eq!(MSE, MSE_act);













    let MSE_grad = RayBNN_Optimizer::Continuous::Loss::MSE_grad(&Yhat,&Y);


    let mut MSE_grad_act_cpu:Vec<f64> = vec![-1.280000000000000 , -1.800000000000000 ,  1.680000000000000 , 0.0 ,  2.760000000000000];
    let mut MSE_grad_out_cpu = vec!(f64::default();MSE_grad.elements());


    MSE_grad.host(&mut MSE_grad_out_cpu);

    MSE_grad_out_cpu = MSE_grad_out_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    MSE_grad_act_cpu = MSE_grad_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();



    assert_eq!(MSE_grad_act_cpu, MSE_grad_out_cpu);










    



	let mut RMSE = RayBNN_Optimizer::Continuous::Loss::RMSE(&Yhat,&Y);
    let (mut RMSE,_) = arrayfire::mean_all(&RMSE);
    let mut RMSE_act: f64 =  4.375842775968991;

    RMSE = (RMSE * 1.0e8).round() / 1.0e8;
    RMSE_act = (RMSE_act * 1.0e8).round() / 1.0e8;

    assert_eq!(RMSE, RMSE_act);















    let num = 11;
    let mut alpha_vec = RayBNN_Optimizer::Continuous::LR::create_alpha_vec::<f64>(num, 100.0, 0.5);
    assert_eq!(alpha_vec.len() as u64,num);

    alpha_vec = alpha_vec.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    let alpha_act = vec![100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.195312, 0.097656];
    assert_eq!(alpha_vec,alpha_act);

}

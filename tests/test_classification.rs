#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



const TEN: f64 = 10.0;

pub fn rscalar(
	input: f64,
	decimal: u64
	) -> f64  {

	let places = TEN.powf(decimal as f64);
	(input * places).round() / places
}



#[test]
fn test_classification() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);




    let y_cpu: Vec<u32> = vec![2, 0, 2, 2, 0, 1];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![0, 0, 2, 2, 0, 2];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));


    let confusion = RayBNN_Optimizer::Discrete::Classification::confusion_matrix(&yhat,&y,3);



    let mut confusion_cpu = vec!(u32::default();confusion.elements());


    confusion.host(&mut confusion_cpu);


    let mut confusion_act:Vec<u32> = vec![2, 0, 1, 0, 0, 0, 0, 1, 2];


    assert_eq!(confusion_act, confusion_cpu);
















    let y_cpu: Vec<u32> = vec![2,1,3,1,3,1,0,3,2,2,0,0,1,3,3,1];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![2,3,3,1,2,1,0,3,0,2,0,0,1,2,2,3];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));

    let confusion = RayBNN_Optimizer::Discrete::Classification::confusion_matrix(&yhat,&y,4);

    let mut confusion_cpu = vec!(u32::default();confusion.elements());


    confusion.host(&mut confusion_cpu);


    let mut confusion_act:Vec<u32> = vec![3, 0, 1, 0, 0, 3, 0, 0, 0, 0, 2, 3, 0, 2, 0, 2];


    assert_eq!(confusion_act, confusion_cpu);

















    //TP = 6     TN = 3     FP = 1      FN = 2
    let y_cpu: Vec<u32> =    vec![1,1,1,1,1,1,1,1,0,0,0,0];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![0,0,1,1,1,1,1,1,0,0,0,1];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));


    let result = RayBNN_Optimizer::Discrete::Classification::precision_recall_f1_MCC_binary(&yhat,&y);

    let mut result_cpu = vec!(f64::default();result.elements());
    result.host(&mut result_cpu);

    let precision = result_cpu[0];
    let recall = result_cpu[1];
    let f1 = result_cpu[2];
    let MCC = result_cpu[3];


    assert_eq!( rscalar(precision,6) ,  rscalar(0.8571428571428571,6) );
    assert_eq!( rscalar(recall,6) ,  rscalar(0.75,6) );
    assert_eq!( rscalar(f1,6) ,  rscalar(0.7999999999999999,6) );
    assert_eq!( rscalar(MCC,6) ,  rscalar(0.47809144373375745,6) );
























    let y_cpu: Vec<u32> =    vec![0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));


    let result = RayBNN_Optimizer::Discrete::Classification::precision_recall_f1_MCC_binary(&yhat,&y);


    let mut result_cpu = vec!(f64::default();result.elements());
    result.host(&mut result_cpu);

    let precision = result_cpu[0];
    let recall = result_cpu[1];
    let f1 = result_cpu[2];
    let MCC = result_cpu[3];

    assert_eq!( rscalar(precision,6) ,  rscalar(0.6,6) );
    assert_eq!( rscalar(recall,6) ,  rscalar(0.8571428571428571,6) );
    assert_eq!( rscalar(f1,6) ,  rscalar(0.7058823529411764,6) );
    assert_eq!( rscalar(MCC,6) ,  rscalar(0.37796447300922725,6) );





















    let y_cpu: Vec<u32> =    vec![2, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 0, 0, 1, 0, 2];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![2, 1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 2];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));

    let weights_cpu: Vec<f64> = vec![1.0, 1.0, 1.0];
	let mut weights = arrayfire::Array::new(&weights_cpu, arrayfire::Dim4::new(&[1, weights_cpu.len() as u64, 1, 1]));


    let result = RayBNN_Optimizer::Discrete::Classification::precision_recall_f1_MCC_multi(&yhat,&y,&weights,3);



    let mut result_cpu = vec!(f64::default();result.elements());
    result.host(&mut result_cpu);

    let precision = result_cpu[0];
    let recall = result_cpu[1];
    let f1 = result_cpu[2];
    let MCC = result_cpu[3];

    assert_eq!( rscalar(precision,6) ,  rscalar(0.6333333333333333,6) );
    assert_eq!( rscalar(recall,6) ,  rscalar(0.6222222222222222,6) );
    assert_eq!( rscalar(f1,6) ,  rscalar(0.6242424242424242,6) );
    assert_eq!( rscalar(MCC,6) ,  rscalar(0.441129,6) );



























    let y_cpu: Vec<u32> =    vec![0, 1, 2, 4, 4, 4, 1, 5, 2, 3, 0, 2, 1, 3, 0, 4, 5, 0, 3, 4, 1, 4,
    2, 3, 2, 3, 0, 5, 4, 5, 2, 5, 5, 5, 1, 2, 3, 1, 4, 1, 4, 4, 5, 0,
    4, 0, 1, 1, 2, 0];
    let y = arrayfire::Array::new(&y_cpu, arrayfire::Dim4::new(&[y_cpu.len() as u64, 1, 1, 1]));

    let yhat_cpu: Vec<u32> = vec![0, 1, 2, 4, 4, 2, 1, 1, 2, 3, 0, 2, 1, 3, 5, 4, 5, 0, 3, 4, 1, 2,
    2, 3, 2, 3, 0, 5, 1, 5, 2, 5, 5, 5, 1, 2, 4, 1, 4, 1, 4, 4, 0, 0,
    4, 0, 3, 1, 2, 0];
    let yhat = arrayfire::Array::new(&yhat_cpu, arrayfire::Dim4::new(&[yhat_cpu.len() as u64, 1, 1, 1]));

    let weights_cpu: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
	let mut weights = arrayfire::Array::new(&weights_cpu, arrayfire::Dim4::new(&[1, weights_cpu.len() as u64, 1, 1]));


    let result = RayBNN_Optimizer::Discrete::Classification::precision_recall_f1_MCC_multi(&yhat,&y,&weights,6);



    let mut result_cpu = vec!(f64::default();result.elements());
    result.host(&mut result_cpu);

    let precision = result_cpu[0];
    let recall = result_cpu[1];
    let f1 = result_cpu[2];
    let MCC = result_cpu[3];

    assert_eq!( rscalar(precision,6) ,  rscalar(0.84239417989418,6) );
    assert_eq!( rscalar(recall,6) ,  rscalar(0.8457491582491583,6) );
    assert_eq!( rscalar(f1,6) ,  rscalar(0.8398879142300194,6) );
    assert_eq!( rscalar(MCC,6) ,  rscalar(0.810891,6) );









}

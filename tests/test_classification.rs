#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



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



}

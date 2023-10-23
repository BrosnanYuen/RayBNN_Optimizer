use arrayfire;



const TWO_F64: f64 = 2.0;






pub fn confusion_matrix(
	yhat: &arrayfire::Array<u32>,
	y: &arrayfire::Array<u32>,
	label_num: u64) -> arrayfire::Array<u32> {

		let mut confusion = y + yhat*(label_num as u32);
		confusion = arrayfire::sort(&confusion, 0, true);


		let ones = arrayfire::constant::<u32>(1,confusion.dims());
		let  (keys, values) = arrayfire::sum_by_key(&confusion, &ones, 0);


		confusion = arrayfire::constant::<u32>(0,arrayfire::Dim4::new(&[label_num, label_num,1,1]));


		let mut idxrs = arrayfire::Indexer::default();
		idxrs.set_index(&keys, 0, None);
		arrayfire::assign_gen(&mut confusion, &idxrs, &values);


		confusion
}







pub fn precision_recall_f1_MCC_binary(
	yhat: &arrayfire::Array<u32>,
	y: &arrayfire::Array<u32>) -> arrayfire::Array<f64>
	{

	let con_matrix = confusion_matrix(
		yhat,
		y,
		2
    ).cast::<f64>();

	let mut con_matrix_cpu = vec!(f64::default();con_matrix.elements());

	con_matrix.host(&mut con_matrix_cpu);


	

	let TP = con_matrix_cpu[3];
	let FP = con_matrix_cpu[2];
	let FN = con_matrix_cpu[1];
	let TN = con_matrix_cpu[0];



	let P = TP/(TP + FP);
	let R = TP/(TP + FN);


	let result: Vec<f64> = vec![ P , R ,  TWO_F64*((P*R)/(P + R)),  (TP*TN  -  FP*FN)/( ( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ).sqrt()  ) ];



	arrayfire::Array::new(&result, arrayfire::Dim4::new(&[result.len() as u64, 1, 1, 1]))
}







pub fn precision_recall_f1_MCC_multi(
	yhat: &arrayfire::Array<u32>,
	y: &arrayfire::Array<u32>,
	weights: &arrayfire::Array<f64>,
	label_num: u64) -> arrayfire::Array<f64>
	{


	let mut count:u32 = 0;

	let mut temp_yhat = arrayfire::eq(yhat,&count,false).cast::<u32>();

	let mut temp_y = arrayfire::eq(y,&count,false).cast::<u32>();

	let mut result = precision_recall_f1_MCC_binary(
		&temp_yhat,
		&temp_y
    );



	count = count + 1;

	while count < (label_num as u32)
	{
		temp_yhat = arrayfire::eq(yhat,&count,false).cast::<u32>();

		temp_y = arrayfire::eq(y,&count,false).cast::<u32>();

		let temp_result = precision_recall_f1_MCC_binary(
            &temp_yhat,
            &temp_y
        );



		result = arrayfire::join(1, &result, &temp_result);

		count = count + 1;
	}


	arrayfire::mean_weighted(&result, weights, 1)
}








use arrayfire;





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






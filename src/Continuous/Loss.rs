use arrayfire;



pub fn softmax_cross_entropy<Z: arrayfire::FloatingPoint>(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> Z {
		let output_size = y.dims()[0];
		let batch_size = y.dims()[1];
		let batch_size_f64 = batch_size as f64;

		let mut yhatmax = arrayfire::max(yhat,0);
		yhatmax = arrayfire::transpose(&yhatmax, false);


		let (_,mut yidx) = arrayfire::imax(y,0);
		yidx = arrayfire::transpose(&yidx, false);


		let mut actmax = arrayfire::flat(yhat);

		let N_dims = arrayfire::Dim4::new(&[batch_size,1,1,1]);
		let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
		let offset = (output_size as u32)*arrayfire::iota::<u32>(N_dims,repeat_dims);

		yidx = yidx + offset;
		actmax = arrayfire::lookup(&actmax, &yidx, 0);


		let diff = yhatmax -  actmax;

		let (r0,_) = arrayfire::sum_all::<f64>(&diff);

		(one/batch_size_f64)*( r0 )
}










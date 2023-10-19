use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;









pub fn softmax_cross_entropy_grad(
	yhat: &arrayfire::Array<f64>,
	y: &arrayfire::Array<f64>) -> arrayfire::Array<f64> {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
	
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();


		let output_size = y.dims()[0];
		let batch_size = y.dims()[1];
		let batch_size_f64 = batch_size as f64;


		let mut expyhat = arrayfire::exp(yhat);
		expyhat = arrayfire::clamp(&expyhat, &ZERO, &high, false);

		let mut sumyhat = arrayfire::sum(&expyhat,0);
		sumyhat = arrayfire::clamp(&sumyhat, &ZERO, &high, false);

		expyhat = arrayfire::div(&expyhat,&sumyhat, true);


		(ONE/batch_size_f64)*( expyhat - y )
}










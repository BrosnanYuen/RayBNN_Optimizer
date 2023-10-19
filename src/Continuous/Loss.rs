use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;



const HIGH_F64: f64 = 1000000.0;





pub fn softmax_cross_entropy<Z: arrayfire::FloatingPoint >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> f64 {
		let output_size = y.dims()[0];
		let batch_size = y.dims()[1];


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


		let diff = yhatmax.cast::<Z>() -  actmax;

		let (r0,_) = arrayfire::mean_all(&diff);

		r0
}





pub fn softmax_cross_entropy_grad<Z: arrayfire::FloatingPoint<UnaryOutType = Z, AggregateOutType = Z>  >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();

		let HIGH = arrayfire::constant::<f64>(HIGH_F64,single_dims).cast::<Z>();




		let batch_size = y.dims()[1];
		let batch_size_f64 = batch_size as f64;

		let ONEdivbatch_size = arrayfire::constant::<f64>(ONE_F64/batch_size_f64,single_dims).cast::<Z>();

		let mut expyhat = arrayfire::exp(yhat);
		expyhat = arrayfire::clamp(&expyhat, &ZERO, &HIGH, false);

		let mut sumyhat = arrayfire::sum(&expyhat,0);
		sumyhat = arrayfire::clamp(&sumyhat, &ZERO, &HIGH, false);

		expyhat = arrayfire::div(&expyhat,&sumyhat, true);


		(ONEdivbatch_size)*( expyhat - y )
}










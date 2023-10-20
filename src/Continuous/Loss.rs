use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;



const HIGH_F64: f64 = 1000000.0;

const EPS_F64: f64 = 1.0e-13;
const EPS2_F64: f64 = 2.0e-13;



pub fn mean_all<Z: arrayfire::FloatingPoint<MeanOutType = Z> >(
	input: &arrayfire::Array<Z>) -> arrayfire::Array<Z>
	{

	let mut arr = arrayfire::mean(input,2);
	arr = arrayfire::mean(&arr,1);
	arr = arrayfire::mean(&arr,0);


	arr
}






pub fn softmax_cross_entropy<Z: arrayfire::FloatingPoint<InType = Z,MeanOutType = Z> >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {
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


		let diff = yhatmax -  actmax;

		let r0 = mean_all(&diff);

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









pub fn sigmoid_cross_entropy<Z: arrayfire::FloatingPoint<AbsOutType = Z, UnaryOutType = Z, MeanOutType = Z> >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
	
		let EPS = arrayfire::constant::<f64>(EPS_F64,single_dims).cast::<Z>();
		let EPS2 = arrayfire::constant::<f64>(EPS2_F64,single_dims).cast::<Z>();
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();


		let minus = ONE.clone() - y.clone();
		let sigmoid = arrayfire::sigmoid(yhat) + EPS;
		let logsigmoid = arrayfire::log(&sigmoid);
		let minussigmoid = ONE - sigmoid + EPS2;
		let logminus = arrayfire::log(&minussigmoid);

		let total = ZERO-( arrayfire::mul(y, &logsigmoid, false) + arrayfire::mul(&minus, &logminus, false)  );
		let r0 = mean_all(&total);
		r0
}















pub fn weighted_sigmoid_cross_entropy<Z: arrayfire::FloatingPoint<AbsOutType = Z, UnaryOutType = Z, MeanOutType = Z>  >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>,
	weight: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
		let EPS = arrayfire::constant::<f64>(EPS_F64,single_dims).cast::<Z>();
		let EPS2 = arrayfire::constant::<f64>(EPS2_F64,single_dims).cast::<Z>();
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();




		let minus = ONE.clone() - y.clone();
		let sigmoid = arrayfire::sigmoid(yhat) + EPS;
		let logsigmoid = arrayfire::log(&sigmoid);
		let minussigmoid = ONE - sigmoid + EPS2;
		let logminus = arrayfire::log(&minussigmoid);

		let total = ZERO-( (weight*arrayfire::mul(y, &logsigmoid, false)) + arrayfire::mul(&minus, &logminus, false)  );
		let r0 = mean_all(&total);
		r0
}










pub fn sigmoid_cross_entropy_grad<Z: arrayfire::FloatingPoint<AbsOutType = Z, UnaryOutType = Z>  >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z>  {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();


		let minus = ONE - y.clone();

		let yhatneg = ZERO.clone() - yhat.clone();
		let p0 = arrayfire::sigmoid(&yhatneg);
		let p1 = ZERO-arrayfire::sigmoid(&yhat);

		let size: f64 = yhat.elements() as f64;
		let ONEdivSIZE = arrayfire::constant::<f64>(-ONE_F64/size,single_dims).cast::<Z>();


		(ONEdivSIZE)*( arrayfire::mul(y, &p0, false)  +    arrayfire::mul(&minus, &p1, false)    )
}
















pub fn weighted_sigmoid_cross_entropy_grad<Z: arrayfire::FloatingPoint<AbsOutType = Z, UnaryOutType = Z>  >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>,
	weight: &arrayfire::Array<Z>) -> arrayfire::Array<Z>  {


		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
		let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();




		let minus = ONE - y.clone();


		let yhatneg = ZERO.clone() - yhat.clone();
		let p0 = arrayfire::sigmoid(&yhatneg);
		let p1 = ZERO -arrayfire::sigmoid(&yhat);


		let size: f64 = yhat.elements() as f64;
		let onedivsize = arrayfire::constant::<f64>(-ONE_F64/size,single_dims).cast::<Z>();

		(onedivsize)*( (weight*arrayfire::mul(y, &p0, false))  +    arrayfire::mul(&minus, &p1, false)    )
}








pub fn MAE<Z: arrayfire::FloatingPoint<AbsOutType = Z,MeanOutType = Z> >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {
		let mut diff = yhat.clone() - y.clone();

		diff = arrayfire::abs(&diff);

		let r0 =  mean_all(&diff);

		r0 
}









pub fn MSE<Z: arrayfire::FloatingPoint<MeanOutType = Z> >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	
		let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();



		let diff = yhat.clone() - y.clone();
		

		let diff = arrayfire::pow(&diff,&TWO,false);
		let r0 = mean_all(&diff);

		r0 
}








pub fn MSE_grad<Z: arrayfire::FloatingPoint>(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {
		let size: f64 = yhat.elements() as f64;

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	
		let twodivsize = arrayfire::constant::<f64>(TWO_F64/size,single_dims).cast::<Z>();

		(twodivsize)*(yhat.clone() - y.clone())
}



pub fn RMSE<Z: arrayfire::FloatingPoint<MeanOutType = Z, UnaryOutType = Z> >(
	yhat: &arrayfire::Array<Z>,
	y: &arrayfire::Array<Z>) -> arrayfire::Array<Z> {
		let MSE = MSE(yhat,y);
		arrayfire::sqrt(&MSE)
}




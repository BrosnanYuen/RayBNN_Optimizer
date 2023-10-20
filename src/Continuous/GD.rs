use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;







pub fn momentum(
	beta: f64
	,grad: &arrayfire::Array<f64>
	,dir: &mut arrayfire::Array<f64>)
	{

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
	


		*dir = (dir.clone()*beta)  + (one-beta)*(grad.clone());
}




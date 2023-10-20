use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;







pub fn momentum<Z: arrayfire::FloatingPoint>(
	beta: &arrayfire::Array<Z>
	,grad: &arrayfire::Array<Z>
	,dir: &mut arrayfire::Array<Z>)
	{

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
	


		*dir = (dir.clone()*beta)  + (ONE-beta)*(grad.clone());
}




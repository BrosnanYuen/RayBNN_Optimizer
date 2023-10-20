use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;







pub fn momentum(
	beta: f64
	,grad: &arrayfire::Array<f64>
	,dir: &mut arrayfire::Array<f64>)
	{
		*dir = (dir.clone()*beta)  + (one-beta)*(grad.clone());
}




use arrayfire;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPS_F64: f64 = 1.0e-13;
const EPS2_F64: f64 = 2.0e-13;




pub fn adam<Z: arrayfire::FloatingPoint<UnaryOutType = Z> >(
	beta0: &arrayfire::Array<Z>
	,beta1: &arrayfire::Array<Z>
	,direction: &mut arrayfire::Array<Z>
	,mt: &mut arrayfire::Array<Z>
	,vt: &mut arrayfire::Array<Z>)
	{

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
		let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();


		let EPS = arrayfire::constant::<f64>(EPS_F64,single_dims).cast::<Z>();


		*mt = (mt.clone())*beta0  + (ONE.clone()-beta0)*(direction.clone());
		*vt =  (vt.clone())*beta1  + (ONE.clone()-beta1)*arrayfire::pow(direction,&TWO,false);

		let nmt = mt.clone()/(ONE.clone()-beta0);
		let mut nvt = vt.clone()/(ONE-beta1);
		nvt = arrayfire::sqrt(&nvt) + EPS;

		*direction = (nmt/nvt);
}





pub fn momentum<Z: arrayfire::FloatingPoint>(
	beta: &arrayfire::Array<Z>
	,grad: &arrayfire::Array<Z>
	,dir: &mut arrayfire::Array<Z>)
	{

		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

		let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();
	


		*dir = (dir.clone()*beta)  + (ONE-beta)*(grad.clone());
}




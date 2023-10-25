use arrayfire;


const ONEHALF_F64: f64 = 0.5;

const ONE_F64: f64 = 1.0;



pub fn create_alpha_vec<Z: arrayfire::FloatingPoint >(
	num: u64,
	alpha: f64,
	gamma: f64,
	) -> Vec<Z>
{

	let N_dims = arrayfire::Dim4::new(&[num,1,1,1]);
    let mut alpha_arr = arrayfire::constant::<f64>(gamma,N_dims);

	let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let exponent = arrayfire::iota::<f64>(N_dims,repeat_dims);

	let multiplier = arrayfire::constant::<f64>(alpha,repeat_dims);

	alpha_arr = multiplier*arrayfire::pow(&alpha_arr,&exponent, false);


	let alpha_arr = alpha_arr.cast::<Z>();

	let mut alpha_vec = vec!(Z::default();alpha_arr.elements());
	alpha_arr.host(&mut alpha_vec);


	alpha_vec
}






pub fn BTLS<Z: arrayfire::FloatingPoint<AggregateOutType = Z> + arrayfire::ConstGenerator<OutType = Z> >(
	loss: impl Fn(&arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,loss_grad: impl Fn(&arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,init_point: &arrayfire::Array<Z>
	,direction: &arrayfire::Array<Z>
	,alpha_vec: &Vec<Z>
	,rho: &arrayfire::Array<Z>
    ,alpha: &mut arrayfire::Array<Z>)
	{
		let init_loss = loss(init_point);

		let mut next_point = init_point.clone() + ((*alpha).clone())*direction.clone();
		let mut f0  = loss(&next_point);

		let init_grad = loss_grad(init_point);
		let v0 = rho*(arrayfire::mul(direction, &init_grad, false));
		//let (t0,t1) = arrayfire::sum_all(&v0);
		let mut t0 = arrayfire::sum(&v0, 2);
        t0 = arrayfire::sum(&t0, 1);
        t0 = arrayfire::sum(&t0, 0);
        
        let mut f1  = init_loss.clone() + ((*alpha).clone())*t0.clone();


        let mut ret = arrayfire::gt(&f0,&f1, false);

		let mut ret_cpu = vec!(u8::default();ret.elements());
		ret.host(&mut ret_cpu);



		let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
		let alpha_size = alpha_vec.len();
		

		for i in 0..alpha_size
		{
			(*alpha) = arrayfire::constant(alpha_vec[i], single_dims);
			next_point = init_point.clone() + (alpha.clone())*direction.clone();
			f0  = loss(&next_point);
			f1  = init_loss.clone() + (alpha.clone())*t0.clone();


            ret = arrayfire::gt(&f0,&f1, false);
            ret.host(&mut ret_cpu);
			if ret_cpu[0] == 0
			{
				break;
			}
		}
		


}








pub fn cosine_annealing<Z: arrayfire::FloatingPoint >(
	start_epoch: u64,
	window_epoch: f64,
	min_alpha: f64,
	max_alpha: f64,

	counter: &mut u64,
	alpha: &mut arrayfire::Array<Z>)
{

	*counter =   (*counter) + 1;

	let tmp_counter = (*counter).clone();
	
	let mut alpha_cpu = min_alpha;
	if tmp_counter  >  start_epoch
	{
		alpha_cpu =  min_alpha  +   (ONEHALF_F64*(max_alpha - min_alpha)*(ONE_F64 +   (  ( (tmp_counter as f64) / window_epoch)*std::f64::consts::PI  ).cos())   );
	}

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	*alpha  = arrayfire::constant(alpha_cpu, single_dims).cast::<Z>();
}












pub fn plateau(
	loss_val: f64
	,control_state: &mut neural_controller_type
	,alpha0: &mut f64
	,alpha1: &mut f64)
{

	(*control_state).mean_loss = 0.9*((*control_state).mean_loss) + 0.1*loss_val;

	
	if ((*control_state).mean_loss*1.05 < (*control_state).min_loss)
	{
		(*control_state).min_loss = (*control_state).mean_loss;
		(*control_state).counter0 = 0;
	}




	if ((*control_state).counter0  > (*control_state).window_epoch)
	{
		*alpha0 =  (*alpha0)*((*control_state).decrease_alpha);
		*alpha1 =  (*alpha1)*((*control_state).decrease_alpha);
		(*control_state).counter0 = 0;
	}
	else
	{
		(*control_state).counter0 = (*control_state).counter0  + 1;
	}



	
	if (*alpha0 <  (*control_state).min_alpha) && (*alpha1 > 0.0)
	{
		*alpha0 = (*control_state).max_alpha;
		*alpha1 =  0.0;
	}
	

}








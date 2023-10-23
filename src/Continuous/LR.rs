use arrayfire;







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
	window_epoch: u64,
	min_alpha: &arrayfire::Array<Z>,
	max_alpha: &arrayfire::Array<Z>,

	counter: &mut u64,
	alpha: &mut arrayfire::Array<Z>)
{

	*counter =   (*counter) + 1;

	if (*counter)  >  start_epoch
	{
		*alpha  =    min_alpha  +   (onehalf*(max_alpha - min_alpha)*(one +   (  ( ((*control_state).counter1 as f64) / (window_epoch as f64))*std::f64::consts::PI  ).cos())   );
	}
	else
	{
		*alpha  =  min_alpha.clone();
	}


}











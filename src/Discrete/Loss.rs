use arrayfire;










pub fn BTLS<Z: arrayfire::FloatingPoint<AggregateOutType = Z> >(
	loss: impl Fn(&arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,loss_grad: impl Fn(&arrayfire::Array<Z>) -> arrayfire::Array<Z>
	,init_point: &arrayfire::Array<Z>
	,direction: &arrayfire::Array<Z>
	,gamma: &arrayfire::Array<Z>
	,rho: &arrayfire::Array<Z>
    ,alpha_max: &arrayfire::Array<Z>) -> arrayfire::Array<Z>
	{
		let mut alpha = alpha_max.clone();
		let init_loss = loss(init_point);

		let mut next_point = init_point.clone() + (alpha)*direction.clone();
		let mut f0  = loss(&next_point);

		let init_grad = loss_grad(init_point);
		let v0 = rho*(arrayfire::mul(direction, &init_grad, false));
		//let (t0,t1) = arrayfire::sum_all(&v0);
		let mut t0 = arrayfire::sum(&v0, 2);
        t0 = arrayfire::sum(&t0, 1);
        t0 = arrayfire::sum(&t0, 0);
        
        let mut f1  = init_loss.clone() + (alpha)*t0;


        let mut ret = arrayfire::gt(&f0,&f1, false);
        let mut ret_cpu = arrayfire::any_true_all(&ret).0;
    


		while (ret_cpu)
		{
			alpha = (alpha)*gamma;
			next_point = init_point.clone() + (alpha)*direction.clone();
			f0  = loss(&next_point);
			f1  = init_loss.clone() + (alpha)*t0;
		}
		alpha
}






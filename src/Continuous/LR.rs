use arrayfire;









pub fn BTLS<Z: arrayfire::FloatingPoint>(
	loss: impl Fn(&arrayfire::Array<f64>) -> f64
	,loss_grad: impl Fn(&arrayfire::Array<f64>) -> arrayfire::Array<f64>
	,init_point: &arrayfire::Array<f64>
	,direction: &arrayfire::Array<f64>
	,gamma: f64
	,rho: f64) -> f64
	{
		let mut alpha: f64 = LR_MAX;
		let init_loss = loss(init_point);

		let mut next_point = init_point.clone() + (alpha)*direction.clone();
		let mut f0  = loss(&next_point);

		let init_grad = loss_grad(init_point);
		let v0 = rho*(arrayfire::mul(direction, &init_grad, false));
		let (t0,t1) = arrayfire::sum_all(&v0);
		let mut f1  = init_loss.clone() + (alpha)*t0;

		while (f0 > f1)
		{
			alpha = (alpha)*gamma;
			next_point = init_point.clone() + (alpha)*direction.clone();
			f0  = loss(&next_point);
			f1  = init_loss.clone() + (alpha)*t0;
		}
		alpha
}





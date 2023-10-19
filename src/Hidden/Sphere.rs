use arrayfire;

use std::collections::HashMap;


use crate::Util::Math::set_diag;

const TWO_F64: f64 = 2.0;

const NEURON_RAD_FACTOR: f64 = 1.1;

const HIGH_F64: f64 = f64::INFINITY;

const ONEHALF_F64: f64 = 0.5;

const TARGET_DENSITY: f64 = 3500.0;






pub fn get_inside_idx_cubeV2<Z: arrayfire::FloatingPoint>(
	pos: &arrayfire::Array<Z>
	, cube_size: f64
	, pivot: &Vec<f64>)
	-> arrayfire::Array<u32>
{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let pivot_pos = pivot.clone();
	let space_dims = pivot_pos.len();





	//let mut negative_range = pivot_pos[0].clone();
	//let mut positive_range = negative_range + cube_size;

	let mut negative_range = arrayfire::constant::<f64>(pivot_pos[0].clone(),single_dims).cast::<Z>();
	let mut positive_range = arrayfire::constant::<f64>(pivot_pos[0].clone() + cube_size,single_dims).cast::<Z>();


	let mut axis = arrayfire::col(pos,0);

	let mut cmp1 = arrayfire::lt(&axis, &positive_range, false);
	let mut cmp2 = arrayfire::lt(&negative_range,  &axis, false);
	cmp1 = arrayfire::and(&cmp1,&cmp2, false);

	for idx in 1..space_dims
	{
		//negative_range = pivot_pos[idx].clone();
		//positive_range = negative_range + cube_size;


		negative_range = arrayfire::constant::<f64>(pivot_pos[idx].clone(),single_dims).cast::<Z>();
		positive_range = arrayfire::constant::<f64>(pivot_pos[idx].clone() + cube_size,single_dims).cast::<Z>();
	

	
		axis = arrayfire::col(pos,idx as i64);

		cmp2 = arrayfire::lt(&axis, &positive_range, false);
		cmp1 = arrayfire::and(&cmp1,&cmp2, false);
		cmp2 = arrayfire::lt(&negative_range,  &axis, false);
		cmp1 = arrayfire::and(&cmp1,&cmp2, false);
	
	}

	arrayfire::locate(&cmp1)
}






pub fn select_non_overlap<Z: arrayfire::FloatingPoint<AggregateOutType = Z>  >(
	pos: &arrayfire::Array<Z>,
	neuron_rad: f64
) -> arrayfire::Array<u32>
{

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let HIGH = arrayfire::constant::<f64>(HIGH_F64,single_dims).cast::<Z>();



	let mut p1 = pos.clone();

	p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));

	let mut magsq = arrayfire::sub(&p1, pos, true);
	drop(p1);
	magsq = arrayfire::pow(&magsq,&TWO,false);

	magsq = arrayfire::sum(&magsq,1);


	set_diag(
		&mut magsq,
		&HIGH
	);

	let neuron_sq: f64 = 4.0*neuron_rad*neuron_rad*NEURON_RAD_FACTOR;

	let neuron_sq_Z = arrayfire::constant::<f64>(neuron_sq,single_dims).cast::<Z>();


	//Select close objects
	let mut cmp = arrayfire::lt(&magsq , &neuron_sq_Z, false);
	drop(magsq);
		

	cmp = arrayfire::any_true(&cmp, 2);
	//Lookup  1 >= dir_line  >= 0
	arrayfire::locate(&cmp)
}










pub fn generate_uniform_sphere_posiiton<Z: arrayfire::FloatingPoint<UnaryOutType = Z> >(
    modeldata_float: &HashMap<String, f64>,
    modeldata_int: &HashMap<String, u64>
	) -> arrayfire::Array<Z>
	{



	let active_size: u64 = modeldata_int["active_size"].clone();





	let sphere_rad: f64 = modeldata_float["sphere_rad"].clone();
	let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();



	let generate_dims = arrayfire::Dim4::new(&[2*active_size,1,1,1]);
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);



	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let ONEHALF = arrayfire::constant::<f64>(ONEHALF_F64,single_dims).cast::<Z>();

	let TWO_PI = arrayfire::constant::<f64>(TWO_F64*std::f64::consts::PI,single_dims).cast::<Z>();

	let sphere_rad_Z = arrayfire::constant::<f64>(sphere_rad-neuron_rad,single_dims).cast::<Z>();


	
	let mut r = arrayfire::randu::<Z>(generate_dims);
	r = (sphere_rad_Z)*arrayfire::cbrt(&r).cast::<Z>();
	let mut theta = TWO*(arrayfire::randu::<Z>(generate_dims)-ONEHALF);
	theta = arrayfire::acos(&theta);
	let mut phi = TWO_PI*arrayfire::randu::<Z>(generate_dims);
	

	

	let x = r.clone()*arrayfire::sin(&theta)*arrayfire::cos(&phi);
	let y = r.clone()*arrayfire::sin(&theta)*arrayfire::sin(&phi);
	let z = r.clone()*arrayfire::cos(&theta);

	drop(r);
	drop(theta);
	drop(phi);

	arrayfire::join_many(1, vec![&x,&y,&z])

}










/*
Detects cell collisions in minibatch, where groups/minibatches of cells are checked

Inputs
modeldata_float:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created
cell_pos:  The 3D position of neurons in the shape of a 3D sphere

Outputs
Indicies of non colliding cells
*/

pub fn check_cell_collision_minibatch<Z: arrayfire::FloatingPoint<AggregateOutType = Z, UnaryOutType = Z> >(
    modeldata_float: &HashMap<String, f64>,


	cell_pos: &arrayfire::Array<Z>) -> arrayfire::Array<bool>
	{



	let space_dims: u64 = cell_pos.dims()[1];




	let sphere_rad: f64 = modeldata_float["sphere_rad"].clone();
	let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();


	

	let mut pivot_rad = ((4.0/3.0)*std::f64::consts::PI*TARGET_DENSITY*sphere_rad*sphere_rad*sphere_rad);
	pivot_rad = (pivot_rad/((cell_pos.dims()[0]) as f64)).cbrt();

	let pivot_rad2 = pivot_rad + (2.05f64*neuron_rad*NEURON_RAD_FACTOR);

	let mut loop_end_flag = false;
	let mut pivot_pos = vec![-sphere_rad; space_dims as usize];




	let select_idx_dims = arrayfire::Dim4::new(&[cell_pos.dims()[0],1,1,1]);
	let mut select_idx = arrayfire::constant::<bool>(true,select_idx_dims);

	loop 
	{

		let idx = get_inside_idx_cubeV2(
			&cell_pos
			, pivot_rad2
			, &pivot_pos
		);

		
		if idx.dims()[0] > 1
		{
			let tmp_obj = arrayfire::lookup(&cell_pos, &idx, 0);

			let mut neg_idx = select_non_overlap(
				&tmp_obj,
				neuron_rad
			);
	

			if neg_idx.dims()[0] > 0
			{
				neg_idx = arrayfire::lookup(&idx, &neg_idx, 0);

				let insert = arrayfire::constant::<bool>(false,neg_idx.dims());

				let mut idxrs = arrayfire::Indexer::default();
				idxrs.set_index(&neg_idx, 0, None);
				arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
			}

			
		}
		drop(idx);


		pivot_pos[0] = pivot_pos[0] + pivot_rad;

		for idx in 0..space_dims
		{
			if pivot_pos[idx as usize] > sphere_rad
			{
				if idx == (space_dims-1)
				{
					loop_end_flag = true;
					break;
				}

				pivot_pos[idx as usize] = -sphere_rad;
				pivot_pos[(idx+1) as usize] = pivot_pos[(idx+1) as usize] + pivot_rad;
			}
		}

		if loop_end_flag
		{
			break;
		}
	}
	

	select_idx
}






pub fn split_into_glia_neuron<Z: arrayfire::FloatingPoint >(
    modeldata_float: &HashMap<String, f64>,

	cell_pos: &arrayfire::Array<Z>,

	glia_pos: &mut arrayfire::Array<Z>,
	neuron_pos: &mut arrayfire::Array<Z>)
	{


	let nratio: f64 = modeldata_float["nratio"].clone();



	let total_obj_size = cell_pos.dims()[0];

	let split_idx = ((total_obj_size as f64)*nratio) as u64;

	*neuron_pos = arrayfire::rows(&cell_pos, 0, (split_idx-1)  as i64);
	
	*glia_pos = arrayfire::rows(&cell_pos, split_idx  as i64, (total_obj_size-1)  as i64);

}






















/*
Detects cell collisions in batch, where all cells are checked at once

Inputs
modeldata_float:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created
cell_pos:  The 3D position of neurons in the shape of a 3D sphere

Outputs
Indicies of non colliding cells
*/

pub fn check_cell_collision_batch<Z: arrayfire::FloatingPoint<AggregateOutType = Z, UnaryOutType = Z> >(
    modeldata_float: &HashMap<String, f64>,


	cell_pos: &arrayfire::Array<Z>) -> arrayfire::Array<bool>
	{





	let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();


	




	let select_idx_dims = arrayfire::Dim4::new(&[cell_pos.dims()[0],1,1,1]);
	let mut select_idx = arrayfire::constant::<bool>(true,select_idx_dims);




	let mut neg_idx = select_non_overlap(
		&cell_pos,
		neuron_rad
	);


	if neg_idx.dims()[0] > 0
	{

		let insert = arrayfire::constant::<bool>(false,neg_idx.dims());

		let mut idxrs = arrayfire::Indexer::default();
		idxrs.set_index(&neg_idx, 0, None);
		arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
	}

	


	select_idx
}















/*
Detects cell collisions in serial, where cells are checked one by one

Inputs
modeldata_float:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created
cell_pos:  The 3D position of neurons in the shape of a 3D sphere

Outputs
Indicies of non colliding cells
*/

pub fn check_cell_collision_serial<Z: arrayfire::FloatingPoint<AggregateOutType = Z, UnaryOutType = Z> >(
    modeldata_float: &HashMap<String, f64>,


	cell_pos: &arrayfire::Array<Z>) -> arrayfire::Array<bool>
	{





	let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();


	


	let mut select_idx = vec![true; cell_pos.dims()[0] as usize];





	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();


	let neuron_sq = (4.0*neuron_rad*neuron_rad) as f32;

	for i in 0u64..cell_pos.dims()[0]
	{
		let select_pos = arrayfire::row(&cell_pos,i as i64);

		let mut dist = arrayfire::sub(&select_pos,cell_pos, true);
		let mut magsq = arrayfire::pow(&dist,&TWO,false);
		let mut magsq = arrayfire::sum(&magsq,1);


		let mut magsq = magsq.cast::<f32>();
		let insert = arrayfire::constant::<f32>(1000000.0,single_dims);

		arrayfire::set_row(&mut magsq, &insert, i as i64);

		let (m0,_) = arrayfire::min_all::<f32>(&magsq);


		if m0 < neuron_sq
		{
			select_idx[i as usize] = false;
		}
	}



	


	let select_idx = arrayfire::Array::new(&select_idx, arrayfire::Dim4::new(&[cell_pos.dims()[0]  , 1, 1, 1]));

	select_idx
}





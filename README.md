# RayBNN_Cell

Cell posiition generation library for RayBNN using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI 



# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.7.0"
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
RayBNN_DataLoader = "0.1.3"
RayBNN_Cell = "0.1.2"
```

# List of Examples


# Generate Cells and Check them for Collisions
```

//Generate Random Uniform Cells within a Sphere
let mut cell_pos: arrayfire::Array<f32>  = RayBNN_Cell::Hidden::Sphere::generate_uniform_sphere_posiiton(&modeldata_float, &modeldata_int);


//Get indicies of non colliding cells
let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_minibatch(
    &modeldata_float, 
    &cell_pos
);

let idx = arrayfire::locate(&idx);

//Select non colliding cells
cell_pos = arrayfire::lookup(&cell_pos, &idx, 0);

```

# Types of cell collision checkers
```
//Checks One by one
let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_serial(
    &modeldata_float, 
    &cell_pos
);

//Checks All cells at once
let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_batch(
    &modeldata_float, 
    &cell_pos
);

//Checks in minibatches
let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_minibatch(
    &modeldata_float, 
    &cell_pos
);

```


# Generate Input Neuron Positions

```
let input_neurons: arrayfire::Array<f64> = RayBNN_Cell::Input::Sphere::create_spaced_neurons_1D(
    sphere_rad,
    input_size,
);
```




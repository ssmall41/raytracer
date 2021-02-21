#define NEGINF	-1e8f

typedef float (*collision_func)(float ray_x, float ray_y, float* params, float* scaling);
__device__ float sphere_collision(float ray_x, float ray_y, float* params, float* scaling);

__device__ collision_func collision_functions[] {sphere_collision};
__device__ collision_func next_collision_func;


//params = {r, g, b, x, y, z, radius}
__device__ float sphere_collision(float ray_x, float ray_y, float* params, float* scaling)
{
	//Unpack the parameters
	float sph_x = params[3], sph_y = params[4], sph_z = params[5];
	float sph_r = params[6];

	//if(sph_x > 150)
	//	printf("Got %f %f %f %f\n", sph_x, sph_y, sph_z, sph_r);

	float dx = ray_x - sph_x;
	float dy = ray_y - sph_y;
	float dz;
	float r2 = sph_r * sph_r;
	
	if(dx*dx + dy*dy < r2)
	{
		dz = sqrtf(r2 - dx*dx - dy*dy);
		*scaling = dz / sqrtf(r2);
		return dz + sph_z;
	}
	return NEGINF;
}


__device__ void find_closest(int ray_x, int ray_y, float* object_params, unsigned int* n_objs, int n_params, unsigned int n_types, 
			      int* closest_type, int* closest_obj, float* closest_scaling)
{
	int t, i, curr_obj=0;
	float curr_scaling;
	float closest_distance = NEGINF;
	*closest_type = -1;
	*closest_obj = -1;  //The same as the row in object_params of the closest object
	*closest_scaling = 0.0;

	for(t=0;t<n_types;t++)
	{
		next_collision_func = collision_functions[t];
	
		for(i=0;i<n_objs[t];i++)
		{
			//printf("Checking obj %i\n", i);
			//printf("param size is %i\n", n_params);
		
			float* params = &(object_params[curr_obj*n_params]);
			float distance = next_collision_func(ray_x, ray_y, params, &curr_scaling);

			//printf("distance: %f\n", distance);

			if(distance > closest_distance)
			{
				closest_distance = distance;
				*closest_type = t;
				*closest_obj = curr_obj;
				*closest_scaling = curr_scaling;
			}
			
			curr_obj += 1;
		}
	}
}


__global__ void draw_scene(int* bitmap, unsigned int dim, float* object_params, 
			    unsigned int* n_objs, int n_params, int n_types)
{

	int ray_x = threadIdx.x + blockIdx.x * blockDim.x;
	int ray_y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = ray_x + ray_y * blockDim.x * gridDim.x;
	float closest_scaling;
	int closest_type, closest_obj;
	
	//printf("%i %i\n", ray_x, ray_y);
	
	//Find the closest object
	find_closest(ray_x, ray_y, object_params, n_objs, n_params, n_types, &closest_type, &closest_obj, &closest_scaling);

	//Find the color to display for the pixel
	float r=0.0, g=0.0, b=0.0;
	if(closest_type > -1)
	{
		float* params = &(object_params[closest_obj*n_params]);
		r = params[0] * closest_scaling;
		g = params[1] * closest_scaling;
		b = params[2] * closest_scaling;
	}
	
	// Convert from colors between 0-1 and 0-255
	bitmap[offset*3] = (int)(r*255);
	bitmap[offset*3 + 1] = (int)(g*255);
	bitmap[offset*3 + 2] = (int)(b*255);
}

__global__ void hello_world()
{
	printf("Hello world!\n");
}



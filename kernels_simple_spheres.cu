#define NEGINF	-1e8f

__device__ float sphere_collision(float ray_x, float ray_y, 
				   float sph_x, float sph_y, float sph_z, float sph_r, 
				   float* scaling)
{
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

__global__ void simple_spheres(int* bitmap, float* coords, float* radius, float* colors, 
				unsigned int dim, unsigned int n_spheres)
{

	int ray_x = threadIdx.x + blockIdx.x * blockDim.x;
	int ray_y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = ray_x + ray_y * blockDim.x * gridDim.x;
	int i;
	float sph_x, sph_y, sph_z, sph_r, scaling, distance;
	float closest_distance = NEGINF, closest_scaling = 0.0;
	int closest_sphere = -1;

	for(i=0;i<n_spheres;i++)
	{
		sph_x = coords[i*3];
		sph_y = coords[i*3+1];
		sph_z = coords[i*3+2];
		sph_r = radius[i];

		distance = sphere_collision(ray_x, ray_y, sph_x, sph_y, sph_z, sph_r, &scaling);
		if(distance > closest_distance)
		{
			closest_distance = distance;
			closest_sphere = i;
			closest_scaling = scaling;
		}
	}
	
	float r=0.0, g=0.0, b=0.0;
	i = closest_sphere;
	if(i > -1)
	{
		r = colors[i*3] * closest_scaling;
		g = colors[i*3+1] * closest_scaling;
		b = colors[i*3+2] * closest_scaling;
	}

	bitmap[offset*3] = (int)r;
	bitmap[offset*3 + 1] = (int)g;
	bitmap[offset*3 + 2] = (int)b;
	//bitmap[offset*4 + 3] = 255;
	
	/*
	bitmap[offset*3] = (int)(r*255);	//Why *255??
	bitmap[offset*3 + 1] = (int)(g*255);
	bitmap[offset*3 + 2] = (int)(b*255);
	*/
}




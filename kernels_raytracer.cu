#define INF 1e8f
#define EPS 1e-3

typedef float (*collision_func)(float[], float[], float*, float*, float*);
__device__ float sphere_collision(float[], float[], float*, float*, float*);
__device__ float plane_collision(float[], float[], float*, float*, float*);
__device__ float triangle_collision(float[], float[], float*, float*, float*);

__device__ collision_func collision_functions[] {sphere_collision, plane_collision, triangle_collision};


__device__ void print_vector(float x[], int size)
{
	int i;
	for(i=0;i<size;i++)
		printf("%f ", x[i]);
	printf("\n");
}

__device__ float vector_dot(float x[], float y[])
{
	int i, size=3;
	float res = x[0]*y[0];
	for(i=1;i<size;i++)
		res += x[i]*y[i];
	return res;
}

__device__ void vector_saxpy(float x[], float y[], float alpha, float res[])
{
	int i, size=3;
	for(i=0;i<size;i++)
		res[i] = alpha*x[i] + y[i];
}

__device__ void vector_diff(float x[], float y[], float res[])
{
	int i, size=3;
	for(i=0;i<size;i++)
		res[i] = x[i] - y[i];
}

__device__ float norm_squared(float x[])
{
	int i, size=3;
	float sum = 0.0;
	for(i=0;i<size;i++)
		sum += x[i]*x[i];
	return sum;
}

__device__ float vector_norm(float x[])
{
	int i, size=3;
	float sum = 0.0;
	for(i=0;i<size;i++)
		sum += x[i]*x[i];
	return sqrt(sum);
}

__device__ void vector_normalize(float x[])
{
	int i, size=3;
	float norm = vector_norm(x);
	for(i=0;i<size;i++)
		x[i] = x[i] / norm;
}

__device__ void vector_copy(float dest[], float src[], int size)
{
	int i;
	for(i=0;i<size;i++)
		dest[i] = src[i];
}

__device__ void vector_cross_product(float x[], float y[], float res[])
{
	res[0] = x[1]*y[2] - x[2]*y[1];
	res[1] = x[2]*y[0] - x[0]*y[2];
	res[2] = x[0]*y[1] - x[1]*y[0];
}

__device__ float triangle_area(float x[], float y[], float z[])
{
	float edge1[3], edge2[3], cp[3];
	vector_diff(y, x, edge1);
	vector_diff(z, x, edge2);
	vector_cross_product(edge1, edge2, cp);
	return 0.5 * vector_norm(cp);
}


//params = {r, g, b, x, y, z, radius}
__device__ float sphere_collision(float ray_orig[], float ray_dir[], float* params, float* intersect_point, 
				   float* intersect_normal)
{
	// Unpack the sphere parameters
	float* sphere_center = &(params[3]);
	float sph_r = params[6];
	float recenter[3];

	vector_diff(ray_orig, sphere_center, recenter);
	float dot_product = vector_dot(ray_dir, recenter);
	float norm_recenter = norm_squared(recenter);
	
	float delta = dot_product * dot_product - (norm_recenter - sph_r*sph_r);
	float distance_plus = -dot_product + sqrtf(delta);
	float distance_minus = -dot_product - sqrtf(delta);
	
	// Find the distance
	float d;
	if(delta < 0.0 || (distance_plus < 0.0 && distance_minus < 0.0))  // No solution
		d = INF;
	else if(delta < EPS)  // One solution
		d = distance_plus;
	else  // Multiple solutions, return the closest
	{
		if(distance_plus < EPS)  // If both points are behind us, return INF, otherwise distance_minus
			d = (distance_minus < EPS) ? INF : distance_minus;
		else  // If distance_negative is behind us, return distance_plus, otherwise return the nearest
			d = (distance_minus < EPS) ? distance_plus : fmin(distance_plus, distance_minus);
	}
	
	if(d < INF)
	{
		vector_saxpy(ray_dir, ray_orig, d, intersect_point);
		vector_diff(intersect_point, sphere_center, intersect_normal);
		vector_normalize(intersect_normal);
	}
	return d;
}


//params = {r, g, b, x, y, z, nx, ny, nz}
__device__ float plane_collision(float ray_orig[], float ray_dir[], float* params, float* intersect_point, 
				  float* intersect_normal)
{
	// Unpack the plane parameters
	float* point = &(params[3]);
	float* normal = &(params[6]);
	float difference[3];
	
	float denominator = vector_dot(ray_dir, normal);
	
	if(fabs(denominator) < EPS)
		return INF;
	else
	{
		vector_diff(point, ray_orig, difference);
		float numerator = vector_dot(difference, normal);
		float d = numerator / denominator;
		
		vector_saxpy(ray_dir, ray_orig, d, intersect_point);
		vector_copy(intersect_normal, normal, 3);
		
		return (d > EPS) ? d : INF;
	}
}


//params = {r, g, b, x1, y1, z1, x2, y2, z2, x3, y3, z3}
__device__ float triangle_collision(float ray_orig[], float ray_dir[], float* params, float* intersect_point, 
				     float* intersect_normal)
{
	// Unpack the parameters
	float* point1 = &(params[3]);
	float* point2 = &(params[6]);
	float* point3 = &(params[9]);
	
	// Find a unit normal to the triangle
	float edge1[3], edge2[3];
	vector_diff(point2, point1, edge1);
	vector_diff(point3, point1, edge2);
	vector_cross_product(edge1, edge2, intersect_normal);
	vector_normalize(intersect_normal);
	
	// Check where the ray hits the plane defined by the triangle
	float plane_params[9];
	vector_copy(plane_params, params, 6);
	vector_copy(&(plane_params[6]), intersect_normal, 3);
	float d = plane_collision(ray_orig, ray_dir, plane_params, intersect_point, intersect_normal);
	
	if(d < INF)
	{	
		float area = triangle_area(point1, point2, point3);
		float alpha = triangle_area(intersect_point, point2, point3) / area;
		float beta = triangle_area(intersect_point, point1, point3) / area;
		float gamma = triangle_area(intersect_point, point1, point2) / area;
		float sum = alpha + beta + gamma;
		
		if(alpha > -EPS && beta > -EPS && gamma > -EPS && fabs(sum - 1.0) < EPS)
			return d;
	}

	return INF;
}


__global__ void find_closest(float* ray_orig, float* ray_dir, int obj_type, float* object_params, int dim, int n_objs,
			      int n_params, float* closest_distance, float* closest_intersect_point, 
			      float* closest_intersect_normal, float* closest_color)
{
	int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
	int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
	if((pixel_x > dim) || (pixel_y > dim))	return;
	int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
	int i;
	float intersect_point[3], intersect_normal[3];
	float* this_pixel_intersection_point = &(closest_intersect_point[3*offset]);
	float* this_pixel_intersection_normal = &(closest_intersect_normal[3*offset]);
	float* this_ray_orig = &(ray_orig[3*offset]);
	float* this_ray_dir = &(ray_dir[3*offset]);
	float* this_closest_color = &(closest_color[3*offset]);

	for(i=0;i<n_objs;i++)
	{
		float* params = &(object_params[i*n_params]);
		float distance = collision_functions[obj_type](this_ray_orig, this_ray_dir, params, 
								 intersect_point, intersect_normal);

		if(distance < closest_distance[offset])
		{
			closest_distance[offset] = distance;
			vector_copy(this_pixel_intersection_point, intersect_point, 3);
			vector_copy(this_pixel_intersection_normal, intersect_normal, 3);
			vector_copy(this_closest_color, params, 3);
		}
	}
}

__global__ void find_ray_to_light(int dim, float* light, float* closest_intersect_point, float* ray_to_light, 
				   float* distance_to_light)
{
	int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
	int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
	if((pixel_x > dim) || (pixel_y > dim))	return;
	int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
	float* this_intersection_point = &(closest_intersect_point[3*offset]);
	float* this_ray_to_light = &(ray_to_light[3*offset]);
	
	// Get the vector pointing from the intersection point to the light
	vector_diff(light, this_intersection_point, this_ray_to_light);
	distance_to_light[offset] = vector_norm(this_ray_to_light);
	vector_normalize(this_ray_to_light);
}

__global__ void lambertian_shading(int* bitmap, int dim, float* closest_distance_light,
				    float* distance_to_light, float* closest_color,
				    float* closest_intersect_normal, float* ray_to_light)
{
	int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
	int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
	if((pixel_x > dim) || (pixel_y > dim))	return;
	int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
	float* this_closest_color = &(closest_color[3*offset]);
	float* this_closest_intersect_normal = &(closest_intersect_normal[3*offset]);
	float* this_ray_to_light = &(ray_to_light[3*offset]);

	// Find the color to display for the pixel
	float scaling=255.0, r=0.0, g=0.0, b=0.0;
	if(closest_distance_light[offset] > distance_to_light[offset])
	{
		r = this_closest_color[0];
		g = this_closest_color[1];
		b = this_closest_color[2];
		scaling = 255.0 * fabs(vector_dot(this_closest_intersect_normal, this_ray_to_light));
	}
	
	// Convert from colors between 0-1 and 0-255
	bitmap[offset*3] = (int)(r*scaling);
	bitmap[offset*3 + 1] = (int)(g*scaling);
	bitmap[offset*3 + 2] = (int)(b*scaling);
}


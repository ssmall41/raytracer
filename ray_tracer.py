import pycuda.autoinit
import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import cv2
import itertools
import pywavefront


class raytracer:
	
	def __init__(self, dim, grid_size):
		
		self.dim = np.uint32(dim)
		self.block = (dim//grid_size, dim//grid_size, 1)
		self.grid = (grid_size, grid_size)
		self.float = np.float32
		self.int = np.int32
		
		self.sphere_type = self.int(0)
		self.plane_type = self.int(1)
		self.triangle_type = self.int(2)
		self.n_object_types = 3
		self.n_object_params = np.array([7, 9, 12], dtype=self.int)
		self.required_params = ["r,g,b,x,y,z,r",
					 "r,g,b,x,y,z,nx,ny,nz",
					 "r,g,b,x1,y1,z1,x2,y2,z2,x3,y3,z3"]
		
		self.module = SourceModule(open("kernels_raytracer.cu", "r").read())
		self.kernel_find_closest = self.module.get_function("find_closest")
		self.kernel_find_ray_to_light = self.module.get_function("find_ray_to_light")
		self.kernel_lambertian_shading = self.module.get_function("lambertian_shading")
		
		self.object_params = self.n_object_types*[np.array([], dtype=self.float)]
		self.object_params_gpu = []
		self.n_objects = np.full((self.n_object_types, ), -1, dtype=self.int)
		
		self.light = None
		self.bitmap_gpu = None
		self.ray_orig_gpu = None
		self.ray_dir_gpu = None
	
	def add_object(self, params, obj_type):
		
		if obj_type > self.n_object_types:
			raise ValueError(f"Invalid object type {obj_type}")
		
		if len(params) != self.n_object_params[obj_type]:
			n_params_required = self.n_object_params[obj_type]
			params_format = self.required_params[obj_type]
			raise ValueError(f"Object requires {n_params_required} parameters: {params_format}")
		
		params_np = np.array([params], dtype=self.float)
		
		if obj_type == self.plane_type:
			params_np[7:] = params_np[7:] / np.linalg.norm(params_np[7:])
		
		if len(self.object_params[obj_type]) == 0:
			self.object_params[obj_type] = params_np
		else:
			self.object_params[obj_type] = np.concatenate([self.object_params[obj_type], params_np])
				
	def add_mesh(self, filename, params):
		
		if len(params) != 9:
			raise ValueError("Meshes require 9 parameters: r,g,b,sx,sy,sz,x1,y1,z1")
	
		scene = pywavefront.Wavefront(filename, collect_faces=True)
		color = np.array(params[0:3], dtype=self.float)
		scale = params[3:6]
		center = params[6:9]
		scale_np = np.array(scale + scale + scale, dtype=self.float)
		center_np = np.array(center + center + center, dtype=self.float)

		params_np = []
		for face in scene.mesh_list[0].faces:
			vertices = np.array([scene.vertices[i] for i in face], dtype=self.float).reshape(9,)
			params_np.append(np.concatenate([color, scale_np*vertices+center_np]))
		params_np = np.array(params_np, dtype=self.float)

		obj_type = self.triangle_type
		if len(self.object_params[obj_type]) == 0:
			self.object_params[obj_type] = params_np
		else:
			self.object_params[obj_type] = np.concatenate([self.object_params[obj_type], params_np])
	
	def add_light(self, params):
		
		if len(params) != 3:
			raise ValueError("A light source requires 3 parameters: x,y,z")
		
		if self.light is not None:
			print("Notice: replacing previous light sourse.")
			
		self.light = np.array(params, dtype=self.float)
	
	def compile_objects(self):
		
		self.bitmap_gpu = gpuarray.empty((self.dim, self.dim, 3), dtype=np.int32)
		ray_orig = np.zeros((self.dim, self.dim, 3), dtype=np.float32)
		ray_dir = np.zeros((self.dim, self.dim, 3), dtype=np.float32)
		for i in range(self.dim):
			for j in range(self.dim):
				ray_orig[i][j][0] = j
				ray_orig[i][j][1] = i
				ray_dir[i][j][2] = 1.0
		self.ray_orig_gpu = gpuarray.to_gpu(ray_orig)
		self.ray_dir_gpu = gpuarray.to_gpu(ray_dir)
		
		for obj_type in range(self.n_object_types):
			self.n_objects[obj_type] = self.int(len(self.object_params[obj_type]))
			self.object_params_gpu.append(gpuarray.to_gpu(self.object_params[obj_type]))
		
		self.light_gpu = gpuarray.to_gpu(self.light)
	
	def draw_scene(self):
	
		[_, 
		closest_intersect_point_gpu, 
		closest_intersect_normal_gpu,
		closest_color_gpu] = self._find_closest(self.ray_orig_gpu, self.ray_dir_gpu)

		# Calculate the ray from each intersection point to the light
		[ray_to_light_gpu, 
		 distance_to_light_gpu] = self._find_ray_to_light(closest_intersect_point_gpu)

		# Shoot a ray from each intersection point to the light source
		[closest_distance_light_gpu, 
		 _, _, _] = self._find_closest(closest_intersect_point_gpu, ray_to_light_gpu)

		# Decide the color of each pixel
		self._lambertian_shading(closest_distance_light_gpu, 
					  distance_to_light_gpu, closest_color_gpu,
					  closest_intersect_normal_gpu, ray_to_light_gpu)
	
	def write_image_file(self, filename):
		
		image = cv2.cvtColor(self.bitmap_gpu.get().astype(np.uint8), cv2.COLOR_RGB2BGR)
		cv2.imwrite(filename, image)
		
	def _find_closest(self, ray_orig_gpu, ray_dir_gpu):
	
		# Allocate space on the gpu for calculations
		closest_type = np.full((self.dim, self.dim), -1, dtype=self.int) 
		closest_obj = np.full((self.dim, self.dim), -1, dtype=self.int)
		closest_distance = np.full((self.dim, self.dim), 1e8, dtype=self.float)
		closest_intersect_point = np.full((self.dim, self.dim, 3), 0.0, dtype=self.float)
		closest_intersect_normal = np.full((self.dim, self.dim, 3), 0.0, dtype=self.float)
		closest_color = np.full((self.dim, self.dim, 3), 0.0, dtype=self.float)

		closest_distance_gpu = gpuarray.to_gpu(closest_distance)
		closest_intersect_point_gpu = gpuarray.to_gpu(closest_intersect_point)
		closest_intersect_normal_gpu = gpuarray.to_gpu(closest_intersect_normal)
		closest_color_gpu = gpuarray.to_gpu(closest_color)

		# Find the closest object
		for obj_type in range(self.n_object_types):
		
			if self.n_objects[obj_type] > 0:
				self.kernel_find_closest(ray_orig_gpu, 
							  ray_dir_gpu,
							  self.int(obj_type),
							  self.object_params_gpu[obj_type],
							  self.dim,
							  self.n_objects[obj_type],
							  self.n_object_params[obj_type],
							  closest_distance_gpu,
							  closest_intersect_point_gpu,
							  closest_intersect_normal_gpu,
							  closest_color_gpu,
							  block=self.block,
							  grid=self.grid)

		return [closest_distance_gpu, closest_intersect_point_gpu, 
				closest_intersect_normal_gpu, closest_color_gpu]
	
	def _find_ray_to_light(self, closest_intersect_point_gpu):
	
		# Allocate space on the gpu for calculations
		ray_to_light = np.full((self.dim, self.dim, 3), 0.0, dtype=np.float32)
		distance_to_light = np.full((self.dim, self.dim), 1e8, dtype=np.float32)

		ray_to_light_gpu = gpuarray.to_gpu(ray_to_light)
		distance_to_light_gpu = gpuarray.to_gpu(distance_to_light)

		self.kernel_find_ray_to_light(self.dim,
					       self.light_gpu,
					       closest_intersect_point_gpu,
					       ray_to_light_gpu,
					       distance_to_light_gpu,
					       block=self.block,
					       grid=self.grid)
		return ray_to_light_gpu, distance_to_light_gpu
	
	def _lambertian_shading(self, closest_distance_light_gpu, 
				distance_to_light_gpu, closest_color_gpu, 
				closest_intersect_normal_gpu, ray_to_light_gpu):

		self.kernel_lambertian_shading(self.bitmap_gpu,
						self.dim,
						closest_distance_light_gpu,
						distance_to_light_gpu,
						closest_color_gpu,
						closest_intersect_normal_gpu,
						ray_to_light_gpu,
						block=self.block,
						grid=self.grid)


if __name__ == "__main__":

	# Create a scene
	scene = raytracer(dim=512, grid_size=16)

	# Add objects
	scene.add_object([1, 0, 0, 200, 200, 100, 50], scene.sphere_type)
	scene.add_object([0, 0.5, 1, 200, 30, 300, 50], scene.sphere_type)
	scene.add_object([0.5, 0, 0, 400, 350, -60, 20], scene.sphere_type)
	scene.add_object([1, 1, 1, 350, 350, 800, 20], scene.sphere_type)
	scene.add_object([0, 1, 0, 250, 220, 100, 30], scene.sphere_type)

	scene.add_object([0.8, 0.8, 0.0, 0, 600, 0, 0, 0.97, 0.243], scene.plane_type)
	scene.add_object([0.4, 0.9, 0.7, 0, 0, 900, 0, 0, 1.0], scene.plane_type)

	scene.add_object([1.0, 0.5, 0.0, 250, 270, 300, 320, 300, 290, 300, 340, 295], scene.triangle_type)
	scene.add_object([1.0, 0.2, 1.0, 350, 450, 350, 250, 480, 400, 310, 850, 350], scene.triangle_type)

	scene.add_mesh("teapot.obj", [0.6, 0.6, 0.6, 30, -30, 30, 150,350,500])

	scene.add_light([-50, 20, -100])

	scene.compile_objects()
	scene.draw_scene()
	scene.write_image_file("color_img.jpg")



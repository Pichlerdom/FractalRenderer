#include"fractal_kernel.h"



__global__ void fractal_kernel(uint32_t *iterations,
			       uint32_t w, uint32_t h,
			       double world_x, double world_y,
			       double world_width, double world_height,
			       uint32_t max_iterations){
  uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;

  double x_pos = world_x + (world_width/(double)w) * x_idx;
  double y_pos = world_y + (world_height/(double)h) * y_idx;


  double x_temp;
  double x = 0.0;
  double y = 0.0;

  uint32_t curr_iteration = 0;

  //  #pragma unroll
  while(x * x + y * y <= 4.0 &&
	curr_iteration < max_iterations){
    x_temp = x * x - y * y + x_pos;
    y = 2 * x * y + y_pos;
    x = x_temp;
    curr_iteration++;
  }

  iterations[ y_idx * w + x_idx] = curr_iteration;
}

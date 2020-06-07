#include"fractal_kernel.h"



__global__ void fractal_kernel(uint8_t *d_pixel_buffer,
			       uint32_t w, uint32_t h,
			       uint32_t bytes_per_pixel,
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
  while(x * x + y * y <= 2*2 &&
	curr_iteration < max_iterations){
    x_temp = x * x - y * y + x_pos;
    y = 2 * x * y + y_pos;
    x = x_temp;
    curr_iteration++;
  }

  //did this to make shore we are not writing into
  //unallocated memory but can still run any block
  //size like 16x16 for better performance.
  uint8_t *curr_pixel = d_pixel_buffer + (y_idx * w + x_idx) * bytes_per_pixel;

  if(x_idx < w ||
     y_idx < h){
    if(curr_iteration == max_iterations){
      curr_pixel[0] = 255;
      curr_pixel[1] = 0;
      curr_pixel[2] = 0;
      curr_pixel[3] = 0;
    
    }else{
      curr_pixel[0] = 255;
      curr_pixel[1] = 255 - (uint8_t) (((double)curr_iteration/(double)max_iterations) * 255.0);
      curr_pixel[2] = 255 - (uint8_t) (((double)curr_iteration/(double)max_iterations) * 255.0);
      curr_pixel[3] = 255 - (uint8_t) (((double)curr_iteration/(double)max_iterations) * 255.0);
    
    }
  }
}

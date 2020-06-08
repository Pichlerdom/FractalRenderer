#include"fractal_kernel.h"
__device__ void generate_color(uint8_t *curr_pixel, float h, float s, float v){
  float nNormalizedH = h;
  float nNormalizedS = s;
  float nNormalizedV = v;
  float nR;
  float nG;
  float nB;
   if (nNormalizedS == 0.0F)
  {
      nR = nG = nB = nNormalizedV;
  }
  else
  {
      if (nNormalizedH == 1.0F)
          nNormalizedH = 0.0F;
      else
          nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F
  }

  float nI = floorf(nNormalizedH);
  float nF = nNormalizedH - nI;
  float nM = nNormalizedV * (1.0F - nNormalizedS);
  float nN = nNormalizedV * (1.0F - nNormalizedS * nF);
  float nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
  if (nI == 0.0F)
      { nR = nNormalizedV; nG = nK; nB = nM; }
  else if (nI == 1.0F)
      { nR = nN; nG = nNormalizedV; nB = nM; }
  else if (nI == 2.0F)
      { nR = nM; nG = nNormalizedV; nB = nK; }
  else if (nI == 3.0F)
      { nR = nM; nG = nN; nB = nNormalizedV; }
  else if (nI == 4.0F)
      { nR = nK; nG = nM; nB = nNormalizedV; }
  else if (nI == 5.0F)
      { nR = nNormalizedV; nG = nM; nB = nN; }

  curr_pixel[1] = (uint8_t)(nB * 255.0F);
  curr_pixel[2] = (uint8_t)(nG * 255.0F);
  curr_pixel[3] = (uint8_t)(nR * 255.0F);

}


__global__ void fractal_kernel(uint8_t *pixel_buffer,
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
  double xx = x * x;
  double yy = y * y;

  uint32_t curr_iteration;
  
#pragma unroll
  for(curr_iteration = 0;
      curr_iteration < max_iterations;
      curr_iteration++){    
    if(xx + yy > 4.0) break;
        
    x_temp = xx - yy + x_pos;
    y = 2 * x * y + y_pos;
    x = x_temp;
    
    xx = x * x;
    yy = y * y;
  }
  __syncthreads();
  
  uint8_t *curr_pixel = pixel_buffer + (y_idx * w + x_idx) * 4;
  if(curr_iteration < max_iterations){

    generate_color(curr_pixel,
		   ((float)curr_iteration)/((float)max_iterations),
		   1.0f/logf(1.0f/sqrtf(xx + yy)),
		   1.0f);
  }else{
    curr_pixel[1] = 0;
    curr_pixel[2] = 0;
    curr_pixel[3] = 0;
  }


}

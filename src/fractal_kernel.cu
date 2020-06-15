#include"fractal_kernel.h"

__device__ void generate_color(float *curr_pixel, float h, float s, float v){
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

  curr_pixel[0] = nR;
  curr_pixel[1] = nG;
  curr_pixel[2] = nB;
  curr_pixel[3] = 0.0f;

}


__global__ void mandelbrot_kernel(cudaSurfaceObject_t surface,
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

  double x_avg = 0.0f;
  double y_avg = 0.0f;

  uint32_t curr_iteration;
    
  
  if(x_idx < w && y_idx < h){
  

    
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

      x_avg += x;
      y_avg += y;

    }
    __syncthreads();


    float curr_pixel[4];
    double angle = 0 ,sat = 0;
    if(curr_iteration < max_iterations){
      
      angle = ((float)curr_iteration)/((float)max_iterations);
      sat = 2.0f/sqrtf(xx + yy);
    }else{

      x_avg /= (double)curr_iteration;
      y_avg /= (double)curr_iteration;
      angle = abs((y_avg * y_avg)/(x_avg * x_avg));
      if(angle >= 1.0f){
	angle = abs((x_avg * x_avg)/(y_avg * y_avg));
      }
      sat = 1.0f;
    }
    
    generate_color(curr_pixel,
		   angle,
		   sat,
		   1.0f);
    

    float4 data = make_float4(curr_pixel[0],
			      curr_pixel[1],
			      curr_pixel[2],
			      curr_pixel[3]);

    surf2Dwrite(data, surface, x_idx * sizeof(float4), y_idx);

  }

}

__global__ void burning_ship_kernel(cudaSurfaceObject_t surface,
				    uint32_t w, uint32_t h,
				    double world_x, double world_y,
				    double world_width, double world_height,
				    uint32_t max_iterations){

  uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;

  double x_pos = world_x + (world_width/(double)w) * x_idx;
  double y_pos = world_y + (world_height/(double)h) * y_idx;
    
  if(x_idx < w && y_idx < h){
  

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
      y = fabs(2 * x * y + y_pos);
      x = fabs(x_temp);
    
      xx = x * x;
      yy = y * y;
    }
    __syncthreads();


    float curr_pixel[4];
    double angle = 0 ,sat = 0;
    if(curr_iteration < max_iterations){
      
      angle = ((float)curr_iteration)/((float)max_iterations);
      sat = 2.0f/sqrtf(xx + yy);
    }else{
      angle = abs((yy/xx));
      if(angle >= 1.0f){
	angle = abs((xx/yy));
      }
      sat = 1.0f;
      angle = 0.0f;
    }
    
    generate_color(curr_pixel,
		   angle,
		   sat,
		   1.0f);
    

    float4 data = make_float4(curr_pixel[0],
			      curr_pixel[1],
			      curr_pixel[2],
			      curr_pixel[3]);

    surf2Dwrite(data, surface, x_idx * sizeof(float4), y_idx);

  }
}


__global__ void julia_set_kernel(cudaSurfaceObject_t surface,
				 uint32_t w, uint32_t h,
				 double mandelbrot_x, double mandelbrot_y,
				 double world_x, double world_y,
				 double world_width, double world_height,
				 uint32_t max_iterations){

  uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;

  double x_temp;
  double x = world_x + (world_width/(double)w) * x_idx;
  double y = world_y + (world_height/(double)h) * y_idx;
  double xx = x * x;
  double yy = y * y;
  
  double x_avg = 0.0f;
  double y_avg = 0.0f;
  
  uint32_t curr_iteration;
    
  if(x_idx < w && y_idx < h){
  
    #pragma unroll  
    for(curr_iteration = 0;
	curr_iteration < max_iterations;
	curr_iteration++){    
      if(xx + yy > 4.0) break;
        
      x_temp = xx - yy + mandelbrot_x;
      y = 2 * x * y + mandelbrot_y;
      x = x_temp;
    
      xx = x * x;
      yy = y * y;
      
      x_avg += x;
      y_avg += y;
    }
    __syncthreads();


    float curr_pixel[4];
    double angle = 0 ,sat = 0;
    if(curr_iteration < max_iterations){
      
      angle = ((float)curr_iteration)/((float)max_iterations);
      sat = 2.0f/sqrtf(xx + yy);
    }else{

      x_avg /= (double)curr_iteration;
      y_avg /= (double)curr_iteration;
      angle = abs((y_avg * y_avg)/(x_avg * x_avg));
      if(angle >= 1.0f){
	angle = abs((x_avg * x_avg)/(y_avg * y_avg));
      }
      sat = 1.0f;
    }
    
    generate_color(curr_pixel,
		   angle,
		   sat,
		   1.0f);
    

    float4 data = make_float4(curr_pixel[0],
			      curr_pixel[1],
			      curr_pixel[2],
			      curr_pixel[3]);

    surf2Dwrite(data, surface, x_idx * sizeof(float4), y_idx);

  }
}

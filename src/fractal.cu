#include "fractal.h"

CudaFractalGenerator::CudaFractalGenerator(uint32_t w, uint32_t h,
						uint32_t bytes_per_pixel){
  m_w = w;
  m_h = h;
  m_bytes_per_pixel = bytes_per_pixel;
  
  cudaMalloc((void **) &d_pixel_buffer,
	     sizeof(uint8_t) * m_w * m_h * m_bytes_per_pixel);

}

CudaFractalGenerator::~CudaFractalGenerator(){

  cudaFree(d_pixel_buffer);

}

void CudaFractalGenerator::generate_fractal(uint8_t *pixel_buffer,
		      double world_x, double world_y,
		      double world_width, double world_height,
		      uint32_t max_iterations){

  dim3 block(BLOCK_N, BLOCK_N);
  dim3 grid((uint32_t) ceil( (double)m_w / (double)BLOCK_N ),
	    (uint32_t) ceil( (double)m_h / (double)BLOCK_N ));

  fractal_kernel<<<grid,block>>>(d_pixel_buffer,
				 m_w, m_h, m_bytes_per_pixel,
				 world_x, world_y,
				 world_width, world_height,
				 max_iterations);

  cudaMemcpy((void*) pixel_buffer,
	     (void*) d_pixel_buffer,
	     sizeof(uint8_t) * m_w * m_h * m_bytes_per_pixel,
	     cudaMemcpyDeviceToHost);
  
}

#include "fractal.h"

CudaFractalGenerator::CudaFractalGenerator(uint32_t w, uint32_t h){
  m_w = w;
  m_h = h;
  
  cudaMalloc((void **) &d_iterations,
	     sizeof(uint32_t) * m_w * m_h);

}

CudaFractalGenerator::~CudaFractalGenerator(){

  cudaFree(d_iterations);

}

void CudaFractalGenerator::generate_fractal(uint32_t *iterations,
		      double world_x, double world_y,
		      double world_width, double world_height,
		      uint32_t max_iterations){

  dim3 block(BLOCK_N, BLOCK_N);
  dim3 grid((uint32_t) ceil( (double)m_w / (double)BLOCK_N ),
	    (uint32_t) ceil( (double)m_h / (double)BLOCK_N ));

  fractal_kernel<<<grid,block>>>(d_iterations,
				 m_w, m_h, 
				 world_x, world_y,
				 world_width, world_height,
				 max_iterations);

  cudaMemcpy((void*) iterations,
	     (void*) d_iterations,
	     sizeof(uint32_t) * m_w * m_h,
	     cudaMemcpyDeviceToHost);
  
}

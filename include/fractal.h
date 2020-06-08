#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"

#include "fractal_kernel.h"


class CudaFractalGenerator{
public:
  CudaFractalGenerator(uint32_t w, uint32_t h);
  ~CudaFractalGenerator();

  void generate_fractal(uint8_t *pixel_buffer,
			double world_x, double world_y,
			double world_width, double world_height,
			uint32_t max_iterations);
private:
  uint32_t m_w, m_h;
  uint8_t *d_pixel_buffer;
};

#endif

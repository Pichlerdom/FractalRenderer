#ifndef _FRACTAL_KERNEL_H_
#define _FRACTAL_KERNEL_H_

#include<stdint.h>
#include<stdlib.h>

#include "cuda_runtime.h"
#include <cuComplex.h>


__global__ void fractal_kernel(uint8_t *d_pixel_buffer,
				 uint32_t w, uint32_t h,
				 uint32_t bytes_per_pixel,
				 double world_x, double world_y,
				 double world_width, double world_height,
				 uint32_t max_iterations);

#endif

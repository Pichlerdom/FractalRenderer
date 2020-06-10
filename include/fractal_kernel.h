#ifndef _FRACTAL_KERNEL_H_
#define _FRACTAL_KERNEL_H_

#define BLOCK_N 16

#include<stdint.h>
#include<stdlib.h>

#include "cuda_runtime.h"
#include <cuComplex.h>


__global__ void fractal_kernel(cudaSurfaceObject_t surface,
			       uint32_t w, uint32_t h,
			       double world_x, double world_y,
			       double world_width, double world_height,
			       uint32_t max_iterations);

#endif

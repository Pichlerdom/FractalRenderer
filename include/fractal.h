#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include <stdint.h>
#include <stdlib.h>
#include <math.h>


#include <GL/glew.h>
#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "fractal_kernel.h"

#include "model.h"

#include "timer.h"

#define POSITION_ATTR 0
#define TEXTURE_ATTR 1


class CudaFractalGenerator: public Model{
public:
  CudaFractalGenerator(uint32_t w, uint32_t h);
  ~CudaFractalGenerator();

  virtual void render();
  virtual void update();

  void set_iterations(uint32_t iterations);
  void set_world_pos(double world_x, double world_y);
  
  void set_world_size(double world_w, double world_h);

  GLuint get_texture_sampler();
  
private:
  void create_opengl_buffers();

  void cuda_pass();
  void generate_fractal(cudaArray_t *write_array);

  GLuint vertex_array;
  GLuint vbo[2];
  GLuint ibo;
  
  GLuint texture;
  GLuint texture_sampler;

  struct cudaGraphicsResource * cuda_texture;

  uint32_t m_w, m_h;

  uint32_t m_iterations;
  double m_world_x, m_world_y;
  double m_world_w, m_world_h;

  Timer* timer;

  const GLfloat positions[8] = {-1.0f, -1.0f,
			       -1.0f, 1.0f,
			       1.0f, -1.0f,
			       1.0f, 1.0f};
  const GLfloat texture_coords[8] = {1.0f, 1.0f,
				    0.0f, 1.0f,
				    1.0f, 0.0f,
				    0.0f, 0.0f};
  const GLushort indices[6] = {0,1,2,
			      1,2,3};

};

#endif

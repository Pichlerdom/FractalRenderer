#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "fractal_kernel.h"

#include "model.h"

#include "timer.h"

#define POSITION_ATTR 0
#define TEXTURE_ATTR 1

#define TEXTURE_FRONT 0
#define TEXTURE_BACK 1


class CudaFractalGenerator: public Model{
public:
  CudaFractalGenerator(uint32_t w, uint32_t h);
  ~CudaFractalGenerator();

  //model interface
  virtual void render();
  virtual void update();
  virtual glm::mat4 get_model();
  
  void set_iterations(uint32_t iterations);
  void set_world_pos(double world_x, double world_y);  
  void set_scale(double scale);

  void set_fractal(uint32_t fractal);
  
  GLuint get_texture_sampler();

  enum FRACTALS{MANDELBROT = 0,
		BURNING_SHIP,
		JULIA_SET,
		NUM_FRACTALS};

  void move_julia_constant(double delta_x, double delta_y);
  
  double last_frame_time = 0;
private:

  struct Texture_Container{
    struct cudaGraphicsResource * cuda;
    GLuint gl;
  };
  
  void create_opengl_buffers();

  void cuda_pass();
  void generate_fractal(cudaSurfaceObject_t surface);
  
  GLuint vertex_array;
  GLuint vbo[2];
  GLuint ibo;
  
  struct Texture_Container textures[2];
  GLuint texture_sampler;

  cudaSurfaceObject_t surface;
  cudaEvent_t cuda_event;
  
  
  uint32_t m_w, m_h;

  uint32_t selected_fractal;
  
  uint32_t m_iterations;
  double m_world_x, m_world_y;

  double m_curr_world_x, m_curr_world_y;
  double m_new_world_x, m_new_world_y;
  double m_mandelbrot_x, m_mandelbrot_y;
  
  double m_scale, m_curr_scale, m_new_scale;

  Timer* timer;

  bool working_on_texture = false;
  bool changed = false;

  
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

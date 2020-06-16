#include "fractal.h"

#define  _DEBUG_

CudaFractalGenerator::CudaFractalGenerator(uint32_t w, uint32_t h){
  m_w = w;
  m_h = h;

  selected_fractal = MANDELBROT;
  
  m_scale = 1.0f;
  m_curr_scale = 1.0f;
  m_new_scale = 1.0f;

  m_curr_world_x = 0.0f;
  m_curr_world_y = 0.0f;
  m_new_world_x = 0.0f;
  m_new_world_x = 0.0f;

  create_opengl_buffers();
  
  cudaSetDevice(0);
  
  timer = new Timer();
  timer->start();
}

CudaFractalGenerator::~CudaFractalGenerator(){
  glDisableVertexAttribArray(vertex_array);
  glDeleteBuffers(2,vbo);
  glDeleteVertexArrays(1,&vertex_array);  

  delete timer;

}

GLuint CudaFractalGenerator::get_texture_sampler(){
  return texture_sampler;
}

void CudaFractalGenerator::render(){  
  
  glBindVertexArray(vertex_array);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo);
 
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[TEXTURE_FRONT]);

  glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_SHORT, (void*)0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
  glBindVertexArray(0);
  
}

void CudaFractalGenerator::update(){
  cuda_pass();
}

glm::mat4 CudaFractalGenerator::get_model(){

  //TODO::::
  glm::mat4 trans_mat = glm::translate(glm::mat4(1.0f),
				       glm::vec3( m_world_y - m_curr_world_y,
						 m_world_x - m_curr_world_x,
					     
					     1.0f));
  
  glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), glm::vec3(m_curr_scale / m_scale));

  return scale_mat ; 
}

void CudaFractalGenerator::cuda_pass(){
  if(working_on_texture &&
     cudaEventQuery(cuda_event) == cudaSuccess){
    auto e = cudaEventDestroy(cuda_event);
    
    e = cudaDestroySurfaceObject(surface);
    
    e = cudaGraphicsUnmapResources(1, &cuda_texture, 0);
    
    e = cudaGraphicsUnregisterResource(cuda_texture);

    GLuint temp = textures[TEXTURE_FRONT];
    textures[TEXTURE_FRONT] = textures[TEXTURE_BACK];
    textures[TEXTURE_BACK] = temp;
    working_on_texture = false;

    m_curr_scale = m_new_scale;
    m_curr_world_x = m_new_world_x;
    m_curr_world_y = m_new_world_y;

    last_frame_time = timer->tick();
    printf("iter: %.d, mslf: %.4f\n",m_iterations, last_frame_time);
  }
  
  if(!working_on_texture && changed){
    m_new_scale = m_scale;
    m_new_world_x = m_world_x;
    m_new_world_y = m_world_y;
    timer->tick();

    auto e = cudaGraphicsGLRegisterImage(&cuda_texture, textures[TEXTURE_BACK],
				       GL_TEXTURE_2D,
				       cudaGraphicsRegisterFlagsSurfaceLoadStore);  
    e = cudaGraphicsMapResources(1, &cuda_texture, 0);
  
    cudaArray_t texture_array;
    e = cudaGraphicsSubResourceGetMappedArray(&texture_array, cuda_texture, 0, 0);

    struct cudaResourceDesc desc;
    memset(&desc, 0, sizeof(struct cudaResourceDesc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = texture_array;


    e = cudaCreateSurfaceObject(&surface, &desc);
    e = cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);

    working_on_texture = true;
    changed = false;
    generate_fractal(surface);
    
  }
}

void CudaFractalGenerator::set_iterations(uint32_t iterations){
  if(m_iterations != iterations){
    changed = true;
    m_iterations = iterations;
  }
}

void CudaFractalGenerator::set_world_pos(double world_x, double world_y){
  if(m_world_x != world_x ||
     m_world_y != world_y){
    changed = true;
    m_world_x = world_x;
    m_world_y = world_y;
  }
}

void CudaFractalGenerator::move_julia_constant(double delta_x, double delta_y){
  m_mandelbrot_x += delta_x;
  m_mandelbrot_y += delta_y;
  changed = true;
}

void CudaFractalGenerator::set_scale(double scale){
  if(m_scale != scale){
    m_scale = scale;
    changed = true;
  }
}

void CudaFractalGenerator::set_fractal(uint32_t fractal){
  if (fractal < NUM_FRACTALS)
    selected_fractal = fractal;
  changed = true;
}

void CudaFractalGenerator::create_opengl_buffers(){

  glGenVertexArrays(1, &vertex_array);
  glBindVertexArray(vertex_array);

  glGenBuffers(2, vbo);

  //Positions
  
  glBindBuffer(GL_ARRAY_BUFFER, vbo[POSITION_ATTR]);
  glBufferData(GL_ARRAY_BUFFER,
	       4 * 2 * sizeof(GLfloat),
	       positions, GL_STATIC_DRAW);

  glVertexAttribPointer(POSITION_ATTR, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(POSITION_ATTR);

  //Texture coords
  
  glBindBuffer(GL_ARRAY_BUFFER, vbo[TEXTURE_ATTR]);
  glBufferData(GL_ARRAY_BUFFER,
	       4 * 2 * sizeof(GLfloat),
	       texture_coords, GL_STATIC_DRAW);

  glVertexAttribPointer(TEXTURE_ATTR, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(TEXTURE_ATTR);

  //Texture
    
  glGenTextures(2, textures);
  glBindTexture(GL_TEXTURE_2D, textures[TEXTURE_FRONT]);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
	       m_w, m_h, 0, GL_RGBA,
	       GL_FLOAT, nullptr);
  
  glBindTexture(GL_TEXTURE_2D, textures[TEXTURE_BACK]);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
	       m_w, m_h, 0, GL_RGBA,
	       GL_FLOAT, nullptr);

  texture_sampler = 0;
  //Indices
  glGenBuffers(1,&ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	       2 * 3 * sizeof(GLushort),
	       indices, GL_STATIC_DRAW);
  

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);  
  glBindVertexArray(0);
}

void CudaFractalGenerator::generate_fractal(cudaSurfaceObject_t surface){

  dim3 block(BLOCK_N, BLOCK_N);
  dim3 grid((uint32_t) ceil( (double)m_w / (double)BLOCK_N ),
	    (uint32_t) ceil( (double)m_h / (double)BLOCK_N ));


  switch(selected_fractal){
  case BURNING_SHIP:
    burning_ship_kernel<<<grid,block,0>>>(surface,
					  m_w, m_h, 
					  m_world_x - (((double) m_h) * m_scale) / 2.0f,
					  m_world_y - (((double) m_w) * m_scale) / 2.0f,
					  ((double) m_w) * m_scale,
					  ((double) m_h) * m_scale,
					  m_iterations);
    break;
  case JULIA_SET:
    julia_set_kernel<<<grid,block>>>(surface,
				   m_w, m_h,
				   m_mandelbrot_x, m_mandelbrot_y,
				   m_world_x - (((double) m_h) * m_scale) / 2.0f,
				   m_world_y - (((double) m_w) * m_scale) / 2.0f,
				   ((double) m_w) * m_scale,
				   ((double) m_h) * m_scale,
				     m_iterations);

    break;
  case MANDELBROT:
  default:
    m_mandelbrot_x = m_world_x - (((double) m_h) * m_scale) / 2.0f;
    m_mandelbrot_y = m_world_y - (((double) m_w) * m_scale) / 2.0f;
    mandelbrot_kernel<<<grid,block,0>>>(surface,
					m_w, m_h, 
					m_world_x - (((double) m_h) * m_scale) / 2.0f,
					m_world_y - (((double) m_w) * m_scale) / 2.0f,
					((double) m_w) * m_scale,
					((double) m_h) * m_scale,
					m_iterations);

  }
  cudaEventRecord(cuda_event,0);  
}

#include "fractal.h"

#include <stdio.h>

CudaFractalGenerator::CudaFractalGenerator(uint32_t w, uint32_t h){
  m_w = w;
  m_h = h;

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
  glBindTexture(GL_TEXTURE_2D, texture);

  glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_SHORT, (void*)0);
  //glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
  glBindVertexArray(0);
}

void CudaFractalGenerator::update(){
  cuda_pass();
}

void CudaFractalGenerator::cuda_pass(){
  timer->tick();

  auto e = cudaGraphicsGLRegisterImage(&cuda_texture, texture,
				       GL_TEXTURE_2D,
				       cudaGraphicsRegisterFlagsSurfaceLoadStore);  
  e = cudaGraphicsMapResources(1, &cuda_texture, 0);
  
  cudaArray_t texture_array;
  e = cudaGraphicsSubResourceGetMappedArray(&texture_array, cuda_texture, 0, 0);

  
  generate_fractal(&texture_array);


  e = cudaGraphicsUnmapResources(1, &cuda_texture, 0);

  e = cudaGraphicsUnregisterResource(cuda_texture);

  #ifdef _DEBUG_
  printf("iter %d ,w_x %.4lf ,w_y %.4lf, t %.4lf\n",
	 m_iterations,
	 m_world_x, m_world_y,
	 timer->tick());
  fflush(stdout);
  #endif
}

void CudaFractalGenerator::set_iterations(uint32_t iterations){
  m_iterations = iterations;
}

void CudaFractalGenerator::set_world_pos(double world_x, double world_y){
  m_world_x = world_x;
  m_world_y = world_y;
}

void CudaFractalGenerator::set_world_size(double world_h, double world_w){
  m_world_w = world_w;
  m_world_h = world_h;
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
  //glGenSamplers(1, &texture_sampler);
    
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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

void CudaFractalGenerator::generate_fractal(cudaArray_t *write_array){

  struct cudaResourceDesc desc;
  memset(&desc, 0, sizeof(struct cudaResourceDesc));
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = *write_array;

  cudaSurfaceObject_t surface;
  auto e = cudaCreateSurfaceObject(&surface, &desc);

  dim3 block(BLOCK_N, BLOCK_N);
  dim3 grid((uint32_t) ceil( (double)m_w / (double)BLOCK_N ),
	    (uint32_t) ceil( (double)m_h / (double)BLOCK_N ));

  fractal_kernel<<<grid,block>>>(surface,
				 m_w, m_h, 
				 m_world_x, m_world_y,
				 m_world_w, m_world_h,
				 m_iterations);
  e = cudaPeekAtLastError();

  e = cudaDestroySurfaceObject(surface);

  if (e != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(e));
  
}

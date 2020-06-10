#ifndef _SHADER_H_
#define _SHADER_H_

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>


class Shader
{
 public:
  Shader();
  bool init();

  bool load_shader(const std::string &filename, GLenum shader_type);
  bool link_shaders();

  void use_program() const;
  
  void set_mvp_matrix(const glm::mat4 model,
		      const glm::mat4 view,
		      const glm::mat4 projection);
  void set_normal_matrix(const glm::mat4 mat);
  void set_light(const glm::vec3 light_pos);

  void set_sampler(GLuint texture);
  
  void cleanup();

private:
  GLuint shaderProgram;

  GLint model_matrix_id;
  GLint view_matrix_id;
  GLint projection_matrix_id;
  bool mvp_matrix_id_set;

  GLint normal_matrix_id;
  bool normal_matrix_id_set;

  GLint sampler_location;
  bool sampler_location_set;
  
  GLint light_id;
  bool light_id_set;
  
  std::vector<int32_t> shader_ids;
  
  //this might be useless
  GLuint vertex_shader, fragment_shader;

  
  std::string read_file(const char* file);
  bool try_compile_shader(int shader_id);
  int create_shader(const std::string &filename, GLenum shader_type);

  void print_shader_compilation_error_info(int32_t shaderId);
  void print_shader_linking_error(int32_t shaderId);

};

#endif

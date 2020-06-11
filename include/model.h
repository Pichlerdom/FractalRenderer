#ifndef _MODEL_H_
#define _MODEL_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

class Model{
 public:
  virtual void render() = 0;  
  virtual void update() = 0;
  virtual glm::mat4 get_model() = 0;
};
#endif

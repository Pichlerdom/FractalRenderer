#include "display.h"

//#define PERF_TEST

Display::Display(){
  renderer = new Renderer(WINDOW_WIDTH, WINDOW_HEIGHT);

  frac_gen = new CudaFractalGenerator(WINDOW_WIDTH, WINDOW_HEIGHT);

  handler = new EventHandler();


  if(!renderer->set_up_shader("simple.vert", "simple.frag"))
    exit(1);
    
  scale = 1.0f;
  world_x = 0.0f;
  world_y = 0.0f;
  max_iterations = 1024;
 
}

Display::~Display(){
  delete renderer;
  delete frac_gen;
  
}

void Display::display_loop(){
  while(!handler->WasQuit()){
    update();
    
    render();
  }
}

void Display::render(){
  renderer->push_matrix(glm::mat4(1.0f));
  renderer->render_start();
  
  renderer->set_sampler(frac_gen->get_texture_sampler());
  renderer->render_model(frac_gen);

  renderer->render_end();
  renderer->pop_matrix();
}

void Display::update(){
  handler->Update();

  if(handle_keyboard())
    frac_gen->update();
}

bool Display::handle_keyboard(){
  bool changed = false;
  if(handler->IsKeyDown(SDLK_w)){
    world_x -= MOVEMENT_SPEED * scale;
    changed = true;
  }else if(handler->IsKeyDown(SDLK_s)){
    world_x += MOVEMENT_SPEED * scale;
    changed = true;
  }

  if(handler->IsKeyDown(SDLK_a)){
    world_y += MOVEMENT_SPEED * scale;
    changed = true;
  }else if(handler->IsKeyDown(SDLK_d)){
    world_y -= MOVEMENT_SPEED * scale;
    changed = true;
  }

  if(handler->IsKeyDown(SDLK_f)){
    scale *= ZOOM_SPEED;
    changed = true;
  }else if(handler->IsKeyDown(SDLK_v)){
    scale /= ZOOM_SPEED;
    changed = true;
  }

  if(handler->IsKeyDown(SDLK_g)){
    max_iterations += 1;
    changed = true;
  }else if(handler->IsKeyDown(SDLK_b)){
    if(max_iterations > 1)
      max_iterations -= 1;
    changed = true;
  }

  frac_gen->set_world_pos(world_x - (((double)WINDOW_HEIGHT * scale)/2.0f),
			  world_y - (((double)WINDOW_WIDTH) * scale)/2.0f);
  frac_gen->set_world_size((double)WINDOW_WIDTH * scale,
			   (double)WINDOW_HEIGHT * scale);
  frac_gen->set_iterations(max_iterations);
  return changed;
}

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
  max_iterations = 1028;
 
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
  renderer->push_matrix(frac_gen->get_model());
  renderer->render_start();
  
  renderer->set_sampler(frac_gen->get_texture_sampler());
  renderer->render_model(frac_gen);

  renderer->render_end();
  renderer->pop_matrix();
}

void Display::update(){
  handler->Update();

  handle_keyboard();

  frac_gen->update();
}

void Display::handle_keyboard(){
  if(handler->IsKeyDown(SDLK_w)){
    world_x -= MOVEMENT_SPEED * scale;
  }else if(handler->IsKeyDown(SDLK_s)){
    world_x += MOVEMENT_SPEED * scale;
  }

  if(handler->IsKeyDown(SDLK_a)){
    world_y += MOVEMENT_SPEED * scale;
  }else if(handler->IsKeyDown(SDLK_d)){
    world_y -= MOVEMENT_SPEED * scale;
  }

  if(handler->IsKeyDown(SDLK_f)){
    scale *= ZOOM_SPEED;
  }else if(handler->IsKeyDown(SDLK_v)){
    scale /= ZOOM_SPEED;
  }

  if(handler->IsKeyDown(SDLK_g)){
    max_iterations += 1;
  }else if(handler->IsKeyDown(SDLK_b)){
    if(max_iterations > 1)
      max_iterations -= 1;
  }

  frac_gen->set_world_pos(world_x, world_y);

  frac_gen->set_iterations(max_iterations);

  frac_gen->set_scale(scale);
}

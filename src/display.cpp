#include "display.h"

//#define PERF_TEST

Display::Display(){
  renderer = new Renderer(WINDOW_WIDTH, WINDOW_HEIGHT);

  frac_gen = new CudaFractalGenerator(WINDOW_WIDTH, WINDOW_HEIGHT);

  handler = new EventHandler();


  if(!renderer->set_up_shader("simple.vert", "simple.frag"))
    exit(1);
    
  timer = new Timer();
  timer->start();
  time_since_iter_update = 0.0f;
  

  scale = 1.0f;
  world_x = 0.0f;
  world_y = 0.0f;
  max_iterations = 128;
 
}

Display::~Display(){
  delete renderer;
  delete frac_gen;  
}

void Display::display_loop(){
  double dt;
  while(!handler->WasQuit()){
    dt = timer->tick();    
    update(dt);
    
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

void Display::update(float dt){
  handler->Update();

  handle_keyboard(dt);

  frac_gen->update();
}

void Display::handle_keyboard(float dt){
  if(handler->IsKeyDown(SDLK_w)){
    world_x -= MOVEMENT_SPEED * scale * dt;
  }else if(handler->IsKeyDown(SDLK_s)){
    world_x += MOVEMENT_SPEED * scale * dt;
  }

  if(handler->IsKeyDown(SDLK_a)){
    world_y += MOVEMENT_SPEED * scale * dt;
  }else if(handler->IsKeyDown(SDLK_d)){
    world_y -= MOVEMENT_SPEED * scale * dt;
  }

  if(handler->IsKeyDown(SDLK_f)){
    scale /= ZOOM_SPEED /** dt*/;
  }else if(handler->IsKeyDown(SDLK_v)){
    scale *= ZOOM_SPEED/* * dt*/;
  }


  for (int i = 0;
       i < CudaFractalGenerator::NUM_FRACTALS;
       i++){
    if(handler->IsKeyDown(SDLK_0 + i)){
      frac_gen->set_fractal(i);
      break;
    }
  }
  /*if(handler->IsKeyDown(SDLK_m)){
    frac_gen->set_fractal(CudaFractalGenerator::MANDELBROT);
  }else if(handler->IsKeyDown(SDLK_u)){
    frac_gen->set_fractal(CudaFractalGenerator::BURNING_SHIP);
    }*/
  
  if(time_since_iter_update <= ITER_UPDATE_TIME){
    if(handler->IsKeyDown(SDLK_g)){
      max_iterations += 1;
    }else if(handler->IsKeyDown(SDLK_b)){
      if(max_iterations > 1)
	max_iterations -= 1;
    }
    time_since_iter_update -= ITER_UPDATE_TIME;
  }
  time_since_iter_update += dt;
    
  frac_gen->set_world_pos(world_x, world_y);

  frac_gen->set_iterations(max_iterations);

  frac_gen->set_scale(scale);
}

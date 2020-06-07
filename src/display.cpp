#include "display.h"

Display* init_display(){

  Display *display = (Display *) calloc(1, sizeof(Display));

  if(SDL_Init(SDL_INIT_EVERYTHING) < 0){
    printf("Could not init SDL");
    return NULL;
  }

  display->window = SDL_CreateWindow(WINDOW_NAME,
                                     SDL_WINDOWPOS_UNDEFINED,
                                     SDL_WINDOWPOS_UNDEFINED,
				     WINDOW_WIDTH,
				     WINDOW_HEIGHT,
				     SDL_WINDOW_SHOWN);
  if(display->window != NULL){
    display->renderer = SDL_CreateRenderer( display->window, -1,
					    SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC );
    if(display->renderer == NULL){
      printf("Could not create renderer!");
      return NULL;
    }

    display->texture = SDL_CreateTexture(display->renderer,
					 SDL_PIXELFORMAT_RGBA8888,
					 SDL_TEXTUREACCESS_STREAMING,
					 WINDOW_WIDTH, WINDOW_HEIGHT);
    if(display->texture == NULL){
      printf("Could not create texture!");
      return NULL;
    }
    
  }else{
    printf("Could not create window!");
    return NULL;
  }

  display->pixelFormat = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
  
  SDL_SetRenderDrawColor(display->renderer, 0xFF, 0xFF, 0xFF, 0xFF );
  return display;
}

void clean_display(Display *display){
  SDL_DestroyTexture(display->texture);
  SDL_DestroyRenderer(display->renderer);
  SDL_DestroyWindow(display->window);
  SDL_Quit();
}


//returns false if program should stop
bool handle_events(bool key_down[]){
  SDL_Event e;
  
  while(SDL_PollEvent(&e)){
    switch(e.type)
      {
      case SDL_KEYDOWN:
	switch (e.key.keysym.sym)
	  {
	  case SDLK_a: key_down[A_KEY] = true;
	    break;
	  case SDLK_d: key_down[D_KEY] = true;
	    break;
	  case SDLK_w: key_down[W_KEY] = true;
	    break;
	  case SDLK_s: key_down[S_KEY] = true;
	    break;
	  case SDLK_r: key_down[R_KEY] = true;
            break;
	  case SDLK_f: key_down[F_KEY] = true;
	    break;
	  case SDLK_v: key_down[V_KEY] = true;
	    break;
	  case SDLK_b: key_down[G_KEY] = true;
	    break;
	  case SDLK_g: key_down[B_KEY] = true;
	    break;
	  case SDLK_q: key_down[Q_KEY] = true;
	    return false;
	    break;
	  }
	break;
	  
      case SDL_KEYUP:
	switch (e.key.keysym.sym)
	  {
	  case SDLK_a: key_down[A_KEY] = false;
	    break;
	  case SDLK_d: key_down[D_KEY] = false;
	    break;
	  case SDLK_w: key_down[W_KEY] = false;
	    break;
	  case SDLK_s: key_down[S_KEY] = false;
	    break;
	  case SDLK_r: key_down[R_KEY] = false;
	    break;
	  case SDLK_f: key_down[F_KEY] = false;
	    break;
	  case SDLK_v: key_down[V_KEY] = false;
	    break;
	  case SDLK_b: key_down[G_KEY] = false;
	    break;
	  case SDLK_g: key_down[B_KEY] = false;
	    break;
	  case SDLK_q: key_down[Q_KEY] = false;
	    break;
	  }
	break;
      case SDL_QUIT:
	return false;
      }
  }
  return true;
}


void display_loop(){
  bool key_down[NUM_KEYS];
  for(int i = 0 ;i < 8; i ++){
    key_down[i] = false;
  }
  uint32_t currTime = SDL_GetTicks();
  uint32_t frameTime = 0;

  Display *display = init_display();

  int frame_count = 0;

  uint8_t *pixels_screen;

  bool run = true;


  CudaFractalGenerator* frac_gen = new CudaFractalGenerator(WINDOW_WIDTH,
							    WINDOW_HEIGHT,
							    BYTES_PER_PIXEL);
  double world_x = 0.0;
  double world_y = 0.0;
  double scale = 1.0;
  uint32_t iterations = 256;
  bool changed = true;

  
  while(run){
    currTime = SDL_GetTicks();

    
    run = handle_events(key_down);

    if(key_down[R_KEY]){
      scale = 1.0;
      world_x = 0.0;
      world_y = 0.0;
      changed = true;
    }
    
    if(key_down[W_KEY]){
      world_y -= scale * MOVEMENT_SPEED;
      changed = true;
    }else if(key_down[S_KEY]){
      world_y += scale * MOVEMENT_SPEED;
      changed = true;
    }

    if(key_down[A_KEY]){
      world_x -= scale * MOVEMENT_SPEED;
      changed = true;
    }else if(key_down[D_KEY]){
      world_x += scale * MOVEMENT_SPEED;
      changed = true;
    }

    if(key_down[F_KEY]){
      scale /= ZOOM_SPEED;
      changed = true; 
    }else if(key_down[V_KEY]){
      scale *= ZOOM_SPEED;
      changed = true;
    }

    if(key_down[G_KEY]){
        if(iterations > 1){
	iterations--;
	changed = true;
      } 
    }else if(key_down[B_KEY]){
      iterations++;
      changed = true;
    }
    
    if(changed){
      uint32_t format;
      int32_t w,h;
      int32_t pitch;
    
      SDL_QueryTexture(display->texture, &format, NULL, &w , &h);
      SDL_LockTexture(display->texture, NULL, (void **) &pixels_screen, &pitch);

      frac_gen->generate_fractal(pixels_screen,
				 world_x - ((double)WINDOW_WIDTH * scale)/2.0,
				 world_y - ((double)WINDOW_HEIGHT * scale)/2.0,
				 ((double) WINDOW_WIDTH) * scale,
				 ((double) WINDOW_HEIGHT) * scale,
				 iterations);
    
      SDL_UnlockTexture(display->texture);
    
      SDL_Rect rect = (SDL_Rect){0,0,WINDOW_WIDTH,WINDOW_HEIGHT};
    
      //Update screen
      SDL_RenderCopy(display->renderer, display->texture,NULL, &rect);
      SDL_RenderPresent( display->renderer );
    }

    //FPS stuff
    frameTime = SDL_GetTicks() - currTime;
    printf("\nmspf = %d: iter = %d\n",frameTime, iterations);
    if(frameTime > MS_PER_FRAME){
      frameTime = MS_PER_FRAME;
    }

    SDL_Delay(MS_PER_FRAME-frameTime);
    frame_count++;
    changed = false;
  }
  
  clean_display(display);
}

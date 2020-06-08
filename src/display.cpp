#include "display.h"

//#define PERF_TEST


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
  for(int i = 0 ;i < NUM_KEYS; i ++){
    key_down[i] = false;
  }
  uint32_t currTime = SDL_GetTicks();
  uint32_t frameTime = 0;

  Display *display = init_display();

  uint64_t frame_count = 0;
  uint64_t computing_frame_count = 0;

  uint8_t *pixels_screen;

  bool run = true;


  CudaFractalGenerator* frac_gen = new CudaFractalGenerator(WINDOW_WIDTH,
							    WINDOW_HEIGHT);

  double world_x = 0;
  double world_y = 0;
  
  #ifdef PERF_TEST
  world_x = 0.3928641662323556;
  world_y = -0.1364456387924788;
  #endif
  double scale = 1.0/100.0;
  uint32_t max_iterations = 1024;
  //uint32_t *iterations = (uint32_t *) malloc(WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint32_t));
  bool changed = true;

 double total_time = 0;
  
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
        if(max_iterations > 1){
	max_iterations-=1;
	changed = true;
      } 
    }else if(key_down[B_KEY]){
      max_iterations+=1;
      changed = true;
    }
    #ifdef PERF_TEST
    scale /= ZOOM_SPEED;
    
    if(scale > 1.0/42076995493.1) break;

    changed = true;   
    #endif
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
				 max_iterations);
    

      /*
      hsv curr_hsv;
      rgb curr_rgb;
      uint8_t *curr_pixel;
      uint32_t curr_iter;
      
      for(int x = 0; x < WINDOW_WIDTH; x++){
	for(int y = 0; y < WINDOW_HEIGHT; y++){
	  curr_pixel = pixels_screen + (WINDOW_WIDTH * y + x) * BYTES_PER_PIXEL;
	  curr_iter = iterations[WINDOW_WIDTH * y + x];

	  if(curr_iter < max_iterations){
	    curr_hsv.h = (sqrt((double)curr_iter/(double)max_iterations)) * 360.0;
	    curr_hsv.s = 1.0;
	    curr_hsv.v = 1.0;
	    
	    curr_rgb = hsv2rgb(curr_hsv);
	    	  
	    curr_pixel[1] = (uint8_t)(curr_rgb.b * 255.0);
	    curr_pixel[2] = (uint8_t)(curr_rgb.g * 255.0);
	    curr_pixel[3] = (uint8_t)(curr_rgb.r * 255.0);
	  }else{
	    curr_pixel[1] = 0;
	    curr_pixel[2] = 0;
	    curr_pixel[3] = 0;  
	  }
	}
      }
      */
      
      SDL_UnlockTexture(display->texture);
    
      SDL_Rect rect = (SDL_Rect){0,0,WINDOW_WIDTH,WINDOW_HEIGHT};
    
      //Update screen
      SDL_RenderCopy(display->renderer, display->texture,NULL, &rect);
      SDL_RenderPresent( display->renderer );
    }

    //FPS stuff
    frameTime = SDL_GetTicks() - currTime;
    total_time += (double)frameTime;
    
    printf("mspf = %d: iter = %d: scale = %.1lf: (x,y): (%.16lf,%.16lf)\n",
	   frameTime, max_iterations, 1.0/scale, world_x, world_y);
    if(frameTime > MS_PER_FRAME){
      frameTime = MS_PER_FRAME;
    }

    SDL_Delay(MS_PER_FRAME-frameTime);
    if(changed == true)
      computing_frame_count++;
    frame_count++;
    changed = false;
  }

  double avg_mspf = (double)total_time/(double)computing_frame_count;
  double avg_fps = 1.0/avg_mspf;
  printf("\n\navg_fps: %.4lf | avg_mspf: %.4lf\n", avg_fps, avg_mspf);
  
  clean_display(display);
}

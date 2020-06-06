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
  bool key_down[8];
  for(int i = 0 ;i < 8; i ++){
    key_down[i] = false;
  }
  uint32_t currTime = SDL_GetTicks();
  uint32_t frameTime = 0;

  Display *display = init_display();

  int frame_count = 0;

  uint8_t *pixels_screen;

  bool run = true;

  
  while(run){
    currTime = SDL_GetTicks();

    run = handle_events(key_down);
    
    uint32_t format;
    int32_t w,h;
    int32_t pitch;
    uint8_t *curr_pixel;
    
    SDL_QueryTexture(display->texture, &format, NULL, &w , &h);
    SDL_LockTexture(display->texture, NULL, (void **) &pixels_screen, &pitch);

    
    for(int x = 0; x < WINDOW_WIDTH; x++){
      for(int y = 0; y < WINDOW_HEIGHT; y++){
	curr_pixel = pixels_screen + (y * pitch) + x * BYTES_PER_PIXEL;
	curr_pixel[0] = (uint8_t) (x/WINDOW_WIDTH) * 255;
	curr_pixel[1] = (uint8_t) (y/WINDOW_HEIGHT) * 255;
	curr_pixel[2] = (uint8_t) ((x * y)/(WINDOW_WIDTH * WINDOW_HEIGHT)) * 255;
	curr_pixel[3] = 255; 
      }
    }

    //draw here
    
    SDL_UnlockTexture(display->texture);
    
    SDL_Rect rect = (SDL_Rect){0,0,WINDOW_WIDTH,WINDOW_HEIGHT};
    
    //Update screen
    SDL_RenderCopy(display->renderer, display->texture,NULL, &rect);
    SDL_RenderPresent( display->renderer );
   

    //FPS stuff
    frameTime = SDL_GetTicks() - currTime;
    printf("\nmspf = %d\n",frameTime);
    if(frameTime > MS_PER_FRAME){
      frameTime = MS_PER_FRAME;
    }

    SDL_Delay(MS_PER_FRAME-frameTime);
    frame_count++;
  }
  
  clean_display(display);
}

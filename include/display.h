#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>

#include "fractal.h"
#include "color_utils.h"

#define WINDOW_WIDTH 720
#define WINDOW_HEIGHT 720 

#define MS_PER_FRAME 16

#define BYTES_PER_PIXEL 4

#define WINDOW_NAME "Fractal Generator"

#define MOVEMENT_SPEED 10.0
#define ZOOM_SPEED 1.05

enum KEYS{W_KEY, A_KEY, S_KEY, D_KEY,
	  R_KEY, F_KEY,
	  V_KEY, Q_KEY,
	  G_KEY, B_KEY,
	  NUM_KEYS};

typedef struct{
  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* texture;
  SDL_PixelFormat *pixelFormat;
  int x;
  int y;
}Display;

void display_loop();

#endif

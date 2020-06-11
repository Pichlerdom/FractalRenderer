#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#include <SDL2/SDL.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <time.h>
#include <math.h>


#include "timer.h"

#include "fractal.h"
#include "color_utils.h"

#include "renderer.h"

#include "event_handler.h"

#define WINDOW_WIDTH 1020
#define WINDOW_HEIGHT 1020 

#define MS_PER_FRAME 16

#define BYTES_PER_PIXEL 4

#define WINDOW_NAME "Fractal Generator"

#define MOVEMENT_SPEED 1.0
#define ZOOM_SPEED 1.01


class Display{
public:
  Display();
  ~Display();

  void display_loop();

private:
  void render();
  void update();

  void handle_keyboard();
  
  enum KEYS{W_KEY, A_KEY, S_KEY, D_KEY,
	    R_KEY, F_KEY,
	    V_KEY, Q_KEY,
	    G_KEY, B_KEY,
	    NUM_KEYS};

  Renderer *renderer;
  CudaFractalGenerator *frac_gen;
  EventHandler *handler;

  double scale;
  double world_x, world_y;
  uint32_t max_iterations;

};

#endif

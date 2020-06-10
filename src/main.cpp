#include "display.h"


int main(int argc, char* argv[]){
  Display *disp = new Display();

  
  disp->display_loop();

  delete disp;

  return 0;
}

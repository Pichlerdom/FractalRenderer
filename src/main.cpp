#include "display.h"

#include "cuda_profiler_api.h"


int main(int argc, char* argv[]){
  cudaProfilerStart();
  Display *disp = new Display();

  disp->display_loop();

  delete disp;

  cudaProfilerStop();
 
  return 0;
}

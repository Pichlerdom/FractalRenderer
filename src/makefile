
.SUFFIXES : .cpp .cu

TARGET_NAME = fractal_renderer

#Directories
IDIR = ../include
ODIR = ../obj
TARGET_DIR = ../bin

#Compiler
CC = g++

#Cuda Compiler
CUDA_CC = nvcc

#C Flags
CFLAGS = -Wall $(SDL2CFLAGS)

#Cuda C Flags
CUDA_CFLAGS =  -m64 -maxrregcount=0 -gencode=arch=compute_50,code=\"sm_50,compute_50\" --fmad=false -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored

#define library names
LIBS = -lSDL2 -lSDL2_ttf -lm  -lcuda -lcudart -L/opt/cuda/lib -lGL -lGLEW

#define library paths
LFLAGS = -L"/usr/lib/x86_64-linux-gnu" 

#define SDL2 Options
SDL2CFLAGS = -D_REENRRANT
SDL2INCLUDES = -I/usr/include/SDL2 -I/opt/cuda/include

_INCLUDES = $(wildcard $(IDIR)/*.h)
INCLUDES = $(SDL2INCLUDES) -I$(IDIR) -I../include/gqd

# define the C source files
SRCS =  $(wildcard *.cpp) $(wildcard *.cu) 

#define object files
OBJ =  $(filter %.o, $(patsubst %.cpp, $(ODIR)/%.o, $(SRCS)) $(patsubst %.cu, $(ODIR)/%.o, $(SRCS)) ) 

MAIN = $(TARGET_DIR)/$(TARGET_NAME)


debug: CFLAGS += -g
debug: CUDA_CFLAGS += -g -lineinfo 
debug: $(MAIN) clean

release: CFLAGS += -O2
release: CUDA_CFLAGS += -O2
release: $(MAIN) clean

$(MAIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(INCLUDES) $(LFLAGS) $(LIBS)

$(ODIR)/%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(ODIR)/%.o:%.cu
	$(CUDA_CC) $(CUDA_CFLAGS) $(INCLUDES) -c $< -o $@

clean: $(MAIN)
	rm $(OBJ)

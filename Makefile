# usage: change the following variable COLPACK_ROOT accordingly
#        delete OMP_FLAG=-fopenmp in MAC OS system
#	 change also the Paths from Eigen and dco
COLPACK_ROOT = $(HOME)/Software/ColPack
COLPACK_SRC = $(wildcard ${COLPACK_ROOT}/src/GeneralGraphColoring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/Utilities/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/BipartiteGraphBicoloring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/BipartiteGraphPartialColoring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/SMPGC/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/PartialD2SMPGC/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/Recovery/*.cpp)

COLPACK_OBJ = $(COLPACK_SRC:%.cpp=%.o)
SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o) $(COLPACK_OBJ)
EXE=$(addsuffix .exe, $(basename $(wildcard *.cpp)))

EIGEN_DIR=$(HOME)/Software/Eigen
DCO_DIR=$(HOME)/Software/dco
DCO_INC_DIR=$(DCO_DIR)/include
DCO_LIB_DIR=$(DCO_DIR)/lib
DCO_FLAGS=-DDCO_DISABLE_AUTO_WARNING
DCO_LIB=dcoc

BASE_DIR=$(HOME)/Dokumente/SP_CES/Code
LIBLS_INC_DIR=$(BASE_DIR)/LINEAR_SYSTEM/libls/include
LIBNLS_INC_DIR=$(BASE_DIR)/NONLINEAR_SYSTEM/libnls/include
LIBNLS_APPS_INC_DIR=$(BASE_DIR)/NONLINEAR_SYSTEM/libnls_apps/include


# compiler
COMPILER = g++ -g      # gnu
OMP_FLAG = -fopenmp 

#COMPILER = icc      # intel(R)
#OMP_FLAG = -openmp

# compile flags
CCFLAGS = -Wall -std=c++11 $(OMP_FLAG)  -Ofast #-O3 
# link flags
LDFLAGS = -Wall -std=c++11 $(OMP_FLAG)  -Ofast #-O3



COLPACK_INC_DIR = $(COLPACK_ROOT)/inc
COLPACK_SRC_GGC = $(COLPACK_ROOT)/src/GeneralGraphColoring
COLPACK_SRC_BGC = $(COLPACK_ROOT)/src/BipartiteGraphBicoloring
COLPACK_SRC_BGPC = $(COLPACK_ROOT)/src/BipartiteGraphPartialColoring
COLPACK_SRC_U = $(COLPACK_ROOT)/src/Utilities
COLPACK_SRC_R = $(COLPACK_ROOT)/src/Recovery
COLPACK_SRC_SMPGC = $(COLPACK_ROOT)/src/SMPGC

all: $(EXE)

%.o : %.cpp
	$(COMPILER)  -I$(EIGEN_DIR) -I$(DCO_INC_DIR) -I$(COLPACK_INC_DIR) -I$(COLPACK_SRC_GGC) -I$(COLPACK_SRC_BGC) -I$(COLPACK_SRC_BGPC) -I$(COLPACK_SRC_U) -I$(COLPACK_SRC_R) -I$(COLPACK_SRC_SMPGC) -I$(LIBLS_INC_DIR) -I$(LIBNLS_INC_DIR) -I$(LIBNLS_APPS_INC_DIR) -L$(DCO_LIB_DIR) $(CCFLAGS) -c $< -o $@ -l$(DCO_LIB)

$(EXE): $(OBJ)
	$(COMPILER) $^ -I$(EIGEN_DIR) -I$(DCO_INC_DIR) -I$(COLPACK_INC_DIR) -I$(COLPACK_SRC_GGC) -I$(COLPACK_SRC_BGC) -I$(COLPACK_SRC_BGPC) -I$(COLPACK_SRC_U) -I$(COLPACK_SRC_R) -I$(COLPACK_SRC_SMPGC) -I$(LIBLS_INC_DIR) -I$(LIBNLS_INC_DIR) -I$(LIBNLS_APPS_INC_DIR) -L$(DCO_LIB_DIR) $(CCFLAGS) -o $@ -l$(DCO_LIB)

clean:
	rm -f $(EXE)
	rm -f $(OBJ) 

.PHONY: all clean

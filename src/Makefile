EXE=$(addsuffix .exe, $(basename $(wildcard *.cpp)))
COLPACK_ROOT = ../..
COLPACK_SRC = $(wildcard ${COLPACK_ROOT}/src/GeneralGraphColoring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/Utilities/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/BipartiteGraphBicoloring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/BipartiteGraphPartialColoring/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/SMPGC/*.cpp)
COLPACK_SRC+= $(wildcard ${COLPACK_ROOT}/src/PartialD2SMPGC/*.cpp)

COLPACK_OBJ = $(COLPACK_SRC:%.cpp=%.o)
SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o) $(COLPACK_OBJ)
EXE = ColPack

CCFLAGS = -Wall -std=c++11 $(OMP_FLAG)  -Ofast #-O3 

CPPC=g++ -g
CPPC_FLAGS=-Wall -Wextra -pedantic -Ofast -march=native
EIGEN_DIR=$(HOME)/Software/Eigen
DCO_DIR=$(HOME)/Software/dco
COLPACK_DIR=$(HOME)/Software/ColPack

ADOLC_DIR=$(HOME)/adolc_base
ADOLC_INC_DIR=$(ADOLC_DIR)/include
ADOLC_LIB_DIR=$(ADOLC_DIR)/lib64

ADOLC_LIB=adolc

DCO_INC_DIR=$(DCO_DIR)/include
DCO_LIB_DIR=$(DCO_DIR)/lib
DCO_FLAGS=-DDCO_DISABLE_AUTO_WARNING
DCO_LIB=dcoc

INCLUDES = -I./
INCLUDES = -I${COLPACK_DIR}/inc
INCLUDES+= -I${COLPACK_DIR}/src/GeneralGraphColoring
INCLUDES+= -I${COLPACK_DIR}/src/BipartiteGraphBicoloring
INCLUDES+= -I${COLPACK_DIR}/src/BipartiteGraphPartialColoring
INCLUDES+= -I${COLPACK_DIR}/src/Utilities
INCLUDES+= -I${COLPACK_DIR}/src/Recovery
INCLUDES+= -I${COLPACK_DIR}/src/SMPGC
INCLUDES+= -I${COLPACK_DIR}/src/PartialD2SMPGC

BASE_DIR=$(HOME)/Dokumente/SP_CES/Code
LIBLS_INC_DIR=$(BASE_DIR)/LINEAR_SYSTEM/libls/include
LIBNLS_INC_DIR=$(BASE_DIR)/NONLINEAR_SYSTEM/libnls/include
LIBNLS_APPS_INC_DIR=$(BASE_DIR)/NONLINEAR_SYSTEM/libnls_apps/include

all : $(EXE)	
	
        
%.o : %.cpp
	$(COMPILER) $(INCLUDES) $(CCFLAGS) -c $< -o $@
	$(CPPC) $(CPPC_FLAGS) $(DCO_FLAGS) -I$(EIGEN_DIR) -I$(ADOLC_INC_DIR) -I$(DCO_INC_DIR) -I$(LIBLS_INC_DIR) -I$(LIBNLS_INC_DIR) -I$(LIBNLS_APPS_INC_DIR) $(INCLUDES) -Wl,--rpath -Wl,$(ADOLC_LIB_DIR) -L$(DCO_LIB_DIR) -L$(ADOLC_LIB_DIR) $< -o $@ -l$(DCO_LIB) -l$(ADOLC_LIB) 


$(EXE): $(OBJ)



doc:
	cd doc && $(MAKE)

clean :
	cd doc && $(MAKE) clean
	rm -fr $(EXE) 

.PHONY: all doc clean


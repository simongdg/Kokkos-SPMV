TRILINOS_PATH = ${HOME}/Trilinos
KOKKOS_PATH = ${TRILINOS_PATH}/packages/kokkos
KOKKOS_DEVICES = "OpenMP"
KERNEL = 2

SRC = $(wildcard *.cpp) 
#SRC += $(wildcard ${TRILINOS_PATH}/packages/tpetra/kernels/src/impl/*spmv*.cpp)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
EXE = test_matvec.cuda
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
EXE = test_matvec.host
KOKKOS_ARCH = "SNB"
endif



LINK = ${CXX}
LINKFLAGS = 

CXXFLAGS = -O3 -I./ -I${TRILINOS_PATH}/packages/tpetra/kernels/src -I${TRILINOS_PATH}/packages/tpetra/kernels/src/impl
ifeq ($(KERNEL),1)
CXXFLAGS += -DMATVEC_MINIFE
EXE := $(EXE)-minife
endif
ifeq ($(KERNEL),2)
CXXFLAGS += -DMATVEC_NEW
EXE := $(EXE)-new
endif
ifeq ($(KERNEL),3)
CXXFLAGS += -DMATVEC_TPETRA
EXE := $(EXE)-tpetra
endif

#CXXFLAGS += -I${CUDA_ROOT}/include

DEPFLAGS = -M

SRC_NODIR = $(notdir $(SRC))
OBJ = $(SRC_NODIR:.cpp=.o)

LIB =
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
LIB += -lcusparse
endif

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

#%.o:${TRILINOS_PATH}/packages/tpetra/kernels/src/impl/%.cpp $(KOKKOS_CPP_DEPENDS)
#	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

## Themis directory
THEMISDIR=$(shell pwd)

## Executable directories (automatically makes subdirectories)
EXECDIRS=tests examples validation analyses sandbox hdf5

## Read in config file if it exists, otherwise try some defaults.  However,
## if the local position of the FFTW headers and libraries are not in
## ../FFTW/include and ../FFTW/lib the defaults will not work.  The proper
## solution is to define the CC, CC_FLAGS, and FFTWDIR variables for the
## local system in the Makefile.config file.  You may finde examples in
## the config directory.
ifeq ($(shell test -f Makefile.config && echo yes),yes)
  include Makefile.config
  $(info "INFO: --------------------------------------------------------------------------")
  $(info "INFO: Found Makefile.config.")
  $(info "INFO: Using CC=$(CC)")
  $(info "INFO: Using CC_FLAGS=$(CC_FLAGS)")
  $(info "INFO: Using FFTW_INCLUDE_DIR=$(FFTW_INCLUDE_DIR)")
  $(info "INFO: Using FFTW_LIB_DIR=$(FFTW_LIB_DIR)")
  $(info "INFO: Using TORCH_DIR=$(TORCH_DIR)")
  $(info "INFO: --------------------------------------------------------------------------")
else
  CC=mpicxx
  CC_FLAGS=-Wall -O3 -mtune=native -march=native -ffast-math -std=c++0x -fno-finite-math-only
  FFTW_INCLUDE_DIR=$(THEMISDIR)/../FFTW/include
  FFTW_LIB_DIR=$(THEMISDIR)/../FFTW/lib
  TORCH_DIR=
  $(warning "WARNING: --------------------------------------------------------------------------")
  $(warning "WARNING: Could not find Makefile.config, using defaults, which probably won't work.")
  $(warning "WARNING: Using CC=$(CC)")
  $(warning "WARNING: Using CC_FLAGS=$(CC_FLAGS)")
  $(warning "WARNING: Using FFTW_INCLUDE_DIR=$(FFTW_INCLUDE_DIR)")
  $(warning "WARNING: Using FFTW_LIB_DIR=$(FFTW_LIB_DIR)")
  $(warning "INFO: Using TORCH_DIR=$(TORCH_DIR)")
  $(warning "WARNING: --------------------------------------------------------------------------")
endif



## Export CC
export CC

## Export FFTW directory
export FFTW_INCLUDE_DIR
export FFTW_LIB_DIR
export TORCH_DIR


## Export C++ compiler flags, with MPI enabled in VRT2
CC_FLAGS += -DVRT2_USE_MPI_MAP
CC_FLAGS += -DTHEMISPATH=\"$(THEMISDIR)/..\"
CC_FLAGS += -D_REENTRANT
CC_FLAGS += -DBOOST_DISABLE_ASSERTS
export CC_FLAGS

## Export C++ for VRT2 if required
export CC_VRT2=$(CC)

# TBB stuff for STAN
export CC_STAN=$(CC)
MATH=$(THEMISDIR)/themistan/stan/lib/stan_math/
TBB ?= $(MATH)lib/tbb_2019_U8
TBB_BIN ?= $(MATH)lib/tbb
TBB_RELATIVE_PATH ?= ../$(notdir $(TBB))
TBB_BIN_ABSOLUTE_PATH = $(abspath $(TBB_BIN))
TBB_ABSOLUTE_PATH = $(abspath $(TBB))

# Torch stuff
ifneq ($(TORCH_DIR),)
  TORCH_IFLAGS=-I$(TORCH_DIR)/include
  TORCH_LFLAGS=-Wl,-rpath,$(TORCH_DIR)/lib,--as-needed,"$(TORCH_DIR)/lib/libtorch_cpu.so" $(TORCH_DIR)/lib/libc10.so -pthread
  CC_FLAGS += -DENABLE_TORCH
  $(info "Torch enabled. Using TORCH_DIR=$(TORCH_DIR)")
else
  TORCH_IFLAGS=
  TORCH_LFLAGS=
endif

## Include flags (global path)


export IFLAGS=-I$(THEMISDIR)/data -I$(THEMISDIR)/likelihood -I$(THEMISDIR)/model -I$(THEMISDIR)/sampling -I$(THEMISDIR)/util -I$(THEMISDIR)/optimizing -I$(THEMISDIR)/VRT2/include -I$(FFTW_INCLUDE_DIR) -isystem$(THEMISDIR)/themistan/stan/src -I$(THEMISDIR)/themistan/stan/./src/ -I$(THEMISDIR)/themistan/stan/lib/stan_math -I$(THEMISDIR)/LBFGSpp/include -isystem$(THEMISDIR)/themistan/stan/lib/stan_math/lib/eigen_3.3.3 -isystem$(THEMISDIR)/themistan/stan/lib/stan_math/lib/boost_1.72.0 -isystem$(THEMISDIR)/themistan/stan/lib/stan_math/lib/sundials_4.1.0/include/ -isystem$(THEMISDIR)/themistan/stan/lib/stan_math/lib/tbb_2019_U8/include/ -isystem$(THEMISDIR)/themistan/stan/lib/stan_math/lib/tbb -I$(THEMISDIR)/themistan $(TORCH_IFLAGS)

## Linker flags
LFLAGS=-L$(FFTW_LIB_DIR) -L$(THEMISDIR)/VRT2/lib -L$(THEMISDIR)/themistan/stan/lib/stan_math/lib/tbb -L$(THEMISDIR)/themistan/stan/lib/stan_math/lib/boost_1.72.0/stage/lib -L$(THEMISDIR)/themistan/stan/lib/stan_math/lib/tbb_2019_U8/include -Wl,-L,"$(TBB_BIN_ABSOLUTE_PATH)" -Wl,-rpath,"$(TBB_BIN_ABSOLUTE_PATH)"
LIBS=$(TORCH_LFLAGS) -lm -lfftw3 -lhdf5 -lvrt2 -ltbb 

#export HDF5_MPI="ON"
#export HDF5_CXX=mpiCC
#export HDF5_CLINKER=mpiCC
## Subdirectories to make in
SUBDIRS=data model likelihood themistan sampling optimizing util

# For tests
NOTTESTSRCS=$(shell ls data/*.cpp likelihood/*.cpp model/*.cpp sampling/*.cpp optimizing/*.cpp util/*.cpp themistan/*.cpp)
NOTTESTOBJS=$(NOTTESTSRCS:.cpp=.o)

## Executable directories (automatically makes subdirectories)
EXECNAMES=$(shell find $(EXECDIRS) -name "*.cpp" | sed -e 's/\.cpp//' )

foo:
	echo $(EXECDIRS)
	echo $(EXECNAMES)


default: subdirs

clean:
	rm -f */*.o

distclean: clean
	$(MAKE) -C VRT2/src distclean
	$(MAKE) -f themistan/stan/lib/stan_math/make/standalone math-clean

all: vrt2lib stan lbfgs subdirs tests examples validation docs

subdirs: VRT2/lib/libvrt2.a VRT2/include/vrt2.h
	for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir; \
	done

vrt2lib: 
	$(MAKE) -C VRT2/src install

stan:
	cd ../; git submodule update --init --recursive
	cp themistan/local themistan/stan/make/
	cp themistan/local themistan/stan/lib/stan_math/make/
	$(MAKE) -f themistan/stan/lib/stan_math/make/standalone math-libs

lbfgs:
	cd LBFGSpp; git submodule update --init --recursive

$(EXECDIRS): subdirs
	EDIR=$@ ;\
	EDIRS=$$(ls -d $$EDIR/*/) ;\
	EMAINS=$$(ls $$EDIR/*.cpp $$EDIR/*/*.cpp) ;\
	echo $$EDIR ;\
	echo $$EDIRS ;\
	echo $$EMAINS ;\
	mkdir -p ../bin/$$EDIR ; \
	for td in $$EDIRS; do \
	  mkdir -p ../bin/$$td ;\
	done ;\
	echo $$EDIR ;\
	for tm in $$EMAINS; do \
	  echo " making:" $$tm ;\
	  execname=$$(echo $$tm | sed -e "s/\.cpp//") ;\
	  execname=$$(echo $$execname | awk '{print "../bin/"$$1}') ;\
	  $(CC) $(CC_FLAGS) $(IFLAGS) $(LFLAGS) -o $$execname $$tm $(NOTTESTOBJS) $(LIBS) ;\
	done

$(EXECNAMES): subdirs
	EMAIN=$$(echo $@ | awk '{print $$1".cpp"}') ;\
	EDIR=$$(dirname $@.cpp) ;\
	mkdir -p ../bin/$$EDIR ;\
  echo "=======";\
	execname=$$(echo $@ | awk '{print "../bin/"$$1}') ;\
	echo " making $$execname from $$EMAIN in $$EDIR" ;\
	$(CC) $(CC_FLAGS) $(IFLAGS) $(LFLAGS) -o $$execname $$EMAIN $(NOTTESTOBJS) $(LIBS)
#	$(CC) $(CC_FLAGS) $(IFLAGS) -o $$execname $$EMAIN $(NOTTESTOBJS) $(LFLAGS) $(LIBS)


docs:
	cd ../docs ; \
	doxygen Doxyfile; \
	./copy_header_links; \
	./generate_object_inventory; \


## C++ compilers
#export CC=mpicxx
#export CC=env OMPI_CXX=icpc mpicxx
#export CC=g++
#export CC=icc


## (TITAN) C++ compiler flags
#export CC_FLAGS=-Wall -O3 -mtune=native -march=native -ffast-math -DVRT2_USE_MPI_MAP


# (SHARCNET) C++ compiler flags
# export CC_FLAGS=-Wall -O3 -no-prec-div -axCORE-AVX2 -xHOST
# export CC_FLAGS=-Wall -O3 -no-prec-div -xHOST -ipo -DVRT2_USE_MPI_MAP

## Check for C++ compilers externally set
ifdef CC_VRT2
  $(info "VRT2 INFO: ------------------------------------------------------------------------")
  $(info "VRT2 INFO: CC found.")
  $(info "VRT2 INFO: Using CC=$(CC_VRT2)")
  $(info "VRT2 INFO: ------------------------------------------------------------------------")
else
  CC_VRT2=mpicxx
  $(warning "VRT2 WARNING: ---------------------------------------------------------------------")
  $(warning "VRT2 WARNING: CC was not set, using defaults, which probably won't work.")
  $(warning "VRT2 WARNING: Using CC=$(CC_VRT2)")
  $(warning "VRT2 WARNING: ---------------------------------------------------------------------")
endif
export CC=$(CC_VRT2)

## Check for C++ compiler flags externally set
ifdef CC_FLAGS
  $(info "VRT2 INFO: ------------------------------------------------------------------------")
  $(info "VRT2 INFO: CC_FLAGS found.")
  $(info "VRT2 INFO: Using CC_FLAGS=$(CC_FLAGS)")
  $(info "VRT2 INFO: ------------------------------------------------------------------------")
else
  export CC_FLAGS=-Wall -O3 -mtune=native -march=native -ffast-math -std=c++0x -DVRT2_USE_MPI_MAP
  $(warning "VRT2 WARNING: ---------------------------------------------------------------------")
  $(warning "VRT2 WARNING: CC_FLAGS was not set, using default, which probably won't work.")
  $(warning "VRT2 WARNING: Using CC_FLAGS=$(CC_FLAGS)")
  $(warning "VRT2 WARNING: ---------------------------------------------------------------------")
endif


## Linker flags
LFLAGS=

## Subdirectories to make in
SUBDIRS=AccretionFlows Include Metrics Misc RadiativeTransfer Rays StopConditions


default: libvrt2.a

clean:
	rm -f */*.o libvrt2.a

distclean:
	rm -rf ../lib/* ../include/* */*.o libvrt2.a

install: libvrt2.a
	$(info "Making VRT2")
	mkdir -p ../lib
	cp -a libvrt2.a ../lib/.
	bash ../Scripts/install_headers.sh

libvrt2.a: subdirs
	ar rcs libvrt2.a $(wildcard */*.o)

subdirs:
	for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir; \
	done


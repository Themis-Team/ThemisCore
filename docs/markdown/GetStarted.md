Getting Started {#GetStarted}
============

\brief How to prepare, compile, and execute Themis.


External System Requirements
------------------------
To facilitate portability, the number of external system requirements for Themis has been intentionally kept to the bare minimum.  Nevertheless, it does require some elements to be installed to access its full functionality.  These are:

- **Make**: Themis uses the *make* utility to compile the code.  In addition, there are some compilation scripts that are written assuming that *bash* exists.  Both of these are standard on all POSIX systems.  Make versions that have been tested include *GNU Make 3.81*, *GNU Make 4.1*, *GNU Make 4.2.1*.

- \b C++: The native Themis code is written in C++, and thus requires an installation of the C++ compiler.  Some features of Themis require C++11 directives, though the experimental support present in *gcc* is sufficient.  Again, these are standard on all POSIX systems.  Compilers that have been tested include *clang-800.0.42.1*, *clang-802.0.42*, *gcc 4.4.7*, *gcc 4.8.4*, *gcc 4.8.5*, *gcc 4.9.4*, *gcc 6.1.1*, *gcc 6.3.0*, *icpc 12.1.3*.

- **MPI**: Themis is parallelized via the *Message Passing Interface*.  This is installed in most high-performance computing environments, and may be installed locally on most systems via the local package manager (yum, aptget, homebrew, etc.) or from source for, e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/).  Themis has been tested with *Open MPI 1.6.2*, *Open MPI 1.6.5*, *Open MPI 1.10.3*, *Open MPI 1.10.4*, *Open MPI 2.0.2*, *Open MPI 2.1.1*.

- **FFTW v3.0 or later**: To compute Fourier transforms Themis uses the [FFTW](http://www.fftw.org/) library.  This is installed in most high-performance computing environments, and may be installed locally on most systems via the local package manager (yum, aptget, homebrew, etc.) or from source from the [FFTW website](http://www.fftw.org/) directly.  Themis has been tested with *FFTW 3.2.1*, *FFTW 3.3.3*, *FFTW 3.3.4*, *FFTW 3.3.5*.

- **Doxygen 1.8.5 or later**: The Themis documentation is automatically generated using the [Doxygen](http://www.doxygen.org/) package.  This is only required if you wish to *remake* the documentation; Themis documentation is provided in html format. Themis documation has been tested with *Doxygen 1.8.5*, *Doxygen 1.8.6*, *Doxygen 1.8.9.1*.

- **Graphviz**: Automatic call/caller graph, inheritance diagram, and collaboration diagram support is provided via the graphviz infrastructure.  This is only required if you wish to *remake* the documentation; Themis documentation is provided in html format. Themis documation has been tested with *Graphviz 2.26.0*, *Graphviz 2.30.1*, *Graphviz 2.36.0*, *Graphviz 2.38.0*.

In addition, Themis is distributed with the VRT2 library, which may be made via the Themis make process.  To make use of ASTRORAY or [grtrans](https://github.com/jadexter/grtrans), the relative code libraries must be separately obtained and installed.


Themis Makefile Configuration
------------------------
Because the different external components vary in type and location on individual systems it is necessary to configure the compilation process.  In Themis this is accomplished via the presence of a *Makefile.config* file, to be located in Themis/src; examples can be found in the *config* directory (e.g., Makefile.config.Procyon).  This file defines:

- The MPI C++ compiler to use by defining the variable CC.

- The appropriate compiler flags to use by defining the variable CC_FLAGS.  These should enable C++11 directives (e.g., the *-std=c+0x* flag for gcc, or *-std=c++11* for icpc, etc.).  They should also include the optimal set of compiler optimization (e.g., *-O3 -mtune=native -march=native -ffast-math* for gcc).

- The absolute path of the directory containing fftw3.h by defining the variable FFTW_INCLUDE_DIR.

- The absolute path of the directory containing libfftw3.a by defining the variable FFTW_LIB_DIR.

In the absence of *Makefile.config*, make will make some, likely poor, default choices that will typically fail on most systems (primarily the FFTW include and library directories, which are then assumed to be located in Themis/FFTW/include and Themis/FFTW/lib).  Make will report if it can find a Makefile.config, and the values of CC, CC_FLAGS, FFTW_INCLUDE_DIR, FFTW_LIB_DIR that will be used.

**Note that an individual user's Makefile.config should *never* be uploaded as such, but users are strongly encouraged to add their own configuration file in the *config* directory, appended with an appropriate hostname, to share the appropriate configuration choices on different systems.**


Making Themis
------------------------
After creating an appropriate Makefile.config, Themis can be compiled.  The simplest way to get started with the examples and tests is to run

\verbatim
$ make all
\endverbatim

This will generate the VRT2 library, compile the individual components of Themis, and compile the drivers in the Themis/src/tests, Themis/src/examples, and Themis/src/validation directories, about which information may be found on the [Examples and Tests page](@ref Tests).  The resulting executables will be found in Themis/bin/tests, Themis/bin/examples, and Themis/bin/validation, respectively.

Specific make options are:

*make* -- Makes the Themis components by default.

*make clean* -- Cleans up the Themis component object files.

*make distclean* -- Cleans up the Themis component object files and the VRT2 libraries.

*make all* -- Makes the VRT2 library, Themis components, and drivers in Themis/src/tests, Themis/src/examples, and Themis/src/validation directories, and the Themis documentation.  Executables are in Themis/bin/tests, Themis/bin/examples, and Themis/bin/validation directories, respectively. 

*make subdirs* -- Makes the Themis components.

*make vrt2lib* -- Makes the VRT2 libraries.

*make tests* -- Makes the drivers in Themis/src/tests and Themis/src/tests/*.  Executables are in Themis/bin/tests and its subdirectories.

*make examples* -- Makes the drivers in Themis/src/examples and Themis/src/examples/*.  Executables are in Themis/bin/examples and its subdirectories.

*make validation* -- Makes the drivers in Themis/src/validation and Themis/src/validation/*.  Executables are in Themis/bin/validation and its subdirectories.

*make development_tests* -- Makes the drivers in Themis/src/development_tests and Themis/development_tests/tests/*.  Executables are in Themis/bin/development_tests and its subdirectories.

*make analyses* -- Makes the drivers in Themis/src/analyses and Themis/src/analyses/*.  Executables are in Themis/bin/analyses and its subdirectories.

*make sandbox* -- Makes the drivers in Themis/src/sandbox and Themis/src/sandbox/*.  Executables are in Themis/bin/sandbox and its subdirectories.

*make docs* -- Makes the Themis documentation with the Doxygen package.

You may also make any individual executable within Themis/src/tests, Themis/src/examples, Themis/src/validation, Themis/src/development_tests, Themis/src/analyses, and Themis/src/sandbox by specifying the file name (without .cpp), e.g., to generate the executable associated with Themis/src/tests/gaussian_blob.cpp:

\verbatim
make tests/gaussian_blob
\endverbatim

The corresponding executable will be located in the appropriate subdirectory of Themis/bin, Themis/bin/tests in this case.

Running Themis
------------------------
To run Themis, descend into the appropriate directory and run with the appropriate mpirun command.  For example, with OpenMPI:

\verbatim
mpirun -np <# of procs> <executable>
\endverbatim

Note that currently all internal input file names (e.g., a model input parameter like that used for the model_image_sed_fitted_riaf object) must be either contain absolute paths (and thus not be portable to other machines) or the relative position of the executable must be fixed.  Thus, some examples and tests may fail if they are run in directories other than the Themis/bin/examples or Themis/bin/tests directories.  


Getting help
------------------------
The Themis Development Team is here to help! If you need assistance, please [contact us](@ref Developers).





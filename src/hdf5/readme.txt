
Note
(1) need to load hdf5
(2) make sure CC = mpicxx is assgined in the Makefile.config

For an example
step 1:  make hdf5/example/example_hdf5_mpi
step 2:  go to bin/hdf5/example
step 3a: use readlink -f <h5 file> and save to hfile
step 3b: mpirun -nb 5 example_hdf5_mpi hfile




#include "likelihood.h"
#include <mpi.h>
#include <string>
#include <iostream>
#include <iomanip>

#include "optimizer_simplex.h"
#include "optimizer_powell.h"


int main(int argc, char* argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  int seed = 42;
  if (argc>1) 
    seed = atoi(argv[1]);


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  
  // Dynamically allocate a prior for each variable and add it to the container
  // Here we are using a flat prior for all the parameters and confine them 
  // to the interval [-2.5*pi, 2.5*pi]
  std::vector<double> m, stddev;
  for (size_t j=0; j<20; ++j){
    //P.push_back(new Themis::prior_linear(-5*M_PI, 5*M_PI));
    P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
    m.push_back(0.5);
    stddev.push_back(1.0);
  }
  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  L.push_back(new Themis::likelihood_gaussian(m,stddev));
  //L.push_back(new Themis::likelihood_eggbox(5.0));

  //std::cerr << "Likelihood made " << world_rank << "\n";


  // Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Make an optimizer object
  //Themis::optimizer_simplex opt(seed);
  Themis::optimizer_powell opt(seed);

  // Optimize
  std::vector<double> pbest;
  pbest = opt.run_optimizer(L_obj, "OptDef.dat", 64); 
  //pbest = opt.run_optimizer(L_obj, P, "OptDef.dat", 0, 2, 10000, 1e-10);
  //pbest = opt.run_optimizer(L_obj, P, "OptDef.dat", 0, 2, 20000, 1e-15);

  // Optimize again but with more
  //pbest = opt.run_optimizer(L_obj, P, "Opt32.dat", 32);


  if (world_rank==0)
  {
    for(size_t k=0; k<pbest.size(); ++k)
      std::cerr << std::setw(15) << pbest[k];
    std::cerr << '\n';
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}




/*! 
  \file tests/base_grid_search.cpp
  \author Mansour Karami
  \date Jul 2017
  \brief Test problem for the grid search sampling 
  \details This is a simple and fast test problem for the grid search smapling routine.
  It samples a two diemntional gaussian likelihood with flat priors. 
  The tests produced a single output containing coordinates of the points on the grid and 
  their associated chi squared values. 
  \test Check the sampler on a gaussian likelihood
*/
#include "likelihood.h"
#include "sampler_grid_search.h"
#include <mpi.h>
#include <memory> 
#include <string>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  //int rank = MPI::COMM_WORLD.Get_rank();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  //Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  //Dynamically allocate a prior for each variable and add it to the container 
  P.push_back(new Themis::prior_linear(-10.0, 10.0));
  P.push_back(new Themis::prior_linear(-10.0, 10.0));
  //std::vector<std::shared_ptr<Themis::prior_base> > P;
  //P.push_back(std::shared_ptr<Themis::prior_base>(new Themis::prior_linear(-3.0, 3.0)));



  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  std::vector<double> m = {0.0,0.0};
  std::vector<double> stddev = {0.25,0.25}; 
  L.push_back(new Themis::likelihood_gaussian(m, stddev));


  //Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  //Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  

  //Make a Themis grid_search object
  Themis::sampler_grid_search GS_obj;

  
  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  //std::vector<std::string> var_names;

  std::vector<double> range_min(0), range_max(0);
  std::vector<int> num_samples(0);
  range_min.push_back(-5.0);
  range_max.push_back(5.0);
  num_samples.push_back(10);
  range_min.push_back(-4.0);
  range_max.push_back(4.0);
  num_samples.push_back(8);
  //var_names.push_back("x");
  //var_names.push_back("y");

  int Number_of_batches = 4; // Number of parameter batches 
  int cpu_per_likelihood = 1; // Number of MPI processes allocated to each batch

  // Set the cpu distribution on different parallelization levels
  GS_obj.set_cpu_distribution(Number_of_batches, cpu_per_likelihood);
  // Run the sampler with the given settings
  GS_obj.run_sampler(L_obj, range_min, range_max, num_samples, "grid.dat");

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

/*! 
  \file tests/base_DE_mcmc_sampling.cpp
  \author Mansour Karami
  \date May 2017
  \brief Test problem for the differential evolution parallel tempered MCMC sampling 
  \details This is a simple and fast test problem for the differential evolution parallel tempered MCMC sampling routine.
  It samples a two diemntional gaussian likelihood with flat priors. The test produces a MCMC chain file,
  a likelihood file and a chi-squared file. It also produces checkpoints along the run. 
  \test Check the sampler on a gaussian likelihood
*/
#include "likelihood.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
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
  //P.push_back(new prior_linear(0.0, 5.0));
  //std::vector<std::shared_ptr<Themis::prior_base> > P;
  //P.push_back(std::shared_ptr<Themis::prior_base>(new Themis::prior_linear(-3.0, 3.0)));


  //Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  //T.push_back(new transform_none());


  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  std::vector<double> m = {0.0,0.0};
  std::vector<double> stddev = {0.25,0.25};
  L.push_back(new Themis::likelihood_gaussian(m, stddev));


  //Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  //Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  //std::vector<double> X(0);
  //X.push_back(2.0);
  //X.push_back(2.5);
  //std::cout << L_obj(X) << std::endl;

  //Themis::PTMCMC MC_obj;
  Themis::sampler_differential_evolution_tempered_MCMC MC_obj(42+world_rank);

  
  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;

  std::vector<double> means, ranges;
  means.push_back(0.0);
  ranges.push_back(1.0);
  means.push_back(0.0);
  ranges.push_back(1.0);
  var_names.push_back("x");
  var_names.push_back("y");

  int Number_of_chains = 8;
  int Number_of_temperatures = 3;
  int cpu_per_likelihood = 1;
  int Number_of_steps = 1000; 
  int temp_stride = 50; // Number of MCMC steps between communicating between neighbouring temperatures   
  int chi2_stride = 10; // frequency of calculating chi squared values
  int ckpt_stride = 1000;

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  MC_obj.set_checkpoint(ckpt_stride, "sampler.ckpt");
  // Run the sampler with the given settings
  MC_obj.run_sampler(L_obj, Number_of_steps,
		       temp_stride, chi2_stride, "chain.dat", "lklhd.dat", "chi2.dat", 
		     means, ranges, var_names,  false);

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

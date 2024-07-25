/*! 
  \file examples/eggbox_mcmc_sampling.cpp
  \author Mansour Karami
  \date May 2017
  
  \brief Example of how to use the Affine invariant parallel tempered
  MCMC sampling routine

  \details In Themis, we can use any number of sampling routines to
  sample the posterior probability distribution function in the
  parameter space associated with any model.  Affine invariant
  parallel tempered MCMC (Markov Chain Monte Carlo) is one such
  routine. It runs an ensemble of MCMC chains that collectively sample
  the posterior distribution.  It also has the ability to use parallel
  tempering to facilitate sampling multi-modal distributions.  The
  sampler needs a Themis likelihood object in order to sample the
  posterior distribution.  In the following example we use a five
  dimensional egg-box likelihood which is a multimodal distribution
  with 3125 peaks within the prior region.


*/


#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>

int main(int argc, char* argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  
  //Dynamically allocate a prior for each variable and add it to the container
  //Here we are using a flat prior for all the parameters and confine them 
  //to the interval [-2.5*pi, 2.5*pi]
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  

  //Set the variable transformations.
  //Here we are using no coordinated transformations on the parameters
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());


  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_eggbox(5.0));


  //Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  //Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  //Making a Themis sampler object. Here the affine invariant tempered 
  //Markov Chain Monte Carlo method is used. The seed of the random number 
  //generator is passed to the constructor.
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42);

  
  //The means and standard deviations used to initialize the MCMC walkers.
  //For each parameter the walkers would be drawn from a gaussian distribution
  //with the given mean and standard deviation. This provides the starting point
  //for the walkers.
  std::vector<double> means, ranges;
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);


  // vector to hold the name of variables, if the names are provided it would  
  // be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("test_parameter1");
  var_names.push_back("test_parameter2");
  var_names.push_back("test_parameter3");
  var_names.push_back("test_parameter4");
  var_names.push_back("test_parameter5");


  int Number_of_chains = 100;      // Number of walkers at each temperature
  int Number_of_temperatures = 5;  // Number of temperatures in the parallel tempering
  int cpu_per_likelihood = 1;      // Number of processes used to evaluate a single likelihood 
  int Number_of_steps = 8000;      // Number of monte carlo steps 
  int temp_stride = 50;            // Number of MCMC steps for communication among neighbouring temperatures   
  int chi2_stride = 10000;         // Frequency of calculating chi squared values
  int ckpt_stride = 100;           // Frequency of saving checkpoints

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  MC_obj.set_checkpoint(ckpt_stride, "sampler.ckpt");
  // Run the sampler with the given settings
  MC_obj.run_sampler(L_obj, Number_of_steps, 
                    temp_stride, chi2_stride, "chain.dat", "lklhd.dat", "chi2.dat", 
                    means, ranges, var_names, false);

  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}



/*! 
  \file  examples/eggbox_mcmc_sampling.cpp
  \details 
  
  \code

#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>

int main(int argc, char* argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  
  //Dynamically allocate a prior for each variable and add it to the container
  //Here we are using a flat prior for all the parameters and confine them 
  //to the interval [-2.5*pi, 2.5*pi]
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  P.push_back(new Themis::prior_linear(-2.5*M_PI, 2.5*M_PI));
  

  //Set the variable transformations.
  //Here we are using no coordonated transformations on the parameters
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());


  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_eggbox(5.0));


  //Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  //Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  //Making a Themis sampler object. Here the affine invariant tempered 
  //Markov Chain Monte Carlo method is used. The seed of the random number 
  //generator is passed to the constructor.
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42);

  
  //The means and standard deviations used to initialize the MCMC walkers.
  //For each parameter the walkers would be drawn from a gaussian distribution
  //with the given mean and standard deviation. This provides the starting point
  //for the walkers.
  std::vector<double> means, ranges;
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);
  ranges.push_back(1.0);


  // vector to hold the name of variables, if the names are provided it would  
  // be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("test_parameter1");
  var_names.push_back("test_parameter2");
  var_names.push_back("test_parameter3");
  var_names.push_back("test_parameter4");
  var_names.push_back("test_parameter5");


  int Number_of_chains = 100;      // Number of walkers at each temperature
  int Number_of_temperatures = 5;  // Number of temperatures in the parallel tempering
  int cpu_per_likelihood = 1;      // Number of processes used to evaluate a single likelihood 
  int Number_of_steps = 8000;      // Number of monte carlo steps 
  int temp_stride = 50;            // Number of MCMC steps for communication among neighbouring temperatures   
  int chi2_stride = 10000;         // Frequency of calculating chi squared values
  int ckpt_stride = 100;           // Frequency of saving checkpoints

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  MC_obj.set_checkpoint(ckpt_stride, "sampler.ckpt");
  // Run the sampler with the given settings
  MC_obj.run_sampler(L_obj, Number_of_steps, 
                    temp_stride, chi2_stride, "chain.dat", "lklhd.dat", "chi2.dat", 
                    means, ranges, var_names, false);

  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}
  \endcode
  
*/


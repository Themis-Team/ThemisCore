/*! 
  \file validation/lorentz_mcmc_sampling.cpp
  \author Roman Gold & Paul Tiede & Mansour Karami
  \date Feb 2020
  
  \internal
  \validation Testing the differential evolution tempering scheme
  \endinternal
  
  \brief Samples a 1D Lorentzian function 

  \details Testing the ability of the sampler to sample likelihood distributions with long tails and unisotropic gradients.
  The natural logarithm of the likelihood is  given by:

  \f$\log{(L(\mathbf{x}))} = -100*(y-x^2)^2)+(1-x)^2)\f$

  Using the output chain file the marginalized distributions are calculated and plotted:

  \image html sampler-lorentz-triangle.png "Marginalized posterior probabilty distribution"

*/


#include "likelihood.h"
#include "sampler_differential_evolution_deo_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
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
  
  // Dynamically allocate a prior for each variable and add it to the container
  // Here we are using a flat prior for all the parameters and confine them 
  // to the interval [-2.5*pi, 2.5*pi]
  P.push_back(new Themis::prior_linear(-2, 2));
  P.push_back(new Themis::prior_linear(-2, 2));
  P.push_back(new Themis::prior_linear(-2, 2));
  // P.push_back(new Themis::prior_linear(-2, 2));
  // P.push_back(new Themis::prior_linear(-2, 2));
  

  // Set the variable transformations.
  // Here we are using no coordinate transformations on the parameters
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  // T.push_back(new Themis::transform_none());
  // T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_lorentzian());


  // Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  // Making a Themis sampler object. Here the tempered, differential evolution+deo
  // Markov Chain Monte Carlo method is used. The seed of the random number 
  // generator is passed to the constructor.
  // Themis::sampler_differential_evolution_deo_tempered_MCMC MC_obj(42);
  Themis::sampler_differential_evolution_tempered_MCMC MC_obj(42+4);

  
  // The means and standard deviations used to initialize the MCMC walkers.
  // For each parameter the walkers would be drawn from a Gaussian distribution
  // with the given mean and standard deviation. This provides the starting point
  // for the walkers.
  std::vector<double> means, ranges;
  means.push_back(0.0);
  means.push_back(0.0);
  means.push_back(0.0);
  // means.push_back(0.0);
  // means.push_back(0.0);
  
  ranges.push_back(0.1);
  ranges.push_back(0.1);
  ranges.push_back(0.1);
  // ranges.push_back(0.1);
  // ranges.push_back(0.1);


  // vector to hold the name of variables, if the names are provided it would  
  // be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("$x_0$");
  var_names.push_back("$x_1$");
  var_names.push_back("$x_2$");


  int Number_of_chains = 24;      // Number of walkers at each temperature
  int Number_of_temperatures = 24;  // Number of temperatures in the parallel tempering

  int cpu_per_likelihood = 1;      // Number of processes used to evaluate a single likelihood 
  int temp_stride = 50;            // Number of MCMC steps for communication among neighbouring temperatures   
  int Number_of_steps = temp_stride*100;      // Number of monte carlo steps 
  int chi2_stride = 1000;         // Frequency of calculating chi squared values
  //int ckpt_stride = 1000;           // Frequency of saving checkpoints
  int verbosity = 1;

  //int nthin = 10; // only save every nthin step. autocorrelation in 25-50 D problems tends to be ~50-500

  // int nrounds = 7; // Number of rounds to run. You should see the rejection rate variance decrease over each round. If it doesn't you need to run longer for optimal ladder
  // int b = 2; // Geometric increase for number of steps for each each, last round has Number_of_steps*b^(nrounds-1) steps
  // double initial_geometric_spacing = 1.15; // Initial geometric spacing, be careful don't make this too big! I might change this because geometric is pretty aggressive.
  // MC_obj.set_annealing_schedule(nrounds, b, initial_geometric_spacing);

  // The sampler can read in the ladder from a previous run and 
  // Use that for starting.
  // std::string annealing_file = "annealing.dat";
  // MC_obj.read_initial_ladder(annealing_file);

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  // MC_obj.set_checkpoint(ckpt_stride, "sampler.ckpt");

  // Run DE sampler with the given settings
  MC_obj.run_sampler(L_obj, Number_of_steps, temp_stride, chi2_stride, 
                     "chain-lorentz-2d.dat", "lklhd-lorentz-2d.dat", "chi2-lorentz-2d.dat",  
                     means, ranges, var_names, false, 8, verbosity, true);

  // MC_obj.run_sampler(L_obj, Number_of_steps, nthin, temp_stride, chi2_stride, 
  //                    "chain-lorentzian-deo.dat", "lklhd-lorentzian-deo.dat", "chi2-lorentzian-deo.dat", "annealing.dat", 
  //                    means, ranges, var_names, false, 8, verbosity);

  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}



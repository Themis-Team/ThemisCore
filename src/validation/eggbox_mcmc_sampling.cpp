/*! 
  \file validation/eggbox_mcmc_sampling.cpp
  \author Mansour Karami
  \date June 2017
  
  \internal
  \validation Testing the Affine invariant parallel tempered MCMC sampling routine
  \endinternal
  
  \brief Samples an extremely muti-modal and spiky likelihood surface in five dimentions

  \details Testing the ability of the sampler to sample multi-modal likelihood distributions.
  This main file samples a five dimensional egg-box likelihood with 3125 modes. 
  The natural logarithm of the likelihood is  given by:

  \f$\log{(L(\mathbf{x}))} = -2.0 * (2.0 + \prod_{n=1}^{5} cos(x_i))^{5} \f$

  Using the output chain file the marginalized distributions are calculated and plotted:

  \image html sampler-eggbox-triangle.png "Marginalized posterior probabilty distribution"

  Additionally we used a likelihood composed of 16 well-separated gaussians in two dimensions
  to show we can recover the relative heigth of the peaks. All the gaussians were the same except 
  for one with the likelihood nine times as large. Here is the triangle plot showing the 
  likelihood distribution:

  \image html sampler-multi-gaussian-triangle.png "Marginalized posterior probabilty distribution"

  The following plot shows the integrated probability successfully recovered for each gaussian peak using the 
  output MCMC chain file. As can be seen all the peaks are identical except for one  that is 
  nine times taller.

  \image html sampler-multi-gaussian-relative.png "Recovered integrated likelihood for each peak"

*/


#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
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
  //Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MC_obj(42+world_rank);

  
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
  
  ranges.push_back(0.1);
  ranges.push_back(0.1);
  ranges.push_back(0.1);
  ranges.push_back(0.1);
  ranges.push_back(0.1);


  // vector to hold the name of variables, if the names are provided it would  
  // be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("test_parameter1");
  var_names.push_back("test_parameter2");
  var_names.push_back("test_parameter3");
  var_names.push_back("test_parameter4");
  var_names.push_back("test_parameter5");


  int Number_of_chains = 128; //100; //128;      // Number of walkers at each temperature
  int Number_of_temperatures = 8; //5; //8;  // Number of temperatures in the parallel tempering
  int cpu_per_likelihood = 1;      // Number of processes used to evaluate a single likelihood 
  int Number_of_steps = 100000; //10000; //100000;      // Number of monte carlo steps 
  int temp_stride = 50;            // Number of MCMC steps for communication among neighbouring temperatures   
  int chi2_stride = 10000;         // Frequency of calculating chi squared values
  int ckpt_stride = 10000;           // Frequency of saving checkpoints
  //int verbosity = 0;

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  MC_obj.set_checkpoint(ckpt_stride, "sampler.ckpt");
  // Run the sampler with the given settings
  MC_obj.run_sampler(L_obj, Number_of_steps, temp_stride, chi2_stride, "chain.dat", "lklhd.dat", "chi2.dat", means, ranges, var_names, false);
  /*
  MC_obj.run_sampler(L_obj, Number_of_steps, 
		     temp_stride, chi2_stride, 
		     "chain.dat", "lklhd.dat", "chi2.dat", 
		     means, ranges, var_names,
		     false, 6, verbosity, true);
  */
  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}






/*! 
  \file validation/eggbox_mcmc_sampling.cpp
  \author Paul Tiede & Mansour Karami
  \date Feb 2020
  
  \internal
  \validation Testing the DEO differential evolution tempering scheme
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
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>

int main(int argc, char* argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  
  //Dynamically allocate a prior for each variable and add it to the container
  //Here we are using a flat prior for all the parameters and confine them 
  //to the interval [-2.5*pi, 2.5*pi]
  P.push_back(new Themis::prior_linear(-8, 8));
  P.push_back(new Themis::prior_linear(-8, 8));
  P.push_back(new Themis::prior_linear(-8, 8));
  P.push_back(new Themis::prior_linear(-8, 8));
  P.push_back(new Themis::prior_linear(-8, 8));
  

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
  
  // This creates a power-tempered likelihood object
  Themis::likelihood_power_tempered L_temp(L_obj);
  


  
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




  //Making a Themis sampler object. Here the affine invariant tempered 
  //Markov Chain Monte Carlo method is used. The seed of the random number 
  //generator is passed to the constructor.
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(42+world_rank, L_temp, var_names, means.size());
  // If you want to use NUTS then uncomment this line and comment out the one above
  //Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO(42+world_rank, L_temp, var_names, means.size());
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(means);

  //We can also change the annealing schedule.
  double initial_spacing = 1.15; //initial geometric spacing.
  DEO.set_annealing_schedule(initial_spacing);
  

  int swap_stride = 10; // How often the chain swaps 10 is a reasonable default if not limited by MPI communication
  int num_swaps = 10;   // The initial number of swaps in the first round
  DEO.set_deo_round_params(num_swaps, swap_stride);
  
  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat"); 
  


  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "sampler_summary.txt");

  // Set the ckpt file
  // We will ckpt every 20 swaps
  DEO.set_checkpoint(101, "MCMC.ckpt");
  //DEO.read_checkpoint("MCMC.ckpt"); 
  // Now run the sampler!
  int num_thin = 1;     // How often an mcmc step is saved to the output files. Here we will save everything
  int refresh = 5;     // Number of mcmc steps before stdout is refreshed
  int num_rounds = 10; //Number of rounds to run. You should see the rejection rate variance decrease over each round. If it doesn't you need to run longer for optimal ladder
  DEO.set_cpu_distribution(world_size, 1);
  DEO.run_sampler( num_rounds, 
                   num_thin, refresh, 0);

  if (world_rank == 0){ 
    //Now find the best fit from the previous run for fun
    std::cout << "Finding best fit from first rounds\n";
    std::vector<double> best_fit = DEO.find_best_fit();
    std::cout << "Best fit is given by: ";
    for ( size_t i = 0; i < best_fit.size(); ++i )
      std::cout << std::setw(15) << best_fit[i];
    std::cout << std::endl;
    std::cout << "With likelihood: " << L_obj(best_fit) << std::endl;
  }


  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}






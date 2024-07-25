/*!
	\file validation/crescent_image_comparison_themistan.cpp
	\author Jorge A. Preciado, Paul Tiede
	\date April, 2020
		
	\brief Fits a crescent model to visibility amplitude data using the deo tempering sampler with the stan
        exploration kernel. This is a validation and test for it.
	
	\details Compares a geometric crescent model to the visibility amplitude 
	data taken in 2007 and 2009, permitting a day-specific intensity 
	renormalization. The primary fit result is a measure of the size (\f$ R \f$), 
	the  relative thickness (\f$ \psi \f$), and the degree of symmetry (\f$ \tau \f$)
	of the emission region and can be compared to the fit results reported in 
	[Kamruddin and Dexter 2013](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stt1068).
	
	The resulting parameter distribution is:
		
	\image html plots/validation/Crescent-Triangle.png "Triangle plot for the marginalized posterior probabilty distribution showing the likely parameter values and associated confidence contours."
	
	
	\n Note that the intensity normalization is solved for analytically in the 
	likelihood_marginalized_visibility_amplitude, and thus the intrinsic 
	normalization is fixed near unity by design.
	
	
*/


#include "data_visibility_amplitude.h"
#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"
#include "utils.h"

// Standard Libraries
/// @cond
#include <mpi.h>
#include <memory> 
#include <string>
#include <vector>
/// @endcond



int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << "MPI Initiated - Processor Node: " << world_rank 
            << " of " << world_size << "executing main." << std::endl;
  


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude d2007(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  d2007.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  Themis::data_visibility_amplitude d2009_095(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  Themis::data_visibility_amplitude d2009_096(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  Themis::data_visibility_amplitude d2009_097(Themis::utils::global_path("eht_data/VM_2009_097.d"));


  // Choose the model to compare
  Themis::model_image_crescent intrinsic_image;
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);
	
  // Choose the image size
  //intrinsic_image.set_image_resolution(32);
	
  // Use numerical Visibilities
  // intrinsic_image.use_numerical_visibilities();
	
  // Container of base prior class pointers
  // and prior means and ranges
  double crescent_size = 28. * 1.e-6 /3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  std::vector<double> means;
  P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
  means.push_back(1.0);
	
  P.push_back(new Themis::prior_linear(0.01*crescent_size,3.0*crescent_size)); // Overall size R
  means.push_back(crescent_size);
	
  P.push_back(new Themis::prior_linear(0.01,0.99)); // psi
  means.push_back(0.10);

  P.push_back(new Themis::prior_linear(0.01,0.99)); // tau
  means.push_back(0.10);

  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
  means.push_back(0.4*M_PI);


  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$R$");
  var_names.push_back("$\\psi$");
  var_names.push_back("$\\tau$");
  var_names.push_back("$\\phi$");
	
  // Applying the coordinate transformation on the initial values
  Themis::transform_none Trans;
  for(unsigned int i = 0 ; i < means.size(); ++i)
  {
    means[i] = Trans.forward(means[i]);
  } 



  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  
  Themis::likelihood L_obj(P, L, W);
  Themis::likelihood_power_tempered L_temp(L_obj);

  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO(42+world_rank*10, L_temp, var_names, means.size());

  //If you want to access the exploration sampler to change some setting you can!
  Themis::sampler_stan_adapt_diag_e_nuts_MCMC* hmc = DEO.get_sampler();

  //Lets say you want to change the inverse matrix for the sampler. You can!
  std::vector<double> inverse_metric;
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.9);
  inverse_metric.push_back(1.1);
  hmc->set_initial_inverse_metric(inverse_metric);
  
  DEO.set_cpu_distribution(world_size, 1);
  
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(means);

  //We can also change the annealing schedule.
  double initial_spacing = 1.15; //initial geometric spacing.
  int swap_stride = 10;
  int num_swaps = 10;
  DEO.set_annealing_schedule(initial_spacing);
  DEO.set_deo_round_params(num_swaps, swap_stride);


  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "sampler_summary.txt");

  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat"); 

  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  int num_thin = 1;
  bool save_warmup = true;
  int nadapt = 500; //number of adaptation steps for the exploration kernel.
  hmc->set_adaptation_parameters(nadapt, save_warmup);

  DEO.set_checkpoint(100, "MCMC.ckpt");
  //DEO.read_checkpoint("MCMC.ckpt");
  // Run the sampler with the given settings
  int num_rounds = 8; //number of adaptation rounds.

  
  //We can also restore the sampler's state from a previous run using 
  //DEO.read_checkpoint("MCMC.ckpt")
  DEO.run_sampler( num_rounds, 
                   num_thin, 10, 0);

  
  //Now find the best fit from the previous run for fun
  std::cout << "Finding best fit from first rounds\n";
  std::vector<double> best_fit = DEO.find_best_fit();
  std::cout << "Best fit is given by: ";
  for ( size_t i = 0; i < best_fit.size(); ++i )
    std::cout << std::setw(15) << best_fit[i];
  std::cout << std::endl;
  std::cout << "With likelihood: " << L_obj(best_fit) << std::endl;
  //Lets read in the checkpoint and restart from there
  DEO.read_checkpoint("MCMC.ckpt");
  //Also lets turn off the adaptation here
  hmc->set_adaptation_parameters(0,true);
  num_rounds = 2; //number of adaptation rounds.
  DEO.run_sampler( num_rounds, 
                   num_thin, 10, 0);
  
  //Destruct the sampler to prevent MPI errors when exiting main.
  DEO.mpi_cleanup();

  // Finalize MPI
  MPI_Finalize();
  return 0;
}

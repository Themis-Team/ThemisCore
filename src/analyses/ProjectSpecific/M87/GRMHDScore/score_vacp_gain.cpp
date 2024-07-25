/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Hung-Yi Pu
    \date  Nov, 2018
    \brief test model_image_score clas by mcmc runs *with* gain calibration
    \details 
*/

#include "data_visibility_amplitude.h"
#include "model_image_score.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated in rank: " << world_rank << std::endl;

  // Read in data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("eht_data/ER5/M87_VM3598.d"),"HH");
  Themis::data_closure_phase CP(Themis::utils::global_path("eht_data/ER5/M87_CP3598.d"));
  
  // Choose the model to compare
  Themis::model_image_score image(Themis::utils::global_path("sim_data/Score/example_image.dat"),Themis::utils::global_path("sim_data/Score/README.txt"));
				  
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
    
  P.push_back(new Themis::prior_linear(0.1,10.)); // total I
  means.push_back(0.5);  
  ranges.push_back(0.1);
  
  P.push_back(new Themis::prior_linear(0.1,100.)); // (M/D) in uas
  means.push_back(4.);  
  ranges.push_back(1.);
  
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
  means.push_back(2.2);
  ranges.push_back(0.5);


  // vector to hold the name of variables, if the names are provided it would be added
  // as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("I");
  var_names.push_back("M/D");
  var_names.push_back("PA");

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  // Visability amplitudes with gain correction
  std::vector<std::string> stations = Themis::utils::station_codes("uvfits 2017");
  std::vector<double> gain_sigmas(stations.size(),0.2);
  gain_sigmas[4] = 1.0; // LMT is poorly calibrated
  Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(VM,image,stations,gain_sigmas);
  L.push_back(&L_ogva);

  // Closure phases
  L.push_back(new Themis::likelihood_closure_phase(CP,image));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);


  // Output residual data
  L_obj(means);
  L[0]->output_model_data_comparison("VA_residuals.d");
  L[1]->output_model_data_comparison("CP_residuals.d");
  L_ogva.output_gain_corrections("gain_corrections.d");
  

  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 20;
  int Number_of_temperatures = 4;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 2000;
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;

  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);

  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Run the Sampler
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain.dat", "Lklhd.dat", "Chi2.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);

  // Get the best fit and produce residual/gain files
  std::vector<double> pmax = MCMC_obj.find_best_fit("Chain.dat","Lklhd.dat");
  L_obj(pmax);
  L[0]->output_model_data_comparison("VA_residuals.d");
  L[1]->output_model_data_comparison("CP_residuals.d");
  L_ogva.output_gain_corrections("gain_corrections.d");

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

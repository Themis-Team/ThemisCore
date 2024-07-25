//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "utils.h"

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated in rank: " << world_rank << std::endl;


  // Read in visibility amplitude data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge04/VM01.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/Challenge04/CP01.d"));
  
  // Choose the model to compare
  Themis::model_image_crescent intrinsic_image;
  //Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);
  
  // Use analytical Visibilities
  intrinsic_image.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  
  // Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(3.50);
  ranges.push_back(0.01);
  
  // Overall size R
  double crescent_size = 22.5e-6/ 3600. * (M_PI/180.);
  P.push_back(new Themis::prior_linear(0.0*crescent_size,5.0*crescent_size));
  means.push_back(crescent_size);
  ranges.push_back(0.1*crescent_size);
  
  // psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.16);
  ranges.push_back(0.01);
  
  // tau
  P.push_back(new Themis::prior_linear(0.01,0.99));
  means.push_back(0.990);
  ranges.push_back(0.001);
  
  // Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(33.7*M_PI/180.0);
  ranges.push_back(1.0*M_PI/180.0);


  // vector to hold the name of variables
  std::vector<std::string> var_names = {"$V_{0}$", "$R$", "$\\psi$", "$\\tau$", "$\\xi$"};
  

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_visibility_amplitude(VM,intrinsic_image));
  //L.push_back(new Themis::likelihood_visibility_amplitude(VM,image));

  //Closure Phases
  L.push_back(new Themis::likelihood_closure_phase(CP,intrinsic_image));
  
  
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);
  
  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 16;
  int Number_of_temperatures = 4;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 7500; 
  int Temperature_stride = 50;
  int Chi2_stride = 1;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;


  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"Crescent.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain-Crescent.dat", "Lklhd-Crescent.dat", "Chi2-Crescent.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

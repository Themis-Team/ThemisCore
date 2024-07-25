//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_xsringauss.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_smooth.h"
//#include "model_ensemble_averaged_scattered_image.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "utils.h"

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated in rank: " << world_rank << std::endl;


  // Read in visibility amplitude data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/VM_challenge_2_seed_2_simobs_netcal_scanavg_wo_shortB.d"),"HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/CP_challenge_2_seed_2_simobs_netcal_scanavg_wo_shortB.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_xsringauss xrg;
  Themis::model_image_smooth smxrg(xrg);
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smxrg);
  
  // Use analytical Visibilities
  xrg.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(1.0);
  ranges.push_back(0.5);
  //   Overall size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(20*uas2rad);
  ranges.push_back(1*uas2rad);
  //   relative thickness
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
  //   eccentricity
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
  //   fading
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
  //   g_{ax}
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
  //   a_{q}
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
  //   g_{q}
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.1);
   //   rotation
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0.0);
  ranges.push_back(1e-4);
 
 //   overall Gauss sizex
  P.push_back(new Themis::prior_linear(0.0,40*uas2rad));
  means.push_back(1*uas2rad);
  ranges.push_back(1e-1*uas2rad);
  //   assymetry
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(1e-1);
  // smooth rotation
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(1e-4); 
  
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);
  means[0] = 0.51879537;
  means[1] = 1.3854834e-10;
  means[2] = 0.4726901;
  means[3] = 0.00021086867;
  means[4] = 0.9986462;
  means[5] = 0.90726367;
  means[6] = 0.23845807;
  means[7] = 0.0045044578;
  means[8] = 0.00045807701;
  means[9] = 6.5226908e-13;
  means[10] = 0.038978467;
  means[11] = -0.0017062944;
  means[12] = 3.6701284e-18;
  means[13] = -2.9124753e-18;
 
  std::cerr << "here\n";


  // vector to hold the name of variables
  std::vector<std::string> var_names;
  

  // Set the likelihood functions
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 10% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

  Themis::likelihood_optimal_gain_correction_visibility_amplitude lva(VM,intrinsic_image,station_codes,station_gain_priors);


  std::vector<Themis::likelihood_base*> L;
  L.push_back(&lva);

  //Closure Phases
  L.push_back(new Themis::likelihood_closure_phase(CP,intrinsic_image));
  
  
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);
  
  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Output residual data
  L_obj(means);
  L[0]->output_model_data_comparison("VA_residuals.d");
  L[1]->output_model_data_comparison("CP_residuals.d");
  lva.output_gain_corrections("gain_corrections_direct.d");

  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 120;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 20000; 
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  //bool restart_flag = true;
  int out_precision = 8;
  int verbosity = 0;


  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain.dat", "Lklhd.dat", "Chi2.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

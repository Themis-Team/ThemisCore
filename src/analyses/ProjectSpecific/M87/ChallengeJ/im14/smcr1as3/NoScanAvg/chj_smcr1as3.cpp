//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_crescent.h"
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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ/VM_challenge_14_seed_2_simobs_netcal.d"), "HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ/CP_challenge_14_seed_2_simobs_netcal.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent cr;
  Themis::model_image_smooth smcr(cr);
  Themis::model_image_asymmetric_gaussian asg1,asg2, asg3;
  Themis::model_image_sum intrinsic_image;
  
  intrinsic_image.add_model_image(smcr);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);
  intrinsic_image.add_model_image(asg3);
  
  // Use analytical Visibilities
  cr.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(1.0);
  ranges.push_back(1e-4);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(20.0*uas2rad);
  ranges.push_back(1e-1*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.1);
  ranges.push_back(1e-4);
  //   tau
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.1);
  ranges.push_back(1e-4);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(1e-4); 

  //   smooth size
  P.push_back(new Themis::prior_linear(0,40*uas2rad));
  means.push_back(0.1*uas2rad);
  ranges.push_back(1e-1*uas2rad);
  //   smooth asymmetry
  P.push_back(new Themis::prior_linear(0,1.0));
  means.push_back(0.1);
  ranges.push_back(1e-2);
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
  
  // Gaussian component params
  for (size_t i=0; i<3; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
    //   Size
    P.push_back(new Themis::prior_logarithmic(1e-5*uas2rad,1e4*uas2rad));
    means.push_back(10*uas2rad);
    ranges.push_back(1e-3*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.2);
    ranges.push_back(1e-2);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(1e-3*M_PI);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-2*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-2*uas2rad);
  }

  means[0] = 0.55308129;
  means[1] = 1.0639071e-10;
  means[2] = 0.020394717;
  means[3] = 0.35312687;
  means[4] = 3.1360398;
  means[5] = 2.2216742e-11;
  means[6] = 0.50201836;
  means[7] = 2.3359995;
  means[8] = 4.5921035e-18;
  means[9] = -1.1474069e-18;
  means[10] = 0.11352061;
  means[11] = 1.4409359e-11;
  means[12] = 0.89812628;
  means[13] = 3.1338948;
  means[14] = 8.8063603e-11;
  means[15] = 2.1491334e-11;
  means[16] = 0.082196194;
  means[17] = 1.4004187e-11;
  means[18] = 0.95886235;
  means[19] = 3.1339151;
  means[20] = 1.1039506e-10;
  means[21] = 1.9849083e-10;
  means[22] = 0.37788612;
  means[23] = 1.4700968e-10;
  means[24] = 0.70989637;
  means[25] = 0.19958053;
  means[26] = 2.197332e-11;
  means[27] = 2.3604528e-10;

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
  lva.output_gain_corrections("gain_correction.d");

  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 240;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 200000; 
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

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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/VM_challenge_2_seed_2_simobs_netcal_scanavg_wo_shortB.d"),"HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/CP_challenge_2_seed_2_simobs_netcal_scanavg_wo_shortB.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent cr;
  Themis::model_image_smooth smcr(cr);
  Themis::model_image_asymmetric_gaussian asg1;
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smcr);
  intrinsic_image.add_model_image(asg1);
  
  // Use analytical Visibilities
  cr.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(1.0);
  ranges.push_back(0.5);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(20.0*uas2rad);
  ranges.push_back(10.0*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.5);
  //   tau
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.5);
  ranges.push_back(0.5);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(M_PI); 

  //   overall Gauss size
  P.push_back(new Themis::prior_linear(0.0,40*uas2rad));
  means.push_back(1*uas2rad);
  ranges.push_back(0.5*uas2rad);
  //   assymetry
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.1);
  ranges.push_back(0.1);
  // smooth rotation
  P.push_back(new Themis::prior_linear(0.0,M_PI));
  means.push_back(0.5*M_PI);
  ranges.push_back(M_PI); 
  
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);

  // Gaussian component params
  for (size_t i=0; i<1; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-1);
    //   Size
    P.push_back(new Themis::prior_linear(0.0,2000*uas2rad));
    means.push_back(50*uas2rad);
    ranges.push_back(30*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.999));
    means.push_back(0.5);
    ranges.push_back(0.5);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    //   x offset
    P.push_back(new Themis::prior_linear(-500*uas2rad,500*uas2rad));
    means.push_back(0.0);
    ranges.push_back(200*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-500*uas2rad,500*uas2rad));
    means.push_back(0.0);
    ranges.push_back(200*uas2rad);
  }
  

  means[0] = 0.55779434;
  means[1] = 1.1083689e-10;
  means[2] = 0.0682062;
  means[3] = 0.95874013;
  means[4] = -2.5726231;
  means[5] = 2.1970021e-11;
  means[6] = 0.081331282;
  means[7] = M_PI-0.22225569;
  means[8] = -3.9788969e-18;
  means[9] = -1.3344795e-19;

  /*
  means[0] = 0.53072944;
  means[1] = 1.1791518e-10;
  means[2] = 0.19007631;
  means[3] = 0.96385278;
  means[4] = -2.6893768;
  means[5] = 1.9986741e-11;
  means[6] = 0.013626458;
  means[7] = 2.5895044;
  means[8] = -1.2756771e-20;
  means[9] = -4.0516186e-18;
  means[10] = 0.2191536;
  means[11] = 2.426272e-10;
  means[12] = 0.50232621;
  means[13] = 2.6608604;
  means[14] = 2.0537732e-10;
  means[15] = 2.256845e-10;
  */
  
  for (size_t i=0; i<8; ++i)
    ranges[i] = 1e-4;
  ranges[1] *= uas2rad;
  ranges[5] *= uas2rad;

  /*
  for (size_t i=10; i<16; ++i)
    ranges[i] = 1e-4; 
  ranges[11] *= uas2rad;
  ranges[14] *= uas2rad;
  ranges[15] *= uas2rad;
  */
  
  /*
  means[0] = 0.56881664;
  means[1] = 1.0833223e-10;
  means[2] = 0.02403943;
  means[3] = 0.96032279;
  means[4] = -2.5814664;
  means[5] = 2.2633358e-11;
  means[6] = 0.079609433;
  means[7] = -0.44292309;
  means[8] = 3.8729256e-18;
  means[9] = -1.6938951e-18;
  std::cerr << "here\n";
  */

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
  lva.output_gain_corrections("gain_corrections.d");

  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank + 1234);
  
  // Generate a chain
  int Number_of_chains = 120;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 10000; 
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

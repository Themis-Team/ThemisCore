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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ/VM_challenge_14_seed_2_simobs_netcal_scanavg.d"), "HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ/CP_challenge_14_seed_2_simobs_netcal_scanavg.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_xsringauss xg;
  Themis::model_image_smooth smxg(xg);
  Themis::model_image_asymmetric_gaussian asg1;
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smxg);
  intrinsic_image.add_model_image(asg1);
  
  // Use analytical Visibilities
  xg.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(0.7);
  ranges.push_back(0.01);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(40*uas2rad);
  ranges.push_back(1e-2*uas2rad);
  //   psi (relative thickness
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.1);
  ranges.push_back(1e-4);  
  //eccentricity
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.1);
  ranges.push_back(1e-4);
  //   fading parameter
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.1);
  ranges.push_back(1e-4);
  //gax
  P.push_back(new Themis::prior_linear(1e-3,10.0));
  means.push_back(1.0);
  ranges.push_back(1e-1);
  //ellipse axial ratio
  P.push_back(new Themis::prior_linear(1e-2,1.0));
  means.push_back(0.9);
  ranges.push_back(1e-2);
  //Ratio of the Gaussian flux to the total flux
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.1);
  ranges.push_back(1e-2);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(1e-4); 

  //   smooth size
  P.push_back(new Themis::prior_linear(0,40*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-1*uas2rad);
  //   smooth asymmetry
  P.push_back(new Themis::prior_linear(0,0.99));
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
  for (size_t i=0; i<1; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
    //   Size
    P.push_back(new Themis::prior_logarithmic(1e-6*uas2rad,1e4*uas2rad));
    means.push_back(10*uas2rad);
    ranges.push_back(1e-2*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.2);
    ranges.push_back(0.1);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(1e-2);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-2*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-2*uas2rad);
  }


  means[0] = 0.82650758;
  means[1] = 1.0595691e-10;
  means[2] = 0.048779843;
  means[3] = 0.63403719;
  means[4] = 0.49572071;
  means[5] = 0.0063052788;
  means[6] = 0.2133989;
  means[7] = 0.12589648;
  means[8] = 2.8628668;
  means[9] = 2.7378098e-11;
  means[10] = 0.45732885;
  means[11] = -0.24475294;
  means[12] = 4.4687645e-18;
  means[13] = 1.4225752e-18;
  means[14] = 0.17620538;
  means[15] = 1.3035459e-11;
  means[16] = 0.98253123;
  means[17] = 3.1326771;
  means[18] = 9.4102749e-11;
  means[19] = 5.3918162e-11;

  means[0] = 0.82948101;
  means[1] = 1.0939183e-10;
  means[2] = 0.13214145;
  means[3] = 0.00024078721;
  means[4] = 0.44300996;
  means[5] = 0.0063395849;
  means[6] = 0.73112489;
  means[7] = 0.12199972;
  means[8] = 2.8788856;
  means[9] = 2.8568078e-11;
  means[10] = 0.34507012;
  means[11] = -0.21299518;
  means[12] = -2.7877281e-18;
  means[13] = 1.6640027e-18;
  means[14] = 0.16494312;
  means[15] = 1.0318398e-11;
  means[16] = 0.9872243;
  means[17] = 3.1217698;
  means[18] = 9.4617038e-11;
  means[19] = 5.9880344e-11;


  means[0] = 0.85969217;
  means[1] = 1.0945059e-10;
  means[2] = 0.12741391;
  means[3] = 0.00047901809;
  means[4] = 0.4355589;
  means[5] = 0.0043064224;
  means[6] = 0.21283315;
  means[7] = 0.1231935;
  means[8] = 2.8887378;
  means[9] = 2.8804566e-11;
  means[10] = 0.3638993;
  means[11] = -0.22887934;
  means[12] = 9.7473548e-19;
  means[13] = -2.2088365e-18;
  means[14] = 0.15948135;
  means[15] = 1.004311e-11;
  means[16] = 0.9881562;
  means[17] = 3.1318349;
  means[18] = 9.4081176e-11;
  means[19] = 6.2858134e-11;

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
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 120;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 100000; 
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

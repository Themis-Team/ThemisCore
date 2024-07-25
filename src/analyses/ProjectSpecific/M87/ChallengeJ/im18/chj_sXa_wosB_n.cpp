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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/VM_challenge_18_seed_2_simobs_netcal_wo_shortB.d"),"HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/CP_challenge_18_seed_2_simobs_netcal_wo_shortB.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_xsringauss crescent;
  Themis::model_image_smooth smoothed_crescent(crescent);
  //  1 Asymmetric Gaussian
  Themis::model_image_asymmetric_gaussian asg1;
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smoothed_crescent);
  intrinsic_image.add_model_image(asg1);
  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  
  // Crescent params
  //   Itot
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(0.6);
  ranges.push_back(0.3);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(50*uas2rad);
  ranges.push_back(20*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.2);
  ranges.push_back(0.2);
  //   tau
  P.push_back(new Themis::prior_linear(0.01,0.99));
  means.push_back(0.2);
  ranges.push_back(0.2);
  //   f
  P.push_back(new Themis::prior_linear(0.00,1.0));
  means.push_back(0.5);
  ranges.push_back(0.5);
  //   g
  P.push_back(new Themis::prior_linear(0,3.0));
  means.push_back(0.1);
  ranges.push_back(0.1);
  //   a
  P.push_back(new Themis::prior_linear(0.0,100.0));
  means.push_back(5.0);
  ranges.push_back(2.5);
  //   Ig
  P.push_back(new Themis::prior_linear(0.0,1.0));
  means.push_back(0.3);
  ranges.push_back(0.3);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(M_PI);
  //   Smoothing kernel size
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(1*uas2rad);
  ranges.push_back(1*uas2rad);
  //   Smoothing kernel Asymmetry
  P.push_back(new Themis::prior_linear(0.0,0.99));
  means.push_back(0.1);
  ranges.push_back(0.1);
  //   Smoothing kernel phi
  P.push_back(new Themis::prior_linear(0,M_PI));
  means.push_back(0.5*M_PI);
  ranges.push_back(0.5*M_PI);
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);

  // Gaussian component params
  for (size_t i=0; i<1; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-1);
    //   Size
    P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
    means.push_back(10*uas2rad);
    ranges.push_back(4*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.2);
    ranges.push_back(0.1);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(25*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(25*uas2rad);
  }

  // From X alone
  means[0] = 1.0058039;
  means[1] = 1.4604709e-10;
  means[2] = 0.57823497;
  means[3] = 0.010019118;
  means[4] = 0.097198513;
  means[5] = 0.00403841;
  means[6] = 22.720208;
  means[7] = 0.074400702;
  means[8] = -3.1012512;

  // From XA run
  means[0] = 0.94434665;
  means[1] = 1.4956617e-10;
  means[2] = 0.53091369;
  means[3] = 0.010036462;
  means[4] = 0.023057221;
  means[5] = 0.0051242727;
  means[6] = 33.405246;
  means[7] = 0.13500518;
  means[8] = -3.0265084;

  means[12] = -4.1993931e-18;
  means[13] = 5.6566427e-20;
  means[14] = 0.26944123;
  means[15] = 6.3925553e-11;
  means[16] = 0.64160294;
  means[17] = 1.0664751;
  means[18] = -3.4433178e-11;
  means[19] = 2.7499118e-11;

  // Best fit from scan avg. analysis
  means[0] = 1.0154348;
  means[1] = 1.4743398e-10;
  means[2] = 0.52744186;
  means[3] = 0.010095594;
  means[4] = 0.031196556;
  means[5] = 0.0036024594;
  means[6] = 43.196526;
  means[7] = 0.11869564;
  means[8] = -3.030915;
  means[9] = 3.0745885e-12;
  means[10] = 0.97284083;
  means[11] = 0.09023179;
  means[12] = 2.0643484e-18;
  means[13] = -7.181787e-20;
  means[14] = 0.2209328;
  means[15] = 6.442266e-11;
  means[16] = 0.64808397;
  means[17] = 1.0387159;
  means[18] = -4.7821839e-11;
  means[19] = 3.0275522e-11;

  // Best fit w/o scan avg.
  means[0] = 0.83460012;
  means[1] = 1.3746912e-10;
  means[2] = 0.48610872;
  means[3] = 0.010972865;
  means[4] = 0.21788802;
  means[5] = 0.006260063;
  means[6] = 46.359098;
  means[7] = 0.1373948;
  means[8] = -3.1235211;
  means[9] = 2.6152212e-12;
  means[10] = 0.97674754;
  means[11] = 0.54773405;
  means[12] = 2.9856534e-18;
  means[13] = -4.2302646e-19;
  means[14] = 0.075408102;
  means[15] = 1.121265e-11;
  means[16] = 0.98448932;
  means[17] = 0.57532595;
  means[18] = 8.4448745e-11;
  means[19] = 1.8053146e-10;
  
  for (size_t i=0; i<12; ++i)
    ranges[i] = 1e-4;
  ranges[1] *= uas2rad;
  ranges[9] *= uas2rad;
  
  for (size_t i=0,k=14; i<1; ++i)
  {
    for (size_t j=0; j<6; ++j)
      ranges[k++] = 1e-4;
    ranges[i*6+12+3] *= uas2rad;
    ranges[i*6+15+3] *= uas2rad;
    ranges[i*6+16+3] *= uas2rad;
  }
  
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

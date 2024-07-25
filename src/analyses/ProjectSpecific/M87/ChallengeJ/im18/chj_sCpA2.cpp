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

#include <iostream>
#include <iomanip>


int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated in rank: " << world_rank << std::endl;


  // Read in visibility amplitude data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ/VM_challenge_18_seed_2_simobs_netcal_scanavg.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ/CP_challenge_18_seed_2_simobs_netcal_scanavg.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent crescent;
  Themis::model_image_smooth smoothed_crescent(crescent);
  Themis::model_image_asymmetric_gaussian asg1, asg2;
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smoothed_crescent);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);

  std::cerr << "Model size accounting:"
	    << std::setw(15) << crescent.size()
	    << std::setw(15) << smoothed_crescent.size()
	    << std::setw(15) << asg1.size()
	    << std::setw(15) << asg2.size()
	    << std::setw(15) << intrinsic_image.size()
	    << std::endl;

  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(3.4982414);
  ranges.push_back(0.01);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,200*uas2rad));
  means.push_back(50*uas2rad);
  ranges.push_back(20*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.10132669);
  ranges.push_back(1e-4);
  //   tau
  P.push_back(new Themis::prior_linear(0.01,0.99));
  means.push_back(0.79737272);
  ranges.push_back(1e-4);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(-2.0610537);
  ranges.push_back(1e-4);
  //   sig_maj
  P.push_back(new Themis::prior_linear(0*uas2rad,50*uas2rad));
  means.push_back(3.0*uas2rad);
  ranges.push_back(1.0*uas2rad);
  //   sig_min
  P.push_back(new Themis::prior_linear(0*uas2rad,50*uas2rad));
  means.push_back(3.0*uas2rad);
  ranges.push_back(1.0*uas2rad);
  //   sig_PA
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

  means[0] = 1.0963032;
  means[1] = 1.2366599e-10;
  means[2] = 0.33961386;
  means[3] = 0.33358208;
  means[4] = 3.0300356;
  means[5] = 3.6744064e-11;
  means[6] = -5.9250753e-14;
  means[7] = 0.14096327;
  means[8] = 2.0385286e-18;
  means[9] = -5.5314057e-19;

  ranges[0] = 1e-3;
  ranges[1] = 1e-3*uas2rad;
  ranges[2] = 1e-3;
  ranges[3] = 1e-3;
  ranges[4] = 1e-3;
  ranges[5] = 1e-3*uas2rad;
  ranges[6] = 1e-3*uas2rad;
  ranges[7] = 1e-3;
  ranges[8] = 1e-3*uas2rad;
  ranges[9] = 1e-3*uas2rad;

  // Gaussian component params
  for (size_t i=0; i<2; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
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
    ranges.push_back(22*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(22*uas2rad);
  }

  means[0] = 0.80319589;
  means[1] = 1.3617938e-10;
  means[2] = 0.31019228;
  means[3] = 0.20335637;
  means[4] = 3.1409184;

  means[5] = 0.1*uas2rad;
  means[6] = 0.1*uas2rad;
  means[7] = 0.5*M_PI;

  means[8] = -1.5692853e-18;
  means[9] = 2.2581501e-18;
  means[10] = 0.14672332;
  means[11] = 5.0273523e-12;
  means[12] = 0.98079553;
  means[13] = 3.1411693;
  means[14] = 5.3370815e-11;
  means[15] = -3.8768953e-13;
  means[16] = 0.7433829;
  means[17] = 8.1309427e-11;
  means[18] = 0.54864856;
  means[19] = 1.3092872;
  means[20] = -1.5999223e-11;
  means[21] = 3.7884645e-12;

  

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
  

  std::cerr << "Size check:" << std::setw(15) << intrinsic_image.size() << std::setw(15) << means.size() << std::setw(15) << ranges.size() << std::endl;
  
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

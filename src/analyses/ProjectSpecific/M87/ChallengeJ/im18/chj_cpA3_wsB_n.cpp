//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_crescent.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_sum.h"
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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ/VM_challenge_18_seed_2_simobs_netcal.d"),"HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ/CP_challenge_18_seed_2_simobs_netcal.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent crescent;
  Themis::model_image_asymmetric_gaussian asg1, asg2, asg3;
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(crescent);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);
  intrinsic_image.add_model_image(asg3);
  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  asg3.use_analytical_visibilities();
  
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
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);

  // Bestfit C
  means[0] = 0.83586301;
  means[1] = 1.2701297e-10;
  means[2] = 0.3685228;
  means[3] = 0.31715136;
  means[4] = 2.8538888;
  means[5] = 2.6067482e-19;
  means[6] = -3.8608823e-19;

  // Gaussian component params
  for (size_t i=0; i<3; ++i)
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


  // With short baselines
  means[0] = 0.54995125;
  means[1] = 1.3786362e-10;
  means[2] = 0.35695387;
  means[3] = 0.46634189;
  means[4] = 3.0992696;
  means[5] = -1.9430498e-18;
  means[6] = 1.133481e-18;
  means[7] = 0.1928211;
  means[8] = 9.7315206e-12;
  means[9] = 0.95991212;
  means[10] = 3.1357119;
  means[11] = 5.7453778e-11;
  means[12] = 6.7294812e-12;
  means[13] = 0.24531894;
  means[14] = 7.6830881e-11;
  means[15] = 0.21137639;
  means[16] = 0.14179626;
  means[17] = 4.8552522e-12;
  means[18] = -4.7065546e-11;
  means[19] = 0.12117237;
  means[20] = 2.9097794e-11;
  means[21] = 0.95277297;
  means[22] = 0.82760368;
  means[23] = 1.7283022e-12;
  means[24] = 1.1370285e-10;

  // After first run
  means[0] = 0.53210301;
  means[1] = 1.2775462e-10;
  means[2] = 0.40736894;
  means[3] = 0.46742602;
  means[4] = 3.1399189;
  means[5] = -3.3169928e-18;
  means[6] = 4.1062613e-18;
  means[7] = 0.18556987;
  means[8] = 1.0997143e-11;
  means[9] = 0.97866838;
  means[10] = 3.1390635;
  means[11] = 6.7905394e-11;
  means[12] = 1.8148094e-11;
  means[13] = 0.29287298;
  means[14] = 1.0879618e-10;
  means[15] = 0.83633755;
  means[16] = 3.0441236;
  means[17] = -3.2773696e-11;
  means[18] = -1.4896427e-10;
  means[19] = 0.094455954;
  means[20] = 1.0809833e-11;
  means[21] = 0.98955504;
  means[22] = 0.8678422;
  means[23] = 3.9696458e-11;
  means[24] = 1.5905096e-10;


  // After 2nd run
  means[0] = 0.55946528;
  means[1] = 1.2418894e-10;
  means[2] = 0.25823734;
  means[3] = 0.34930561;
  means[4] = 3.1405846;
  means[5] = 2.2591517e-18;
  means[6] = 2.456876e-18;
  means[7] = 0.091568014;
  means[8] = 5.3943003e-12;
  means[9] = 0.98119584;
  means[10] = 3.1411302;
  means[11] = 5.9533529e-11;
  means[12] = -1.3471884e-11;
  means[13] = 0.37326099;
  means[14] = 9.4723056e-11;
  means[15] = 0.93128555;
  means[16] = 3.0831045;
  means[17] = -2.9299707e-11;
  means[18] = -1.2527644e-10;
  means[19] = 0.081810714;
  means[20] = 6.2370513e-12;
  means[21] = 0.98985882;
  means[22] = 0.42517596;
  means[23] = 7.9657227e-11;
  means[24] = 1.9440498e-10;
  
  
  ranges[0] = 1e-3;
  ranges[1] = 1e-3*uas2rad;
  ranges[2] = 1e-3;
  ranges[3] = 1e-3;
  ranges[4] = 1e-3;
  for (size_t i=0; i<3; ++i)
  {
    ranges[7+i*6+0] = 1e-4;
    ranges[7+i*6+1] = 1e-3*uas2rad;
    ranges[7+i*6+2] = 1e-3;
    ranges[7+i*6+3] = 1e-3;
    ranges[7+i*6+4] = 1e-3*uas2rad;
    ranges[7+i*6+5] = 1e-3*uas2rad;
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
  lva.output_gain_corrections("gain_corrections_direct.d");

  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  //Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(43+world_rank);
  
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

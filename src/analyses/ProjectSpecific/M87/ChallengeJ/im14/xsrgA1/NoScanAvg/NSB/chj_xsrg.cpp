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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/VM_challenge_14_seed_2_simobs_netcal_wo_shortB.d"), "HH");

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ_wo_shortB/CP_challenge_14_seed_2_simobs_netcal_wo_shortB.d"));
  


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
  P.push_back(new Themis::prior_linear(1e-3,10));
  means.push_back(1e-1);
  ranges.push_back(1e-3);
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
    ranges.push_back(1e-2);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(1e-3);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-3*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-3*uas2rad);
  }


  means[0] = 0.55364102;
  means[1] = 1.3439133e-10;
  means[2] = 0.40690477;
  means[3] = 0.074094769;
  means[4] = 0.5247578;
  means[5] = 0.97710547;
  means[6] = 0.06886441;
  means[7] = 0.072457904;
  means[8] = 3.0546638;
  means[9] = 1.870086e-11;
  means[10] = 0.37088758;
  means[11] = 1.3218233;
  means[12] = -1.8062523e-18;
  means[13] = -1.7668472e-18;
  means[14] = 0.2114342;
  means[15] = 1.323966e-11;
  means[16] = 0.93094569;
  means[17] = 3.0651344;
  means[18] = 8.7886016e-11;
  means[19] = 2.6378988e-11;

 means[0] = 0.6454295;
  means[1] = 1.3980176e-10;
  means[2] = 0.49118447;
  means[3] = 0.0057778638;
  means[4] = 0.47584817;
  means[5] = 1.5517034;
  means[6] = 0.074567964;
  means[7] = 0.15090427;
  means[8] = 2.9320106;
  means[9] = 1.5027403e-11;
  means[10] = 0.57794067;
  means[11] = 1.6311851;
  means[12] = -2.8371459e-18;
  means[13] = -1.2340534e-18;
  means[14] = 0.24625543;
  means[15] = 1.8844678e-11;
  means[16] = 0.87695555;
  means[17] = 3.0647788;
  means[18] = 9.0810142e-11;
  means[19] = 2.6699917e-11;

  means[0] = 0.73352202;
  means[1] = 1.3746241e-10;
  means[2] = 0.4505414;
  means[3] = 0.011561774;
  means[4] = 0.39143079;
  means[5] = 2.1058437;
  means[6] = 0.3478691;
  means[7] = 0.12933334;
  means[8] = 3.1400506;
  means[9] = 2.4304664e-11;
  means[10] = 0.18252502;
  means[11] = 1.9697644;
  means[12] = -8.7370359e-19;
  means[13] = 1.4333575e-18;
  means[14] = 0.23709016;
  means[15] = 1.3570361e-11;
  means[16] = 0.93753294;
  means[17] = 3.0530468;
  means[18] = 9.3347948e-11;
  means[19] = 2.4633218e-11;


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

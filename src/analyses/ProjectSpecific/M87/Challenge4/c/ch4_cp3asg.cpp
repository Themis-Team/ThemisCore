//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_crescent.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_sum.h"
//#include "model_ensemble_averaged_scattered_image.h"
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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge4/ch4_VA_im3.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/Challenge4/ch4_CP_im3.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent crescent;
  //  3 Gaussians
  Themis::model_image_asymmetric_gaussian asg1, asg2, asg3;
  //  Combined
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
  P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
  means.push_back(1.0602383e-10);
  ranges.push_back(1e-4*uas2rad);
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

  // Gaussian component params
  for (size_t i=0; i<3; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
    //   Size
    P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
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

  // Revised parameters from try2 with massaging
  means[0] = 3.101843;
  means[1] = 1.0369794e-10;
  means[2] = 0.020992347;
  means[3] = 0.93038431;
  means[4] = -2.2249277;
  means[5] = -1.4180623e-18;
  means[6] = 5.9985842e-19;

  means[7] = 0.11533628;
  means[8] = 8.9051413e-12;
  means[9] = 0.5;
  means[10] = 0.5*M_PI;
  means[11] = 4.9613559e-11;
  means[12] = -8.461318e-11;

  means[13] = 0.13716702;
  means[14] = 1.1470684e-11;
  means[15] = 0.5;
  means[16] = 0.5*M_PI;
  means[17] = 3.9982121e-11;
  means[18] = 7.5722561e-11;

  means[19] = 0.14589241;
  means[20] = 6.113503e-13;
  means[21] = 0.5;
  means[22] = 0.5*M_PI;
  means[23] = -8.3860556e-11;
  means[24] = 3.0977006e-12;

  
  for (size_t i=0; i<3; ++i)
  {
    ranges[7+i*6+0] = 1e-4;
    ranges[7+i*6+1] = 1e-2*uas2rad;
    ranges[7+i*6+2] = 0.25;
    ranges[7+i*6+3] = 1.0;
    ranges[7+i*6+4] = 1e-2*uas2rad;
    ranges[7+i*6+5] = 1e-2*uas2rad;
  }

  
  // Best fit from Symmetry run
  means[0] = 2.9494539;
  means[1] = 1.0173364e-10;
  means[2] = 0.0072483397;
  means[3] = 0.97833302;
  means[4] = -1.4939185;
  means[5] = -2.7034661e-18;
  means[6] = -4.27806e-18;
  means[7] = 0.18188791;
  means[8] = 1.3927365e-11;
  means[9] = 0.81408667;
  means[10] = 0.94815462;
  means[11] = 5.061979e-11;
  means[12] = -8.3395685e-11;
  means[13] = 0.18884709;
  means[14] = 1.3322271e-11;
  means[15] = 0.79624604;
  means[16] = 2.0632781;
  means[17] = 4.6507476e-11;
  means[18] = 7.9850849e-11;
  means[19] = 0.17973119;
  means[20] = 1.8249499e-11;
  means[21] = 0.53006503;
  means[22] = 0.16345276;
  means[23] = -9.2629444e-11;
  means[24] = 1.3326718e-12;


  // Best fit
  means[0] = 2.9216184;
  means[1] = 1.0101223e-10;
  means[2] = 0.0071596714;
  means[3] = 0.98900201;
  means[4] = -0.99599819;
  means[5] = -4.2327131e-19;
  means[6] = 4.022126e-18;
  means[7] = 0.19260189;
  means[8] = 1.5975345e-11;
  means[9] = 0.77102823;
  means[10] = 0.99158886;
  means[11] = 4.9719871e-11;
  means[12] = -8.348775e-11;
  means[13] = 0.19751121;
  means[14] = 1.368318e-11;
  means[15] = 0.82747346;
  means[16] = 2.0904832;
  means[17] = 4.8177823e-11;
  means[18] = 8.268271e-11;
  means[19] = 0.18824028;
  means[20] = 1.6039214e-11;
  means[21] = 0.75037576;
  means[22] = 0.05819597;
  means[23] = -9.7159113e-11;
  means[24] = 7.2291046e-13;


  

  
  for (size_t i=0; i<3; ++i)
  {
    ranges[7+i*6+0] = 1e-4;
    ranges[7+i*6+1] = 1e-2*uas2rad;
    ranges[7+i*6+2] = 1e-3;
    ranges[7+i*6+3] = 1e-3;
    ranges[7+i*6+4] = 1e-2*uas2rad;
    ranges[7+i*6+5] = 1e-2*uas2rad;
  }


  // vector to hold the name of variables
  //std::vector<std::string> var_names = {"$V_{0}$", "$R$", "$\\psi$", "$\\tau$", "$\\xi$"};
  std::vector<std::string> var_names;// = {"$V_{0}$", "$R$", "$\\psi$", "$\\tau$", "$\\xi$"};
  

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

  // Output residual data
  L_obj(means);
  L[0]->output_model_data_comparison("VA_residuals.d");
  L[1]->output_model_data_comparison("CP_residuals.d");

  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 120;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 7500; 
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
  MCMC_obj.set_checkpoint(Ckpt_frequency,"Crescent.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain-Crescent.dat", "Lklhd-Crescent.dat", "Chi2-Crescent.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

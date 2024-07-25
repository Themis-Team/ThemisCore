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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge4/ch4_VA_im5.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/Challenge4/ch4_CP_im5.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent crescent, crescent2;
  //  3 Gaussians
  Themis::model_image_asymmetric_gaussian asg1, asg2, asg3, asg4, asg5;
  //  Combined
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(crescent);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);
  intrinsic_image.add_model_image(asg3);
  intrinsic_image.add_model_image(asg4);
  intrinsic_image.add_model_image(asg5);
  intrinsic_image.add_model_image(crescent2);
  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  crescent2.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  asg3.use_analytical_visibilities();
  asg4.use_analytical_visibilities();
  asg5.use_analytical_visibilities();


  
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
  for (size_t i=0; i<5; ++i)
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

  // Crescent2 params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(0.5);
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




  means[0] = 1.9585735;
  means[1] = 1.2202246e-10;
  means[2] = 0.34763943;
  means[3] = 0.79149591;
  means[4] = -1.4564152;
  means[5] = -3.1809189e-18;
  means[6] = -3.3699943e-18;
  means[7] = 0.82549669;
  means[8] = 9.3940746e-11;
  means[9] = 0.71026369;
  means[10] = 2.0293777;
  means[11] = -1.6266677e-11;
  means[12] = 1.1405847e-10;
  means[13] = 0.2013497;
  means[14] = 2.1859332e-11;
  means[15] = 0.96271647;
  means[16] = 1.6642183;
  means[17] = -6.1969416e-11;
  means[18] = -1.8479018e-10;
  means[19] = 0.12896146;
  means[20] = 3.8598155e-11;
  means[21] = 0.57288564;
  means[22] = 1.689556;
  means[23] = 1.627308e-10;
  means[24] = 1.8684018e-10;
  means[25] = 0.27949084;
  means[26] = 4.647899e-11;
  means[27] = 0.86613154;
  means[28] = 3.1005495;
  means[29] = -1.999447e-10;
  means[30] = -2.0369721e-11;
  means[31] = 0.10571384;
  means[32] = 3.931911e-11;
  means[33] = 0.68285323;
  means[34] = 2.8580728;
  means[35] = 2.0337677e-10;
  means[36] = -1.6669975e-10;

  /*
  // 2nd crescent
  means[37] = 0.5;
  means[38] = 1.0091313e-10;
  means[39] = 0.0060642193;
  means[40] = 0.9899736;
  means[41] = -1.4106404;
  means[42] = 2.759793e-18;
  means[43] = 1.2591185e-18;
  means[44] = 0.75421189;
  */


  // Best fit
  means[0] = 0.62871962;
  means[1] = 1.5433009e-10;
  means[2] = 0.38302348;
  means[3] = 0.010373971;
  means[4] = -1.4705529;
  means[5] = 5.9671563e-19;
  means[6] = -1.4817607e-20;
  means[7] = 0.80028243;
  means[8] = 9.195467e-11;
  means[9] = 0.77232023;
  means[10] = 1.9870209;
  means[11] = -1.2765239e-11;
  means[12] = 1.2767935e-10;
  means[13] = 0.21017114;
  means[14] = 4.3992924e-11;
  means[15] = 0.82164676;
  means[16] = 1.6777327;
  means[17] = -9.6038393e-11;
  means[18] = -1.6997038e-10;
  means[19] = 0.071818891;
  means[20] = 1.5440909e-11;
  means[21] = 0.91833375;
  means[22] = 1.8785698;
  means[23] = 1.7575767e-10;
  means[24] = 1.9590772e-10;
  means[25] = 0.23780301;
  means[26] = 2.70152e-11;
  means[27] = 0.95855799;
  means[28] = 3.0848101;
  means[29] = -2.0402248e-10;
  means[30] = -3.487605e-11;
  means[31] = 0.082343916;
  means[32] = 2.125087e-11;
  means[33] = 0.9189942;
  means[34] = 2.9895152;
  means[35] = 2.0500177e-10;
  means[36] = -1.6449391e-10;
  means[37] = 1.4687403;
  means[38] = 9.7884366e-11;
  means[39] = 0.016871469;
  means[40] = 0.98954042;
  means[41] = -2.8951276;
  means[42] = 3.7738556e-18;
  means[43] = 1.4646348e-18;

  
  for (size_t i=0; i<5; ++i)
  {
    ranges[7+i*6+0] = 1e-5;
    ranges[7+i*6+1] = 1e-4*uas2rad;
    ranges[7+i*6+2] = 1e-5;
    ranges[7+i*6+3] = 1e-5;
    ranges[7+i*6+4] = 1e-4*uas2rad;
    ranges[7+i*6+5] = 1e-4*uas2rad;
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
  int Number_of_chains = 160;
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
  MCMC_obj.set_checkpoint(Ckpt_frequency,"Crescent.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain-Crescent.dat", "Lklhd-Crescent.dat", "Chi2-Crescent.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

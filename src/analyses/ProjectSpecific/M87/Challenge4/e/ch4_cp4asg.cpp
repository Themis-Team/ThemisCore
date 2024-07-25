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
  Themis::model_image_crescent crescent;
  //  3 Gaussians
  Themis::model_image_asymmetric_gaussian asg1, asg2, asg3, asg4;
  //  Combined
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(crescent);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);
  intrinsic_image.add_model_image(asg3);
  intrinsic_image.add_model_image(asg4);
  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  asg3.use_analytical_visibilities();
  asg4.use_analytical_visibilities();


  
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
  for (size_t i=0; i<4; ++i)
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

  means[0] = 1.7077937;
  means[1] = 1.0091313e-10;
  means[2] = 0.0060642193;
  means[3] = 0.9899736;
  means[4] = -1.4106404;
  means[5] = 2.759793e-18;
  means[6] = 1.2591185e-18;
  means[7] = 0.75421189;
  /*
  means[8] = 1.110887e-10;
  means[9] = 0.85134458;
  means[10] = 1.9684624;
  means[11] = 5.8742887e-11;
  means[12] = -1.5689464e-10;
  means[13] = 0.13588267;
  means[14] = 4.953116e-11;
  means[15] = 0.23771009;
  means[16] = 2.4845035;
  means[17] = 2.0275825e-10;
  means[18] = 1.6579622e-10;
  means[19] = 0.89953096;
  means[20] = 9.5475416e-11;
  means[21] = 0.14614355;
  means[22] = 0.38927037;
  means[23] = 1.3490097e-10;
  means[24] = 1.7311458e-11;
  */  

  means[0] = 1.8890426;
  means[1] = 1.0073744e-10;
  means[2] = 0.012398805;
  means[3] = 0.98687545;
  means[4] = -1.4930159;
  means[5] = 3.7299294e-18;
  means[6] = -2.0497586e-18;
  means[7] = 1.1043065;
  means[8] = 2.4234955e-10;
  means[9] = 2.8954841e-05;
  means[10] = 1.1619828;
  means[11] = -1.1646217e-10;
  means[12] = 9.4364154e-11;
  means[13] = 0.21363646;
  means[14] = 1.5279365e-10;
  means[15] = 0.15078911;
  means[16] = 2.6499741;
  means[17] = -1.0430911e-11;
  means[18] = -1.6218054e-10;
  means[19] = 0.16423842;
  means[20] = 2.4230762e-10;
  means[21] = 0.40286845;
  means[22] = 1.3544466;
  means[23] = 1.7474723e-11;
  means[24] = -2.0962802e-10;
  means[25] = 0.14532995;
  means[26] = 2.1262873e-10;
  means[27] = 0.44779395;
  means[28] = 1.6349112;
  means[29] = 6.7101338e-11;
  means[30] = 6.399092e-12;
  
  means[0] = 1.7358613;
  means[1] = 1.0655772e-10;
  means[2] = 0.11270457;
  means[3] = 0.93506983;
  means[4] = -3.1126226;
  means[5] = 1.571268e-18;
  means[6] = 4.1720679e-18;
  means[7] = 0.21335831;
  means[8] = 5.6682245e-11;
  means[9] = 0.41896761;
  means[10] = 2.273095;
  means[11] = 2.3290148e-10;
  means[12] = 1.1735748e-10;
  means[13] = 0.1231538;
  means[14] = 5.0287893e-11;
  means[15] = 0.18653984;
  means[16] = 0.026454087;
  means[17] = 2.1444646e-10;
  means[18] = -2.0631749e-10;
  means[19] = 0.35658099;
  means[20] = 6.1858229e-11;
  means[21] = 0.00068247051;
  means[22] = 2.1239883;
  means[23] = 1.7273345e-10;
  means[24] = -3.3220107e-12;
  means[25] = 1.0778555;
  means[26] = 1.4193729e-10;
  means[27] = 0.29169909;
  means[28] = 2.4430798;
  means[29] = 1.7483493e-11;
  means[30] = -1.0801767e-10;

  // Best fit
  means[0] = 1.6728654;
  means[1] = 1.0055617e-10;
  means[2] = 0.013526265;
  means[3] = 0.8292721;
  means[4] = -2.6665964;
  means[5] = 1.6330612e-18;
  means[6] = 3.4547258e-18;
  means[7] = 0.22115181;
  means[8] = 4.779451e-11;
  means[9] = 0.61223994;
  means[10] = 2.614087;
  means[11] = 2.2887874e-10;
  means[12] = 1.1147174e-10;
  means[13] = 0.12886394;
  means[14] = 3.2440054e-11;
  means[15] = 0.65357505;
  means[16] = 0.00047130199;
  means[17] = 2.1231741e-10;
  means[18] = -2.01724e-10;
  means[19] = 0.32542755;
  means[20] = 5.2304174e-11;
  means[21] = 0.12338907;
  means[22] = 2.2211291;
  means[23] = 1.5386942e-10;
  means[24] = 4.0102622e-12;
  means[25] = 1.1515951;
  means[26] = 1.3073236e-10;
  means[27] = 0.40993131;
  means[28] = 2.6578143;
  means[29] = -2.7571276e-11;
  means[30] = -7.634345e-11;

  
  for (size_t i=0; i<4; ++i)
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

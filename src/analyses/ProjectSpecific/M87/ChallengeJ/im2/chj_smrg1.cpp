//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_ring.h"
//#include "model_image_crescent.h"
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
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/ChallengeJ/VM_challenge_2_seed_2_simobs_netcal_scanavg.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/ChallengeJ/CP_challenge_2_seed_2_simobs_netcal_scanavg.d"));
  
  // Choose the model to compare
  //  1 Ring
  Themis::model_image_ring ring;
  Themis::model_image_smooth smrg(ring);
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(smrg);
  
  // Use analytical Visibilities
  //ring model alwasy use analytical vis
  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // ring params
  
   // Total Flux V00
   P.push_back(new Themis::prior_linear(0.0,2.0));
   means.push_back(0.50);
   ranges.push_back(0.01);
  //   Outer size Rext
  P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
  means.push_back(50*uas2rad);
  ranges.push_back(10*uas2rad);
  //   psi :  inner size Rin=Rext*(1-psi)
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.2);
  ranges.push_back(0.01);
  // Postion angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(1e-4); 

  //smooth_image par1
       P.push_back(new Themis::prior_linear(0,40*uas2rad));
         means.push_back(0);
          ranges.push_back(2.*uas2rad);
 // smooth_image par2
        P.push_back(new Themis::prior_linear(0,0.99));
              means.push_back(0.1);
                ranges.push_back(1e-2);
  
  // smooth rotation
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0.2);
  ranges.push_back(0.1); 
  
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0);
  ranges.push_back(1e-7*uas2rad);
  
//best fit
  means[0] = 1.1212157;
    means[1] = 1.4384737e-10;
      means[2] = 0.55692383;
        means[3] = -1.4396072;
	  means[4] = 2.6477075e-11;
	    means[5] = 0.35802767;
	      means[6] = 2.5054035;
	        means[7] = -3.1732338e-18;
		  means[8] = -1.9387003e-18;



  std::cerr << "here\n";


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
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 60;
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
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC_amrg1.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain_smrg1.dat", "Lklhd_smrg1.dat", "Chi2_amrg1.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

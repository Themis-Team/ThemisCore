/*! 
  \file validation/riaf_va_cp.cpp
  \authors Avery E. Broderick, Roman Gold, Mansour Karami, Jorge A. Preciado
  \date June, 2017
  
  \brief Validation test for the RIAF model with visibility amplitude and closure pahase data ...

  \details
  Validation test for the RIAF model including visibility amplidute and closure phase data.
  This example uses the SED-fitted RIAF model with  scattered image and finds a model
  fit to EHT visibility amplitude and closure phase data using a sampler. 
  
  \image html plots/validation/riaf_va_cp-Triangle.png "Triangle plot showing the likely parameter values and associated confidence contours."

  \image html plots/validation/riaf_va_cp-Trace.png "Trace plot showing the fluctuations in the parameters for each MCMC chain as a function of MCMC step."

  \image html plots/validation/riaf_va_cp-Likelihood.png "Log-likelihoods of the individual chains as a function of MCMC step."

  \details 
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_sed_fitted_riaf.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude VM_d2007(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  VM_d2007.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  Themis::data_visibility_amplitude VM_d2009_095(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  Themis::data_visibility_amplitude VM_d2009_096(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  Themis::data_visibility_amplitude VM_d2009_097(Themis::utils::global_path("eht_data/VM_2009_097.d"));


  // Read in closure phases data from
  //2009
  Themis::data_closure_phase CP_d2009_093(Themis::utils::global_path("eht_data/CP_2009_093.d"));
  Themis::data_closure_phase CP_d2009_096(Themis::utils::global_path("eht_data/CP_2009_096.d"));
  Themis::data_closure_phase CP_d2009_097(Themis::utils::global_path("eht_data/CP_2009_097.d"));
  //2011
  Themis::data_closure_phase CP_d2011_088(Themis::utils::global_path("eht_data/CP_2011_088.d"));
  Themis::data_closure_phase CP_d2011_090(Themis::utils::global_path("eht_data/CP_2011_090.d"));
  Themis::data_closure_phase CP_d2011_091(Themis::utils::global_path("eht_data/CP_2011_091.d"));
  Themis::data_closure_phase CP_d2011_094(Themis::utils::global_path("eht_data/CP_2011_094.d"));
  //2012
  Themis::data_closure_phase CP_d2012_081(Themis::utils::global_path("eht_data/CP_2012_081.d"));
  //2013
  Themis::data_closure_phase CP_d2013_080(Themis::utils::global_path("eht_data/CP_2013_080.d"));
  Themis::data_closure_phase CP_d2013_081(Themis::utils::global_path("eht_data/CP_2013_081.d"));
  Themis::data_closure_phase CP_d2013_082(Themis::utils::global_path("eht_data/CP_2013_082.d"));
  Themis::data_closure_phase CP_d2013_085(Themis::utils::global_path("eht_data/CP_2013_085.d"));
  Themis::data_closure_phase CP_d2013_086(Themis::utils::global_path("eht_data/CP_2013_086.d"));


  // Choose the model to compare
  Themis::model_image_sed_fitted_riaf intrinsic_image(Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"));
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  //Use small size images for testing purposes
  //intrinsic_image.use_small_images();
  intrinsic_image.set_image_resolution(64);

  //Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,0.998)); // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1)); // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  std::vector<double> means, ranges;
  means.push_back(0.10);
  means.push_back(0.50);
  means.push_back(-156.0/180.0*M_PI);
  ranges.push_back(0.05);
  ranges.push_back(0.05);
  ranges.push_back(0.05);

  // vector to hold the name of variables, if the names are provided it would be added
  // as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("Spin");
  var_names.push_back("Cos(Inclination)");
  var_names.push_back("Position_Angle");

  //Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());

  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  //Visibility Amplitudes
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_097,image));

  //Closure Phases
  double sigma_phi = 3.86;
  //2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_093,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_096,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_097,intrinsic_image,sigma_phi));
  //2011
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_088,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_090,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_091,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_094,intrinsic_image,sigma_phi));
  //2012
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2012_081,intrinsic_image,sigma_phi));
  //2013
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_080,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_081,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_082,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_085,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_086,intrinsic_image,sigma_phi));

  //Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  //Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);


  // Create a sampler object, here the PTMCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 8;
  int Number_of_temperatures = 4;
  int Number_of_processors_per_lklhd = 16;
  int Number_of_steps = 100000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 0;

  MC_obj.set_checkpoint(10,"RIAFVACP.ckpt");
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
  MC_obj.run_sampler(L_obj, 
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-RIAFVACP.dat", "Lklhd-RIAFVACP.dat", 
		     "Chi2-RIAFVACP.dat", means, ranges, var_names, false, verbosity);

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

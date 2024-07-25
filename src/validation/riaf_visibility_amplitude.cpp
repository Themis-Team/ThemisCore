/*!
  \file validation/riaf_visibility_amplitude.cpp
  \authors Avery E. Broderick, Mansour Karami, Roman Gold, Jorge A. Preciado
  \date  April, 2017
  
  \internal
  \validation Validation test using the RIAF model with visibility amplitude data.
  \endinternal
  
  \brief Test main file for the RIAF model with visibility amplitude data.
  
  \details Test for the RIAF model including visibility amplidute data only.
  This example shows how to query the SED-fitted RIAF model
  (an explicit model_image class), scatter the image and find a model
  fit to EHT visibility amplitude data using a sampler. 
  
  \image html plots/validation/RIAF-VM-Triangle.png "Triangle plot showing the likely parameter values and associated confidence contours."
  
  \image html plots/validation/RIAF-VM-Trace.png "Trace plot showing the fluctuations in the parameters for each MCMC chain as a function of MCMC step."
  
  \image html plots/validation/RIAF-VM-Likelihood.png "Log-likelihoods of the individual chains as a function of MCMC step."
  
*/

#include "data_visibility_amplitude.h"
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
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude d2007(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  d2007.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  Themis::data_visibility_amplitude d2009_095(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  Themis::data_visibility_amplitude d2009_096(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  Themis::data_visibility_amplitude d2009_097(Themis::utils::global_path("eht_data/VM_2009_097.d"));


  // Choose the model to compare
  Themis::model_image_sed_fitted_riaf intrinsic_image(Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"));
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  //Use small size images for testing purposes
  //intrinsic_image.use_small_images();
  if(world_rank < world_size/2)
    intrinsic_image.set_image_resolution(40);
  else
    intrinsic_image.set_image_resolution(32);

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,0.998)); // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1)); // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  std::vector<double> means, ranges;
  means.push_back(0.10);
  means.push_back(0.5);
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

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);


  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 8;
  int Number_of_temperatures = 2;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 100;
  int Temperature_stride = 10;
  int Chi2_stride = 20;
  int verbosity = 0;


  MC_obj.set_checkpoint(1000,"RIAFVA.ckpt");
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
  MC_obj.run_sampler(L_obj,
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-RIAFVA.dat", "Lklhd-RIAFVA.dat", 
		     "Chi2-RIAFVA.dat", means, ranges, var_names, false, verbosity);

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

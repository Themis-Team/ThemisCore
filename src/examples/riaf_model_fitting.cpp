/*! 
  \file examples/riaf_model_fitting.cpp
  \author Avery E. Broderick
  \date June, 2017
  \brief Example illustrating the use of the RIAF model.
  
  \details Themis allows a vast variety of models to be compared to
  EHT data. This example shows how to query the SED-fitted RIAF model
  (an explicit model_image class), scatter the image and find a model
  fit to EHT visibility amplitude and closure phase data using the
  sampler. Reading in the data follows closely the reading_data.cpp
  example.
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_sed_fitted_riaf.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude VM_d2007("../../eht_data/VM_2007_100.d");
  VM_d2007.add_data("../../eht_data/VM_2007_101.d");
  Themis::data_visibility_amplitude VM_d2009_095("../../eht_data/VM_2009_095.d");
  Themis::data_visibility_amplitude VM_d2009_096("../../eht_data/VM_2009_096.d");
  Themis::data_visibility_amplitude VM_d2009_097("../../eht_data/VM_2009_097.d");

  // Read in closure phases data from
  // 2009
  Themis::data_closure_phase CP_d2009_093("../../eht_data/CP_2009_093.d");
  Themis::data_closure_phase CP_d2009_096("../../eht_data/CP_2009_096.d");
  Themis::data_closure_phase CP_d2009_097("../../eht_data/CP_2009_097.d");
  // 2011
  Themis::data_closure_phase CP_d2011_088("../../eht_data/CP_2011_088.d");
  Themis::data_closure_phase CP_d2011_090("../../eht_data/CP_2011_090.d");
  Themis::data_closure_phase CP_d2011_091("../../eht_data/CP_2011_091.d");
  Themis::data_closure_phase CP_d2011_094("../../eht_data/CP_2011_094.d");
  // 2012
  Themis::data_closure_phase CP_d2012_081("../../eht_data/CP_2012_081.d");
  // 2013
  Themis::data_closure_phase CP_d2013_080("../../eht_data/CP_2013_080.d");
  Themis::data_closure_phase CP_d2013_081("../../eht_data/CP_2013_081.d");
  Themis::data_closure_phase CP_d2013_082("../../eht_data/CP_2013_082.d");
  Themis::data_closure_phase CP_d2013_085("../../eht_data/CP_2013_085.d");
  Themis::data_closure_phase CP_d2013_086("../../eht_data/CP_2013_086.d");


  // Choose the model to compare
  Themis::model_image_sed_fitted_riaf intrinsic_image("../../src/VRT2/DataFiles/2010_combined_fit_parameters.d");

  // Scatter the intrinsic model image 
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  // Use small images: speeds up the computation substantially, 
  // NOT recommended for production runs!
  intrinsic_image.use_small_images();


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,0.998));  // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1));       // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  // Define starting values and ranges for model parameters
  std::vector<double> means, ranges;
  means.push_back(0.10);              // Spin amplitude
  means.push_back(0.50);              // cos(inclination)
  means.push_back(-156.0/180.0*M_PI); // position angle
  ranges.push_back(0.05);             // Spin amplitude
  ranges.push_back(0.05);             // cos(inclination)
  ranges.push_back(0.05);             // position angle

  // vector to hold the name of model parameters, if provided they
  // will be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("Spin");
  var_names.push_back("Cos(Inclination)");
  var_names.push_back("Position_Angle");


  // Optional: Set the variable transformations. Here it is set to not
  // do anything which also is the default. The code below serves as
  // demonstration as to how to control possible transformations
  // performed on the parameter space prior to model fitting
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  // Visibility Amplitudes Likelihoods
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_097,image));

  // Closure Phases Likelihoods
  double sigma_phi = 3.86;
  // 2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_093,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_096,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_097,intrinsic_image,sigma_phi));
  // 2011
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_088,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_090,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_091,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_094,intrinsic_image,sigma_phi));
  // 2012
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2012_081,intrinsic_image,sigma_phi));
  // 2013
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_080,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_081,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_082,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_085,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_086,intrinsic_image,sigma_phi));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);


  // Create a sampler object, here the PTMCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 14;
  int Number_of_temperatures = 4;
  int Number_of_processors_per_lklhd = 9;
  int Number_of_steps = 100000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  // Write checkpoint file to restart from every 10 steps
  MC_obj.set_checkpoint(10,"RIAFVACP.ckpt");

  // Parallelize
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Get down to business
  MC_obj.run_sampler(L_obj, 
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-RIAFVACP.dat", "Lklhd-RIAFVACP.dat", 
                      "Chi2-RIAFVACP.dat", means, ranges, var_names, false);

  // Finalize MPI
  MPI_Finalize();

  return 0;
}


/*!
  \file examples/riaf_model_fitting.cpp
  \details
  
  \code


#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_sed_fitted_riaf.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude VM_d2007("../../eht_data/VM_2007_100.d");
  VM_d2007.add_data("../../eht_data/VM_2007_101.d");
  Themis::data_visibility_amplitude VM_d2009_095("../../eht_data/VM_2009_095.d");
  Themis::data_visibility_amplitude VM_d2009_096("../../eht_data/VM_2009_096.d");
  Themis::data_visibility_amplitude VM_d2009_097("../../eht_data/VM_2009_097.d");

  // Read in closure phases data from
  // 2009
  Themis::data_closure_phase CP_d2009_093("../../eht_data/CP_2009_093.d");
  Themis::data_closure_phase CP_d2009_096("../../eht_data/CP_2009_096.d");
  Themis::data_closure_phase CP_d2009_097("../../eht_data/CP_2009_097.d");
  // 2011
  Themis::data_closure_phase CP_d2011_088("../../eht_data/CP_2011_088.d");
  Themis::data_closure_phase CP_d2011_090("../../eht_data/CP_2011_090.d");
  Themis::data_closure_phase CP_d2011_091("../../eht_data/CP_2011_091.d");
  Themis::data_closure_phase CP_d2011_094("../../eht_data/CP_2011_094.d");
  // 2012
  Themis::data_closure_phase CP_d2012_081("../../eht_data/CP_2012_081.d");
  // 2013
  Themis::data_closure_phase CP_d2013_080("../../eht_data/CP_2013_080.d");
  Themis::data_closure_phase CP_d2013_081("../../eht_data/CP_2013_081.d");
  Themis::data_closure_phase CP_d2013_082("../../eht_data/CP_2013_082.d");
  Themis::data_closure_phase CP_d2013_085("../../eht_data/CP_2013_085.d");
  Themis::data_closure_phase CP_d2013_086("../../eht_data/CP_2013_086.d");


  // Choose the model to compare
  Themis::model_image_sed_fitted_riaf intrinsic_image("../../src/VRT2/DataFiles/2010_combined_fit_parameters.d");

  // Scatter the intrinsic model image 
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  // Use small images: speeds up the computation substantially, 
  // NOT recommended for production runs!
  intrinsic_image.use_small_images();


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,0.998));  // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1));       // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  // Define starting values and ranges for model parameters
  std::vector<double> means, ranges;
  means.push_back(0.10);              // Spin amplitude
  means.push_back(0.50);              // cos(inclination)
  means.push_back(-156.0/180.0*M_PI); // position angle
  ranges.push_back(0.05);             // Spin amplitude
  ranges.push_back(0.05);             // cos(inclination)
  ranges.push_back(0.05);             // position angle

  // vector to hold the name of model parameters, if provided they
  // will be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("Spin");
  var_names.push_back("Cos(Inclination)");
  var_names.push_back("Position_Angle");


  // Optional: Set the variable transformations. Here it is set to not
  // do anything which also is the default. The code below serves as
  // demonstration as to how to control possible transformations
  // performed on the parameter space prior to model fitting
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  // Visibility Amplitudes Likelihoods
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_097,image));

  // Closure Phases Likelihoods
  double sigma_phi = 3.86;
  // 2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_093,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_096,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_097,intrinsic_image,sigma_phi));
  // 2011
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_088,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_090,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_091,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_094,intrinsic_image,sigma_phi));
  // 2012
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2012_081,intrinsic_image,sigma_phi));
  // 2013
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_080,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_081,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_082,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_085,intrinsic_image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_086,intrinsic_image,sigma_phi));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);


  // Create a sampler object, here the PTMCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 14;
  int Number_of_temperatures = 4;
  int Number_of_processors_per_lklhd = 9;
  int Number_of_steps = 100000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  // Write checkpoint file to restart from every 10 steps
  MC_obj.set_checkpoint(10,"RIAFVACP.ckpt");

  // Parallelize
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Get down to business
  MC_obj.run_sampler(L_obj, 
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-RIAFVACP.dat", "Lklhd-RIAFVACP.dat", 
                      "Chi2-RIAFVACP.dat", means, ranges, var_names, false);

  // Finalize MPI
  MPI_Finalize();

  return 0;
}

  \endcode
*/



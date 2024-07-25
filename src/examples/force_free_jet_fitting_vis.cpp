/*! 
  \file sheasring_spot_model_fitting.cpp
  \author Paul Tiede
  \date March, 2018
  \brief Provides ability to fit shearing spot model given some data using themis.
  
  \details This shows how to query the shearing spot model (model_shearing_spot), 
	scatter the image and find a model fit to EHT visibility amplitude 
	and closure phase data using the sampler. Note scattering isn't finished yet. 
	Reading in the data follows closely the reading_data.cpp
  example.
*/

#include "data_visibility_amplitude.h"
#include "model_image_sed_fitted_force_free_jet.h"
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


	//Read in visibility amplitude data from 2009 and 2012
  Themis::data_visibility_amplitude VM_d2009_095("/home/ptiede/Themis/eht_data/VM_2009_095_M87.d");
  Themis::data_visibility_amplitude VM_d2009_096("/home/ptiede/Themis/eht_data/VM_2009_096_M87.d");
  Themis::data_visibility_amplitude VM_d2009_097("/home/ptiede/Themis/eht_data/VM_2009_097_M87.d");
  Themis::data_visibility_amplitude VM_d2012_081("/home/ptiede/Themis/eht_data/VM_2012_081_M87.d");



  Themis::model_image_sed_fitted_force_free_jet model("/home/ptiede/Themis/src/VRT2/DataFiles/IRO_high_mass.d");
  //Set image resolution
  //model.set_image_resolution(120,120);

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0,10)); //Total intensity
  P.push_back(new Themis::prior_linear(0.0,0.998));  // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1));       // cos(inclination)
  P.push_back(new Themis::prior_linear(0,2)); //jet radial power law
  P.push_back(new Themis::prior_linear(1,20)); //jet opening angle in degrees
  P.push_back(new Themis::prior_linear(2, 80)); //Jet loading radius
  P.push_back(new Themis::prior_linear(2,10)); //Jet asymptotic lorentz factor
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); //Position angle

  // Define starting values and ranges for model parameters
  std::vector<double> means, ranges;
  means.push_back(1.0);                 //total intensity
  means.push_back(0.9);                 // Spin parameter (0-1)
  means.push_back(std::cos(25*M_PI/180));  // Cos(Inclination)
  means.push_back(2.0/3); 					 //Jet radial power law
	means.push_back(10);                 //jet opening angle
	means.push_back(40); 							 //Loading radius of particles
  means.push_back(5);
  means.push_back(0);                //position angle

  ranges.push_back(0.3);                 //intensity norm
  ranges.push_back(0.1);                 // Spin parameter (0-1)
  ranges.push_back(0.1);  // Cos(Inclination)
  ranges.push_back(1e-2); 					 //Jet radial power law
	ranges.push_back(1e-2);                 //jet opening angle
	ranges.push_back(10); 							 //Loading radius of particles
	ranges.push_back(1e-2); 							   //Jet asympotitic lorentz factor
  ranges.push_back(0.2);

  // vector to hold the name of model parameters, if provided they
  // will be added as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("I0");
  var_names.push_back("Spin");
  var_names.push_back("Cos(Inclination)");
  var_names.push_back("Jet_p");
	var_names.push_back("OpenAng");
	var_names.push_back("rLoad");
	var_names.push_back("GammaInf");
  var_names.push_back("PosAng");

  //Set the transforms
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());//I0
  T.push_back(new Themis::transform_none());//spin
  T.push_back(new Themis::transform_fixed(means[2]));//cos incl
  T.push_back(new Themis::transform_fixed(means[3]));//rad power (jet p)
  T.push_back(new Themis::transform_fixed(means[4]));//open ang
  T.push_back(new Themis::transform_none()); //rload
  T.push_back(new Themis::transform_fixed(means[6]));//asymptotic lorentz factor
  T.push_back(new Themis::transform_none()); //pos angle


 // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;


	// Closure Phases Likelihoods
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_095, model));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_096, model));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_097, model));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2012_081, model));

 

  
	// Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);

  std::cout << "Starting chi square on rank " << world_rank << " : " << L_obj.chi_squared(means) << std::endl;

  // Create a sampler object, here the PTMCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_steps = 100000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  // Write checkpoint file to restart from every 10 steps
  MC_obj.set_checkpoint(10,"force_free_VACP.ckpt");

  // Parallelize
  int Number_of_walkers = 24;
  int Number_of_temperatures =6;
  int Number_of_processors_per_lklhd = 8;
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_walkers, Number_of_processors_per_lklhd);


  // Get down to business
  MC_obj.run_sampler(L_obj, 
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-ff_VACP.dat", "Lklhd-ff_VACP.dat", 
                      "Chi2-ff_VACP.dat", means, ranges, var_names, false);


  // Finalize MPI
  MPI_Finalize();

  return 0;
}


/*!
  \file examples/shearing_spot_model_fitting.cpp
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

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);

	//STUFF

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



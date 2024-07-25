/*! 
  \file examples/eht_2018_example.cpp
  \author Paul Tiede
  \date Nov, 2018
  \brief Example presented to the EHT collaboration in 2018 that highlights how to use different models.
  
  \details Themis allows a vast variety of models to be compared to
  EHT data. This example shows how to query and fit a few models to 
  proto-eht visibility amplitude data. The models we consider are symmetric Gaussian, cresecent
  model, SED-fitted riaf model from Broderick 2016.
*/

#include "data_visibility_amplitude.h"
//#include "model_symmetric_gaussian.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "model_image_crescent.h"
//#include "model_image_sed_fitted_riaf.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
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
  Themis::data_visibility_amplitude d2007("../../eht_data/VM_2007_100.d");
  d2007.add_data("../../eht_data/VM_2007_101.d");
  Themis::data_visibility_amplitude d2009_095("../../eht_data/VM_2009_095.d");
  Themis::data_visibility_amplitude d2009_096("../../eht_data/VM_2009_096.d");
  Themis::data_visibility_amplitude d2009_097("../../eht_data/VM_2009_097.d");


  // Choose the model we want to use!
  //Themis::model_symmetric_gaussian intrinsic_image;
  
  Themis::model_image_crescent intrinsic_image; 
  intrinsic_image.use_analytical_visibilities();
  //Themis::model_image_sed_fitted_riaf intrinsic_image("../../src/VRT2/DataFiles/2010_combined_fit_parameters.d");
  //Blur the image
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  /*
  //Priors for crescent (Kamruddin and Dexter 2012)
  P.push_back(new Themis::prior_linear(0.9,1.1)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,image_scale*5)); // x-dir size
  P.push_back(new Themis::prior_linear(0.0,1.0)); // x-dir size
  P.push_back(new Themis::prior_linear(0.0,1.0)); // x-dir size
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // x-dir size
  */
 
  /*
  //Priors for RIAF (Broderick 2016)
  P.push_back(new Themis::prior_linear(0.0,0.998));  // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1));       // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
  */


  
  // Container of base prior class pointers
  double image_scale =  43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;
  //P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
  
  //P.push_back(new Themis::prior_linear(0.0,image_scale*5)); // x-dir size
  
  //Priors for the crescent
  P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,image_scale*5)); // x-dir size
  P.push_back(new Themis::prior_linear(0.0,1.0)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,1.0)); // x-dir size
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // Itotal
   
  std::vector<double> means, ranges;
  
  /*
  means.push_back(1.0); //Itot
  means.push_back(0.5*image_scale); //sigma
  ranges.push_back(10);
  ranges.push_back(0.01*image_scale);
  */
  
  //means and std dev for crescent
  means.push_back(1.0);
  means.push_back(0.5*image_scale);
  means.push_back(0.1);
  means.push_back(0.1);
  means.push_back(0.0);
  ranges.push_back(1.0e-6);
  ranges.push_back(0.01*image_scale);
  ranges.push_back(1e-2);
  ranges.push_back(1e-2);
  ranges.push_back(1e-2*2*M_PI);
  
 

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  //var_names.push_back("$I_{norm}$");
  //var_names.push_back("$\\sigma$");

  
  //Var names for crescent
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma$");
  var_names.push_back("$\\psi$");
  var_names.push_back("$\\tau$");
  var_names.push_back("$\\xi$");
  

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);


  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);
  
  
  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 2;
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 5000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 
                              Number_of_processors_per_lklhd);
                              
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                     "Chain-Crescent.dat", "Lklhd-Crescent.dat", "Chi2-Crescent.dat", 
                     means, ranges, var_names, false);


  //Finalize MPI
  MPI_Finalize();
  return 0;
}

/*!
  \file tests/asymmetric_gaussian_image_comparison.cpp
  \author Avery Broderick
  \date June 2017
  \test Gaussian model with visibility amplitude data
  \brief Fit an asymmetric Gaussian model to visibility amplitude data
  
  \details This tests estimates parameters for the asymmetric Gaussian model.
  See reading_data.cpp example for details on reading in EHT data.
*/

#include "data_visibility_amplitude.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_ensemble_averaged_scattered_image.h"
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


  // Choose the model to compare
  // Themis::model_image_gaussian image;
  Themis::model_image_asymmetric_gaussian intrinsic_image;
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);
  
  // Container of base prior class pointers
  double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.9,1.1)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,image_scale)); // size
  P.push_back(new Themis::prior_linear(0.1,0.9)); // asymmetry parameters
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  std::vector<double> means, ranges;
  means.push_back(1.0);
  means.push_back(0.5*image_scale);
  means.push_back(0.5);
  means.push_back(0.25*M_PI);
  ranges.push_back(1.0e-6);
  ranges.push_back(0.01*image_scale);
  ranges.push_back(0.01);
  ranges.push_back(0.01*M_PI);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("Intensity");
  var_names.push_back("$\\sigma$");
  var_names.push_back("A");
  var_names.push_back("Position_Angle");
  
  
  // Applying the coordinate transformation on the initial values
  Themis::transform_none Trans;
  for(unsigned int i = 0 ; i < means.size(); i++)
    {
      means[i] = Trans.forward(means[i]);
      ranges[i] = Trans.forward(ranges[i]);
    } 

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
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
  int Number_of_chains = 128;
  int Number_of_temperatures = 5;
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 5000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 
                              Number_of_processors_per_lklhd);
                              
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-AsymmetricGaussian-5T.dat", "Lklhd-AsymmetricGaussian-5T.dat", 
		     "Chi2-AsymmetricGaussian-5T.dat", means, ranges, var_names, false);


  //Finalize MPI
  MPI_Finalize();
  return 0;
}

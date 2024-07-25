/*!
  \file examples/scattered_gaussian_fitting.cpp
  \author
  \date Apr 2017
  \brief Example of fitting a Gaussian image model to visibility amplitude data.

  \details This example illustrates how to generate a Gaussian image
  model, include scattering to the intrinsic model image, read-in
  visibility amplitude data (as also shown in reading_data.cpp), and
  fitting the model to the eht data. The model can take full advantage
  of the analytically known visibility amplitudes thus making this
  test really fast.
*/

#include "data_visibility_amplitude.h"
#include "model_image_gaussian.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>
#include <string>

int main(int argc, char* argv[])
{
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
  Themis::model_image_gaussian intrinsic_image;
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

  // Container of base prior class pointers
  double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.9999,1.0001)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,image_scale)); // x-dir size
  P.push_back(new Themis::prior_linear(0.0,image_scale)); // y-dir size
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  std::vector<double> means, ranges;
  means.push_back(1.0);
  means.push_back(0.5*image_scale);
  means.push_back(0.5*image_scale);
  means.push_back(0.25*M_PI);
  ranges.push_back(1.0e-6);
  ranges.push_back(image_scale);
  ranges.push_back(image_scale);
  ranges.push_back(0.5*M_PI);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;

  // Applying the coordinate transformation on the initial values
  // RG:REMOVE
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
  int Number_of_chains = 128;           // Number of walkers
  int Number_of_temperatures = 5;       
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 3000;           // Total number of MCMC steps before quitting
  int Temperature_stride = 50;          // Communicate among walkers lessons learned about the sampled parameter space every 50 MCMC steps
  int Chi2_stride = 20;                 // Output chi^2 file every 20 MCMC steps

  // Parallelization settings
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Sample the parameter space
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                    "Chain-Gaussian.dat", "Lklhd-Gaussian.dat", "Chi2-Gaussian.dat", 
                    means, ranges, var_names, false);
  

  //Finalize MPI
  MPI_Finalize();
  return 0;
}



/*! 
  \file examples/scattered_gaussian_fitting.cpp
  \details 
  \code
  #include "data_visibility_amplitude.h"
  #include "model_image_gaussian.h"
  #include "model_ensemble_averaged_scattered_image.h"
  #include "likelihood.h"
  #include "sampler_affine_invariant_tempered_MCMC.h"
  #include <mpi.h>
  #include <memory> 
  #include <string>
  #include <string>

  int main(int argc, char* argv[])
  {
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
    Themis::model_image_gaussian intrinsic_image;
    Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

    // Container of base prior class pointers
    double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
    std::vector<Themis::prior_base*> P;
    P.push_back(new Themis::prior_linear(0.9999,1.0001)); // Itotal
    P.push_back(new Themis::prior_linear(0.0,image_scale)); // x-dir size
    P.push_back(new Themis::prior_linear(0.0,image_scale)); // y-dir size
    P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

    std::vector<double> means, ranges;
    means.push_back(1.0);
    means.push_back(0.5*image_scale);
    means.push_back(0.5*image_scale);
    means.push_back(0.25*M_PI);
    ranges.push_back(1.0e-6);
    ranges.push_back(image_scale);
    ranges.push_back(image_scale);
    ranges.push_back(0.5*M_PI);

    // vector to hold the name of variables, if the names are provided it would be added 
    // as the header to the chain file 
    std::vector<std::string> var_names;

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
    int Number_of_steps = 3000;
    int Temperature_stride = 50;
    int Chi2_stride = 20;

    MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
    MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-Gaussian.dat", "Lklhd-Gaussian.dat", "Chi2-Gaussian.dat", 
                      means, ranges, var_names, false);
    

    //Finalize MPI
    MPI_Finalize();
    return 0;
  }

  \endcode
*/

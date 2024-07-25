/*!
  \file tests/closure_amplitude_test.cpp
  \author Avery Broderick
  \date June 2018
  \test Closure amplitude test
  \brief Fits a symmetric Gaussian model to simulated closure amplitudes.
  \details Fits a symmetric Gaussian model to simulated closure amplitudes.
*/

#include "data_visibility_amplitude.h"
#include "model_symmetric_gaussian.h"
#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "likelihood_closure_amplitude.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <memory> 
#include <string>

#include <iostream>
#include <iomanip>

#include <mpi.h>


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in visibility amplitude test data
  Themis::data_closure_amplitude testdata(Themis::utils::global_path("sim_data/closure_amplitude_gaussian_test_data.d"));
  
  // Choose the model to compare
  Themis::model_symmetric_gaussian image;


  // Set parameters
  std::vector<double> params(2);
  params[0] = 2.5;
  params[1] = 5.0e-6/3600.*M_PI/180.;
  image.generate_model(params);
  if (world_rank==0)
  {
    for (size_t i=0; i<testdata.size(); ++i)
      std::cout << "CAData:"
		<< std::setw(15) << testdata.datum(i).u1
		<< std::setw(15) << testdata.datum(i).v1
		<< std::setw(15) << testdata.datum(i).u2
		<< std::setw(15) << testdata.datum(i).v2
		<< std::setw(15) << testdata.datum(i).u3
		<< std::setw(15) << testdata.datum(i).v3
		<< std::setw(15) << testdata.datum(i).u4
		<< std::setw(15) << testdata.datum(i).v4
		<< std::setw(15) << testdata.datum(i).CA
		<< std::setw(15) << testdata.datum(i).err
		<< std::setw(15) << image.closure_amplitude(testdata.datum(i),0.0)
		<< std::endl;
  }


  // Container of base prior class pointers
  // and prior means and ranges
  double image_size = 10. * 1.e-6 /3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  P.push_back(new Themis::prior_linear(0,3)); // Itotal
  means.push_back(params[0]);
  ranges.push_back(1.0e-2);
  
  P.push_back(new Themis::prior_linear(0.01*image_size,10.0*image_size)); // Overall size R
  means.push_back(params[1]);
  ranges.push_back(0.01*image_size);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma$");
	
  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_closure_amplitude(testdata,image));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 5;
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 1000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 0;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 
                              Number_of_processors_per_lklhd);
                              
  MC_obj.run_sampler(L_obj,
		     Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-CAT.dat", "Lklhd-CAT.dat", 
		     "Chi2-CAT.dat", means, ranges, var_names, false, 8, verbosity);


  //Finalize MPI
  MPI_Finalize();
  return 0;
}

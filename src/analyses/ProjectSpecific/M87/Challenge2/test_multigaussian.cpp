/*!
  \file analyses/Challenge2/test_multigaussian.cpp
  \author Avery E. Broderick
  \date Mar 2018
  \brief Example of fitting a multi-Gaussian image model to test visibility amplitude and closure phase data.

  \details Fits a multi-Gaussian model, with an arbitrary number of Gaussians, to a set of simulated visibility amplitude and closure phase data, in preparation for modeling challenge 2.  Note that this can be compiled to generate an MCMC chain or, having found a good fit, compute the Bayesian evidence.
*/

#include "data_visibility_amplitude.h"
#include "model_image_multigaussian.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include "random_number_generator.h"

/*! \cond */
#include <mpi.h>
#include <memory> 
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
/*! \endcond */

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in test data
  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("bin/analyses/Challenge2/VM_Ngauss_data.d"));
  Themis::data_closure_phase dCP(Themis::utils::global_path("bin/analyses/Challenge2/CP_Ngauss_data.d"));

  std::cout << "Printing data:" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank==0)
    for (size_t j=0; j<dVM.size(); ++j) 
      std::cout << "VMdata:"
		<< std::setw(15) << dVM.datum(j).u
		<< std::setw(15) << dVM.datum(j).v
		<< std::setw(15) << dVM.datum(j).V
		<< std::setw(15) << dVM.datum(j).err
		<< std::endl;
  if (world_rank==0)
    for (size_t j=0; j<dCP.size(); ++j) 
      std::cout << "CPdata:"
		<< std::setw(15) << dCP.datum(j).u1
		<< std::setw(15) << dCP.datum(j).v1
		<< std::setw(15) << dCP.datum(j).u2
		<< std::setw(15) << dCP.datum(j).v2
		<< std::setw(15) << dCP.datum(j).CP
		<< std::setw(15) << dCP.datum(j).err
		<< std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Finished printing data:" << std::endl;

  // Create model
  size_t NG=1;
  if (argc>1) 
    NG = std::atoi(argv[1]);

  if (world_rank==0)
    std::cerr << "Assuming " << NG << " Gaussian components\n";

  Themis::model_image_multigaussian image(NG);
  
  // Container of base prior class pointers
  double image_scale = 4e2*1e-6 / 3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  // First gaussian
  P.push_back(new Themis::prior_linear(0,10)); // I 
  P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // x
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // y
  // Remainder of gaussians
  for (size_t j=1; j<NG; ++j)
  {
    P.push_back(new Themis::prior_linear(0,10)); // I 
    P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // x
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // y
  }
  P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // position angle
  
  
  // Prior means and ranges
  std::vector<double> means, ranges;
  means.push_back(1);
  means.push_back(1e-1*image_scale);
  means.push_back(0.0);
  means.push_back(0.0);
  ranges.push_back(1e-4);
  ranges.push_back(1e-4*image_scale);
  ranges.push_back(1e-7*image_scale);
  ranges.push_back(1e-7*image_scale);
  for (size_t j=1; j<NG; ++j) 
  {
    means.push_back(0.1);
    means.push_back(1e-1*image_scale);
    means.push_back(0.0);
    means.push_back(0.0);
    ranges.push_back(0.01);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
  }
  means.push_back(0.0);
  ranges.push_back(1e-7);



  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_visibility_amplitude(dVM,image));
  L.push_back(new Themis::likelihood_closure_phase(dCP,image));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);


#if 1 // Generate a chain for finding best fits

  // Generate a chain
  int Number_of_chains = 256;
  int Number_of_temperatures = 8;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 1000000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  // Set the CPU distribution
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Run the Sampler                            
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-NGtest.dat", "Lklhd-NGtest.dat", 
		     "Chi2-NGtest.dat", means, ranges, var_names, false);


#else // Compute evidence

  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 128;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 10000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 1;
  int burn_in = 1000;
  std::vector<double> temperatures(Number_of_temperatures);
  std::vector<std::string> likelihood_files(Number_of_temperatures);

  for(int i = 0; i < Number_of_temperatures; ++i)
    {
      temperatures[i] = 1.0*pow(1.1,i);
      likelihood_files[i] = "Lklhd-NGtest.dat"+std::to_string(i);
    }
  likelihood_files[0] = "Lklhd-NGtest.dat";



  // Set the CPU distribution
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
  


  // Run the Sampler                            
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-NGtest.dat", "Lklhd-NGtest.dat", 
		     "Chi2-NGtest.dat", means, ranges, var_names, false, verbosity, false, temperatures);


  MC_obj.estimate_bayesian_evidence(likelihood_files, temperatures, burn_in);
#endif

  // Finalize MPI
  MPI_Finalize();
  return 0;
}

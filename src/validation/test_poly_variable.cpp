#include "data_visibility_amplitude.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_polynomial_variable.h"
#include "likelihood.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
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
  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/PolyVarTest/gaussian_test.d"),"HH");

  std::cout << "Printing data:" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank==0)
    for (size_t j=0; j<dVM.size(); ++j) 
      std::cout << "VMdata:"
		<< std::setw(15) << dVM.datum(j).tJ2000-dVM.datum(0).tJ2000
		<< std::setw(15) << dVM.datum(j).u
		<< std::setw(15) << dVM.datum(j).v
		<< std::setw(15) << dVM.datum(j).V
		<< std::setw(15) << dVM.datum(j).err
		<< std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Finished printing data:" << std::endl;

  // Create model
  Themis::model_image_symmetric_gaussian image_initrinsic;
  //Themis::model_image_symmetric_gaussian image;
  Themis::model_image_polynomial_variable image(image_initrinsic,1,dVM.datum(0).tJ2000);
  //Themis::model_image_polynomial_variable image(image_initrinsic,0,dVM.datum(0).tJ2000);

  
  // Container of base prior class pointers
  double uas2rad = 1e-6 / 3600. /180. * M_PI;
  double image_scale = 4e2*uas2rad;
  std::vector<Themis::prior_base*> P;
  // First gaussian
  //P.push_back(new Themis::prior_logarithmic(1e-9,1e2)); // I 
  //P.push_back(new Themis::prior_logarithmic(1e-3*image_scale,image_scale)); // sigma
  P.push_back(new Themis::prior_linear(0,10)); // I 
  P.push_back(new Themis::prior_linear(-1e-4,1e-4)); // dI/dt
 
  P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
  P.push_back(new Themis::prior_linear(-1e-4*image_scale,1e-4*image_scale));    // dsigma/dt

  
  // Prior means and ranges
  std::vector<double> means, ranges;
  means.push_back(1);
  means.push_back(1.0/(24.*3600.));

  means.push_back(20.0*uas2rad);
  means.push_back(0.0);

  ranges.push_back(0.5);
  ranges.push_back(1e-7);

  ranges.push_back(10.0*uas2rad);
  ranges.push_back(1e-7*uas2rad);


  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  
  // Set the likelihood functions
  Themis::likelihood_visibility_amplitude lva(dVM,image);
  std::vector<Themis::likelihood_base*> L;
  L.push_back(&lva);

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);


  std::vector<double> pstart;
  pstart.push_back(1.0);
  pstart.push_back(1.0/(24.*3600));
  pstart.push_back(20.0*uas2rad);
  pstart.push_back(0.0);
  image.generate_model(pstart);
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank==0)
  {
    std::ofstream resout("VA_residuals.d");
    for (size_t j=0; j<dVM.size(); ++j) 
      resout << "VMdata:"
	     << std::setw(15) << dVM.datum(j).tJ2000-dVM.datum(0).tJ2000
	     << std::setw(15) << dVM.datum(j).u
	     << std::setw(15) << dVM.datum(j).v
	     << std::setw(15) << dVM.datum(j).V
	     << std::setw(15) << dVM.datum(j).err
	     << std::setw(15) << image.visibility_amplitude(dVM.datum(j),0)
	     << std::endl;
    resout.flush();
    resout.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //return 0;

  // Create a sampler object
  int seed = 42;
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);
  
  // Generate a chain
  int Number_of_chains = 120;
  int Number_of_temperatures = 4;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;

  int Number_of_steps_A = 10000;


  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set tempering schedule
  MCMC_obj.set_tempering_schedule(1000.,1.,2.0);
    
  
  /////////////
  // First run from random positions
  // Run the Sampler
  MCMC_obj.run_sampler(L_obj, Number_of_steps_A, Temperature_stride, Chi2_stride, "Chain.dat", "Lklhd.dat", "Chi2.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);

  ////////////
  // Prepare for second run:
  // Get the best fit and produce residual/gain files
  std::vector<double> pmax = MCMC_obj.find_best_fit("Chain.dat","Lklhd.dat");
  L_obj(pmax);

  std::cerr << "Read pmax run A\n";
  if (world_rank==0)
  {
    std::cout << "-------------------------\n";
    for (size_t j=0; j<pmax.size(); ++j)
      std::cout << "pmax[" << j << "] = " << pmax[j] << '\n';
    std::cout << "-------------------------\n";
    std::cout << std::endl;
  }


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

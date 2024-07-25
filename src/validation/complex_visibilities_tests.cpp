/*! 
  \file complex_visibilities_tests.cpp
  \author Avery E. Broderick
  \date  February 2020
  \brief Test rig for developing and testing the likelihood_small_gain_correction_visibility_amplitude class.
  \details Runs various tests for various sets of test data to determine how well the gains can be reconstructed and mitigated in a simple parameter estimation study.  

  The test data set to use can be set on the command line.  Options are:\n
  - 0 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise. (DEFAULT)
  - 1 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise, including constant gain errors of order 10%, with the LMT at 90%.
  - 2 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise, including variable gain errors of order 10%, with the LMT at 90%.
  - 3 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise.
  - 4 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise, including constant gain errors of order 10%, with the LMT at 90%.
  - 5 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise, including variable gain errors of order 10%, with the LMT at 90%.
  - -? .. Get the best fit for each case
*/

#include <mpi.h>
#include <memory> 
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include "data_visibility.h"
#include "model_symmetric_gaussian.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "utils.h"
#include "likelihood.h"
#include "sampler_differential_evolution_deo_tempered_MCMC.h"
#include "optimizer_kickout_powell.h"


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  int seed = 42;


  //bool run_chain=false;
  bool run_chain=true;
  int test_data_set_selection=0;
  if (argc>1)
    test_data_set_selection=atoi(argv[1]);
  std::string data_file_name;
  if (test_data_set_selection<0)
  {
    run_chain=false;
    test_data_set_selection *= -1;
  }

  if (test_data_set_selection==0)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_perfect.d";
  else if (test_data_set_selection==1)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_constant.d";
  else if (test_data_set_selection==2)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_variable.d";
  else if (test_data_set_selection==3)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_perfect_we.d";
  else if (test_data_set_selection==4)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_constant_we.d";
  else if (test_data_set_selection==5)
    data_file_name = "sim_data/TestComplexVisibilities/gaussian_test_variable_we.d";
  else
  {
    std::cerr << "Data set choice not recognized, see documentation for options.\n\n";
    std::exit(1);
  }

  std::cerr << "Using data file: " << data_file_name << "\n\n";

  // Read in data
  Themis::data_visibility data(Themis::utils::global_path(data_file_name),"HH");

  // Create model object
  Themis::model_symmetric_gaussian model;

  // Output a sample of the data being used for clarity
  std::cout << "\nData Sample:\n";
  for (size_t i=0; i<10; ++i)
    std::cout << std::setw(15) << data.datum(i).tJ2000 - data.datum(0).tJ2000
	      << std::setw(15) << data.datum(i).Station1
	      << std::setw(15) << data.datum(i).Station2
	      << std::setw(15) << data.datum(i).u
	      << std::setw(15) << data.datum(i).v
	      << std::setw(15) << data.datum(i).V.real()
	      << std::setw(15) << data.datum(i).err.real()
	      << std::setw(15) << data.datum(i).V.imag()
	      << std::setw(15) << data.datum(i).err.imag()
	      << std::endl;
  std::cout << std::endl;

  // Get the EHT standard station codes
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  //  Output them for clarity
  std::cout << "Station Codes:" << std::endl;
  for (size_t i=0; i<station_codes.size(); ++i)
    std::cout << std::setw(3) << station_codes[i];
  std::cout << '\n' << std::endl;

  // Specify the priors we will be assuming (to 10% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

  // Create the likelihood object -- THIS IS THE THING WE ARE TESTING
  //Themis::likelihood_visibility lvc(data,model);
  Themis::likelihood_optimal_complex_gain_visibility lvc(data,model,station_codes,station_gain_priors);

  // Select a default set of parameters at the "true" values to test gain reconstruction
  std::vector<double> p;
  p.push_back(2.0);
  p.push_back(15.0*1e-6/3600.*M_PI/180.0);

  {
    // Evaluate the likelihood
    std::cout << "Likelihood: " << lvc(p) << std::endl;    
  }

  //lvc.assume_independently_varying_gains();


  lvc.output_gains("complex_gains_truth.d");
  lvc.output_gain_corrections("gain_corrections_truth.d");
  lvc.output_model_data_comparison("visibility_residuals_truth.d");

  ///////////////
  // Run a chain
  //
  // Container of base prior class pointers
  double image_scale = 100.0e-6 / 3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,10.0)); // Itotal
  P.push_back(new Themis::prior_linear(0,image_scale)); // Size

  std::vector<double> means, ranges;
  means.push_back(3.0);
  means.push_back(0.5*image_scale);
  means=p; // Start from "Truth"
  ranges.push_back(0.01);
  ranges.push_back(0.01*image_scale);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma/rad$");


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  //  Option to use the non-gain corrected likelihood
  L.push_back(&lvc);
  //  Use the gain corrected likelihood
  //L.push_back(&test);

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  double Lm = L_obj(means);
  double Lt = L_obj(p);
  if (world_rank==0) {
    std::cerr << "L at means: " << Lm << std::endl;
    std::cerr << "L at true: " << Lt << std::endl;
    std::cerr << "test at means: "<< lvc(means) << std::endl;
    std::cerr << "p/means: " << std::setw(15) << p[0] << std::setw(15) << p[1]
	      << std::setw(15) << means[0] << std::setw(15) << means[1]
	      << std::endl;
  }
  
  // Optimizer
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);
  means = opt_obj.run_optimizer(L_obj, data.size(), means, "PreOptimizeSummary.dat");


  Themis::sampler_differential_evolution_deo_tempered_MCMC MCMC_obj(seed+world_rank);



  int Number_of_temperatures = world_size;
  int Number_of_walkers=16;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 100;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;
  int number_of_rounds = 4;
  int round_geometric_factor = 2;
  int thin_factor = 10;
  int Number_of_steps = 200;


  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set annealing schedule
  MCMC_obj.set_annealing_schedule(number_of_rounds,round_geometric_factor);

  // Parallelization settings
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_walkers, Number_of_procs_per_lklhd);


  std::vector<double> pmax = means;


  if (run_chain) {
    // Run sampler
    MCMC_obj.run_sampler(L_obj, Number_of_steps, thin_factor, Temperature_stride, Chi2_stride, 
			 "Chain.dat", "Lklhd.dat", "Chi2.dat", "Annealing.dat",
			 means, ranges, var_names, restart_flag, out_precision, verbosity);
    
    // Output best fits
    pmax = MCMC_obj.find_best_fit("Chain.dat.r"+std::to_string(number_of_rounds-1),"Lklhd.dat.r"+std::to_string(number_of_rounds-1));
  }


  if (world_rank==0)
    std::cerr << "Read pmax: " << std::setw(15) << pmax[0] << std::setw(15) << pmax[1] << '\n';

  L_obj(pmax);

  lvc.output_gains("complex_gains.d");
  lvc.output_gain_corrections("gain_corrections.d");
  lvc.output_model_data_comparison("visibility_residuals.d");


  double chisq = L_obj.chi_squared(pmax);
  size_t ngains = lvc.number_of_independent_gains();
  size_t nparams = model.size();
  size_t ndata = 2*data.size();

  if (world_rank==0)
    std::cerr << "Chi squared: " << std::setw(15) << chisq << '\n'
	      << "number of independent gains: " << std::setw(15) << ngains << '\n'
	      << "number of parameters: " << std::setw(15) << nparams << '\n'
	      << "number of data points: " << std::setw(15) << ndata << '\n' 
	      << "degrees of freedom: " << std::setw(15) << ndata-ngains-nparams << '\n'
	      << '\n'
	      << "Reduced chi squared: " << std::setw(15) << chisq/(ndata-ngains-nparams) << '\n';
  
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

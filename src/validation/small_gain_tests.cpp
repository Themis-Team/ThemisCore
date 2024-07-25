/*! 
  \file small_gain_tests.cpp
  \author Avery E. Broderick
  \date  August 2018
  \brief Test rig for developing and testing the likelihood_small_gain_correction_visibility_amplitude class.
  \details Runs various tests for various sets of test data to determine how well the gains can be reconstructed and mitigated in a simple parameter estimation study.  

  The test data set to use can be set on the command line.  Options are:\n
  - 0 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas without thermal noise. (DEFAULT)
  - 1 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas without thermal noise, including constant gain errors of order 20%.
  - 2 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas without thermal noise, including variable gain errors of order 20%, with the LMT at 100%.
  - 3 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas with thermal noise.
  - 4 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas with thermal noise, including constant gain errors of order 20%.
  - 5 ... An idealized Gaussian data set with amplitude 2.5 Jy and sigma 28 uas with thermal noise, including variable gain errors of order 20%, with the LMT at 100%.
*/

#include <mpi.h>
#include <memory> 
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include "data_visibility_amplitude.h"
#include "model_symmetric_gaussian.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "utils.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"



int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  int test_data_set_selection=0;
  if (argc>1)
    test_data_set_selection=atoi(argv[1]);
  std::string data_file_name;
  if (test_data_set_selection==0)
    data_file_name = "sim_data/GainTest/gaussian_test_perfect.d";
  else if (test_data_set_selection==1)
    data_file_name = "sim_data/GainTest/gaussian_test_constant.d";
  else if (test_data_set_selection==2)
    data_file_name = "sim_data/GainTest/gaussian_test_variable.d";
  else if (test_data_set_selection==3)
    data_file_name = "sim_data/GainTest/gaussian_test_perfect_we.d";
  else if (test_data_set_selection==4)
    data_file_name = "sim_data/GainTest/gaussian_test_constant_we.d";
  else if (test_data_set_selection==5)
    data_file_name = "sim_data/GainTest/gaussian_test_variable_we.d";
  else
  {
    std::cerr << "Data set choice not recognized, see documentation for options.\n\n";
    std::exit(1);
  }

  std::cerr << "Using data file: " << data_file_name << "\n\n";

  // Read in data
  Themis::data_visibility_amplitude data(Themis::utils::global_path(data_file_name),"HH");

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
	      << std::setw(15) << data.datum(i).V
	      << std::setw(15) << data.datum(i).err
	      << std::endl;
  std::cout << std::endl;

  // Get the EHT standard station codes
  std::vector<std::string> station_codes = Themis::utils::station_codes();
  //  Output them for clarity
  std::cout << "Station Codes:" << std::endl;
  for (size_t i=0; i<station_codes.size(); ++i)
    std::cout << std::setw(3) << station_codes[i];
  std::cout << '\n' << std::endl;

  // Specify the priors we will be assuming (to 10% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  //if (test_data_set_selection==2 || test_data_set_selection==5) // If we are testing the LMT gain problem, let it vary by 100%
  //  station_gain_priors[6] = 1.0;
  station_gain_priors[6] = 1.0; // Allow the LMT to vary by 100%

  // Create the likelihood object -- THIS IS THE THING WE ARE TESTING
  Themis::likelihood_optimal_gain_correction_visibility_amplitude test(data,model,station_codes,station_gain_priors);
    
  // Select a default set of parameters at the "true" values to test gain reconstruction
  std::vector<double> p;
  p.push_back(2.5);
  p.push_back(28.0*1e-6/3600.*M_PI/180.0);

  {
    // Evaluate the likelihood
    std::cout << "Likelihood: " << test(p) << std::endl;
    
    // Get and output the gain estimates
    std::vector<double> tge = test.get_gain_correction_times();
    std::vector< std::vector<double> > gge = test.get_gain_corrections();
    std::ofstream gout("gain_corrections_o.d");
    for (size_t i=0; i<tge.size()-1; ++i) {
      gout << std::setw(15) << tge[i]-tge[0];
      for (size_t j=0; j<gge[0].size(); ++j)
	gout << std::setw(15) << gge[i][j];
      gout << '\n';
    }
    gout.flush();

    // Test new output
    test.output_gain_corrections("gain_corrections_direct.d");

    double csq = test.chi_squared(p);
    double noig = test.number_of_independent_gains();

    if (world_rank==0)
    {
      std::cout << "\nTRUE FIT STATISTICS: ----------------------\n";
      std::cout << "Chi-Sq: " << csq << std::endl;
      std::cout << "Ngains = " << noig << std::endl;
      std::cout << "DoFs = " << int(data.size()) - int(noig) << std::endl;
      std::cout << "\nTRUE FIT STATISTICS: ----------------------\n" << std::endl;
    }
  }




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
  //L.push_back(new Themis::likelihood_visibility_amplitude(data,model));
  //  Use the gain corrected likelihood
  L.push_back(&test);


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  double Lm = L_obj(means);
  double Lt = L_obj(p);
  if (world_rank==0) {
    std::cerr << "L at means: " << Lm << std::endl;
    std::cerr << "L at true: " << Lt << std::endl;
    std::cerr << "test at means: "<< test(means) << std::endl;
    std::cerr << "p/means: " << std::setw(15) << p[0] << std::setw(15) << p[1]
	      << std::setw(15) << means[0] << std::setw(15) << means[1]
	      << std::endl;
  }
  
  
  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 4;
  int Number_of_steps = 5000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 0;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 1);
  MC_obj.run_sampler(L_obj, 
		     Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-GainTest-4T.dat", "Lklhd-GainTest-4T.dat", 
		     "Chi2-GainTest-4T.dat", means, ranges, var_names, false, 10, verbosity);

  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

/*! 
  \file crosshand_visibilities_tests.cpp
  \author Avery E. Broderick
  \date  March 2020
  \brief Test rig for developing and testing the likelihood_small_gain_correction_visibility_amplitude class.
  \details Runs various tests for various sets of test data to determine how well the gains can be reconstructed and mitigated in a simple parameter estimation study.  

  The test data set to use can be set on the command line.  Options are:\n
  - 1 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise. (DEFAULT)
  - 2 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise, including constant D-term errors, randomly chosen with amplitude<=1.
  - 3 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise, including constant gain errors of order 10%, assuming left and right gains are equal, with the LMT at 90%,
  - 4 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas without thermal noise, including constant gain errors of order 10%, assuming left and right gains are independent, with the LMT at 90%,
  - 10- ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise.
  - 20 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise, including constant D-term errors, randomly chosen with amplitude<=1.
  - 30 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise, including constant gain errors of order 10%, assuming left and right gains are equal, with the LMT at 90%,
  - 40 ... An idealized Gaussian data set with amplitude 2.0 Jy and sigma 15 uas with thermal noise, including constant gain errors of order 10%, assuming left and right gains are independent, with the LMT at 90%,
*/

#include <mpi.h>
#include <memory> 
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include "data_crosshand_visibilities.h"
#include "model_polarized_image_constant_polarization.h"
#include "model_image_symmetric_gaussian.h"
#include "likelihood_crosshand_visibilities.h"
#include "likelihood_optimal_complex_gain_constrained_crosshand_visibilities.h"
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

  if (test_data_set_selection==1)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_perfect.d";
  else if (test_data_set_selection==2)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms.d";
  else if (test_data_set_selection==3)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms_constrainedGains.d";
  else if (test_data_set_selection==4)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms_Gains.d";
  else if (test_data_set_selection==10)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_perfect_we.d";
  else if (test_data_set_selection==20)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms_we.d";
  else if (test_data_set_selection==30)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms_constrainedGains_we.d";
  else if (test_data_set_selection==40)
    data_file_name = "sim_data/PolarizationTests/gaussian_test_constDterms_Gains_we.d";
  else
  {
    std::cerr << "Data set choice not recognized, see documentation for options.\n\n";
    std::exit(1);
  }

  std::cerr << "Using data file: " << data_file_name << "\n\n";

  // Read in data
  Themis::data_crosshand_visibilities data(Themis::utils::global_path(data_file_name),"HH");

  // Create model object
  Themis::model_image_symmetric_gaussian stokesI_model;
  Themis::model_polarized_image_constant_polarization model(stokesI_model);

  // Output a sample of the data being used for clarity
  std::cout << "\nData Sample:\n";
  for (size_t i=0; i<10; ++i)
    std::cout << std::setw(15) << data.datum(i).tJ2000 - data.datum(0).tJ2000
	      << std::setw(15) << data.datum(i).Station1
	      << std::setw(15) << data.datum(i).Station2
	      << std::setw(15) << data.datum(i).u
	      << std::setw(15) << data.datum(i).v
	      << std::setw(15) << data.datum(i).RR.real()
	      << std::setw(15) << data.datum(i).RRerr.real()
	      << std::setw(15) << data.datum(i).RR.imag()
	      << std::setw(15) << data.datum(i).RRerr.imag()
	      << std::endl;
  std::cout << std::endl;

  // Get the EHT standard station codes
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  //  Output them for clarity
  std::cout << "Station Codes:" << std::endl;
  for (size_t i=0; i<station_codes.size(); ++i)
    std::cout << std::setw(3) << station_codes[i];
  std::cout << '\n' << std::endl;

  // Model D-terms
  model.model_Dterms(station_codes);

  model.write_model_tag_file();
  
  // Specify the priors we will be assuming (to 10% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

  // Create the likelihood object -- THIS IS THE THING WE ARE TESTING
  //Themis::likelihood_crosshand_visibilities lvc(data,model);
  Themis::likelihood_optimal_complex_gain_constrained_crosshand_visibilities lvc(data,model,station_codes,station_gain_priors);

  // Select a default set of parameters at the "true" values to test gain reconstruction
  std::vector<double> p;
  p.push_back(2.0); // I
  p.push_back(15.0*1e-6/3600.*M_PI/180.0); // sigma
  p.push_back(0.001); // m
  p.push_back(0.0); // phi
  p.push_back(0.0); // theta

  // Add truths
  if (model.modeling_Dterms())
    for (size_t j=0; j<model.number_of_Dterms(); ++j)
    {
      p.push_back(0.0);
      p.push_back(0.0);
    }
  {
    // Evaluate the likelihood
    std::cout << "Likelihood: " << lvc(p) << std::endl;    
  }

  //lvc.assume_independently_varying_gains();

  lvc.output_gains("complex_gains_truth.d");
  lvc.output_gain_corrections("gain_corrections_truth.d");
  lvc.output_model_data_comparison("crosshand_visibilities_residuals.d");


  ///////////////
  // Run a chain
  //
  // Container of base prior class pointers
  double image_scale = 100.0e-6 / 3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,10.0)); // Itotal
  P.push_back(new Themis::prior_linear(0,image_scale)); // Size
  P.push_back(new Themis::prior_linear(0,1.0)); // polarization fraction
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // polarization EVPA
  P.push_back(new Themis::prior_linear(-1.0,1.0)); // polarization mu

  // Add priors
  if (model.modeling_Dterms())
    for (size_t j=0; j<model.number_of_Dterms(); ++j)
    {
      P.push_back(new Themis::prior_linear(-1.0,1.0));
      P.push_back(new Themis::prior_linear(-1.0,1.0));
    }



  std::vector<double> means, ranges;
  means.push_back(3.0);
  means.push_back(0.5*image_scale);
  means=p; // Start from "Truth"
  ranges.push_back(0.001);
  ranges.push_back(0.001*image_scale);
  ranges.push_back(0.0001);
  ranges.push_back(0.001);
  ranges.push_back(0.001);
  if (model.modeling_Dterms())
    for (size_t j=0; j<model.number_of_Dterms(); ++j)
    {
      ranges.push_back(0.001);
      ranges.push_back(0.001);
    }
  
  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  //  Option to use the non-gain corrected likelihood
  L.push_back(&lvc);

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  double Lm = L_obj(means);
  double Lt = L_obj(p);
  if (world_rank==0) {
    std::cerr << "L at means: " << Lm << std::endl;
    std::cerr << "L at true: " << Lt << std::endl;
    std::cerr << "p:     ";
    for (size_t j=0; j<model.size(); ++j)
      std::cerr << std::setw(15) << p[j];
    std::cerr << '\n';
    std::cerr << "means: ";
    for (size_t j=0; j<model.size(); ++j)
      std::cerr << std::setw(15) << means[j];
    std::cerr << '\n';
  }
  
  // Optimizer
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);

  opt_obj.set_kickout_parameters(10.0, 10, 10);
  means = opt_obj.run_optimizer(L_obj, data.size(), means, "PreOptimizeSummary.dat");

  /*
  means[0]=1.95109;
  means[1]=7.27221e-11;
  means[2]=0.118017;
  means[3]=1.60149;
  means[4]=0.557627;
  means[5]=0.00632018;
  means[6]=0.135496;
  means[7]=-0.01918;
  means[8]=-0.138678;
  means[9]=-0.0214697;
  means[10]=0.137626;
  means[11]=-0.043082;
  means[12]=-0.141746;
  means[13]=0.00114561;
  means[14]=0.140191;
  means[15]=-0.0245374;
  means[16]=-0.14186;
  means[17]=-0.00805208;
  means[18]=0.145531;
  means[19]=-0.0356325;
  means[20]=-0.14059;
  means[21]=-0.00792366;
  means[22]=0.140369;
  means[23]=-0.0336251;
  means[24]=-0.140191;
  means[25]=-0.00210064;
  means[26]=0.139885;
  means[27]=-0.0270791;
  means[28]=-0.142449;
  means[29]=-0.00312781;
  means[30]=0.13632;
  means[31]=-0.031368;
  means[32]=-0.132838;
  means[33]=-0.1166;
  means[34]=-0.835131;
  means[35]=-0.109446;
  means[36]=0.998733;


  means[0] = 2.1362413;
  means[1] = 7.3041836e-11;
  means[2] = 0.30593943;
  means[3] = 1.063686;
  means[4] = 0.30331887;
  means[5] = 0.095657029;
  means[6] = -0.020351793;
  means[7] = -0.049639919;
  means[8] = -0.038531172;
  means[9] = 0.09212149;
  means[10] = -0.020861154;
  means[11] = -0.052885606;
  means[12] = -0.029739702;
  means[13] = 0.089492521;
  means[14] = -0.022679869;
  means[15] = -0.047832495;
  means[16] = -0.0353322;
  means[17] = 0.080468306;
  means[18] = -0.010867117;
  means[19] = -0.037492933;
  means[20] = -0.041679682;
  means[21] = 0.094739931;
  means[22] = -0.01708616;
  means[23] = -0.053293866;
  means[24] = -0.02740104;
  means[25] = 0.11292935;
  means[26] = -0.019769222;
  means[27] = -0.072084152;
  means[28] = -0.049828457;
  means[29] = 0.078425657;
  means[30] = -0.022350562;
  means[31] = -0.048603324;
  means[32] = -0.029226669;
  means[33] = 0.061154972;
  means[34] = 0.84441216;
  means[35] = -0.096924931;
  means[36] = 0.40845578;
  */

  Lm = L_obj(means);
  Lt = L_obj(p);
  if (world_rank==0) {
    std::cerr << "L at means: " << Lm << std::endl;
    std::cerr << "L at true: " << Lt << std::endl;
    std::cerr << "p:     ";
    for (size_t j=0; j<model.size(); ++j)
      std::cerr << std::setw(15) << p[j];
    std::cerr << '\n';
    std::cerr << "means: ";
    for (size_t j=0; j<model.size(); ++j)
      std::cerr << std::setw(15) << means[j];
    std::cerr << '\n';
  }


  Themis::sampler_differential_evolution_deo_tempered_MCMC MCMC_obj(seed+world_rank);

  int Number_of_temperatures = world_size;
  int Number_of_walkers=256;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 10000;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;
  int number_of_rounds = 7;
  int round_geometric_factor = 2;
  int thin_factor = 10;
  int Number_of_steps = 500;


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


  L_obj(pmax);
  std::cerr << "Read pmax\n";

  lvc.output_gains("complex_gains.d");
  lvc.output_gain_corrections("gain_corrections.d");
  lvc.output_model_data_comparison("crosshand_visibilities_residuals.d");


  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

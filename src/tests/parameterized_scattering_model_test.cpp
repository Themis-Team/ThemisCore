/*!
  \file tests/parameterized_scattering_model_test.cpp
  \author Avery Broderick
  \date June 2018
  \test Test of model_ensemble_averaged_parameterized_scattered_image.
  \details Fits a fixed symmetric Gaussian model plus a wavelength-dependent scattering model to simulated visibility amplitude data.
*/

#include "data_visibility_amplitude.h"
#include "model_symmetric_gaussian.h"
#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "model_ensemble_averaged_parameterized_scattered_image.h"
#include "likelihood.h"
#include "likelihood_visibility_amplitude.h"
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
  Themis::data_visibility_amplitude testdata(Themis::utils::global_path("sim_data/multiwavelength_scattering_screent_visibility_amplitude_test_data.d"),"HHMM",true);
  
  // Choose the model to compare
  Themis::model_symmetric_gaussian intrinsic_image;
  Themis::model_ensemble_averaged_parameterized_scattered_image image(intrinsic_image,230e9);

  // Set parameters
  std::vector<double> params(9);
  double uas2rad = 1e-6/3600. * M_PI/180.;
  // Gaussian parameters
  params[0] = 2.5;
  params[1] = 5.0 * uas2rad;
  // Screen parameters
  params[2] = 8.45e-4;
  params[3] = 2.0;
  params[4] = 1.69e-4;
  params[5] = 2.0;
  params[6] = -78.0;
  params[7] = 0.0;
  params[8] = 0.0;
  
  image.generate_model(params);
  if (world_rank==0)
  {
    for (size_t i=0; i<testdata.size(); ++i)
      std::cout << "VAData:"
		<< std::setw(15) << testdata.datum(i).u
		<< std::setw(15) << testdata.datum(i).v
		<< std::setw(15) << testdata.datum(i).frequency
		<< std::setw(15) << testdata.datum(i).V
		<< std::setw(15) << testdata.datum(i).err
		<< std::setw(15) << image.visibility_amplitude(testdata.datum(i),0.0)
		<< std::setw(15) << intrinsic_image.visibility_amplitude(testdata.datum(i),0.0)
		<< std::endl;
  }


  // Container of base prior class pointers
  // and prior means and ranges
  double image_size = 10. * 1.e-6 /3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  // Gaussian parameters
  // Itotal
  P.push_back(new Themis::prior_linear(0,3));
  means.push_back(params[0]);
  ranges.push_back(1.0e-2);
  // sigma
  P.push_back(new Themis::prior_linear(0.01*image_size,10.0*image_size));
  means.push_back(params[1]);
  ranges.push_back(0.01*image_size);
  // Scattering screen parameters
  // sigma_maj at 230 GHz
  P.push_back(new Themis::prior_logarithmic(1e-5,10.0));
  means.push_back(params[2]);
  ranges.push_back(0.01);
  // sigma_maj power law index
  P.push_back(new Themis::prior_linear(-5.0,5.0));
  means.push_back(params[3]);
  ranges.push_back(0.01);
  // sigma_min at 230 GHz
  P.push_back(new Themis::prior_logarithmic(1e-5,10.0));
  means.push_back(params[4]);
  ranges.push_back(0.01);
  // sigma_min power law index
  P.push_back(new Themis::prior_linear(-5.0,5.0));
  means.push_back(params[5]);
  ranges.push_back(0.01);
  // position angle at 230 GHz
  P.push_back(new Themis::prior_linear(-180.0,180.0));
  means.push_back(params[6]);
  ranges.push_back(0.1);
  // position angle variation norm at 230 GHz
  P.push_back(new Themis::prior_linear(-720.0,720.0));
  means.push_back(params[7]);
  ranges.push_back(0.1);
  // position angle variation power law index
  P.push_back(new Themis::prior_linear(-5.0,5.0));
  means.push_back(params[8]);
  ranges.push_back(0.01);
  
  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma$");
  var_names.push_back("$\\sigma_A$");
  var_names.push_back("$\\alpha$");
  var_names.push_back("$\\sigma_B$");
  var_names.push_back("$\\beta$");
  var_names.push_back("$\\xi_0$");
  var_names.push_back("$\\xi_1$");
  var_names.push_back("$\\gamma$");

	
  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_visibility_amplitude(testdata,image));

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
		     "Chain-PSST.dat", "Lklhd-PSST.dat", 
		     "Chi2-PSST.dat", means, ranges, var_names, false, verbosity);


  //Finalize MPI
  MPI_Finalize();
  return 0;
}

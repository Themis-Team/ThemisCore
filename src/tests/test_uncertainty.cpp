/*!
  \file tests/test_uncertainty.cpp
  \author Avery Broderick
  \date September 2020
  \test uncertainty_visibility implementation test
  \brief Reads in some data and a light curve and generates some output data with uncertainty prescriptions
  
  \details TBD
*/

#include "data_visibility.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_fixed_lightcurve.h"
#include "uncertainty_visibility.h"
#include "uncertainty_visibility_loose_change.h"
#include "uncertainty_visibility_power_change.h"
#include "uncertainty_visibility_broken_power_change.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "utils.h"
#include <mpi.h>
#include <string>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in visibility test data
  //Themis::data_visibility data(Themis::utils::global_path("sim_data/TestLightCurve/gaussian_test_variable.d"),"HH");
  Themis::data_visibility data(Themis::utils::global_path("sim_data/TestUncertainty/V_simdata_3601_ring_seed3+netcal_scan_average_scanavg_tygtd.dat"),"HH");
  //Themis::data_visibility data(Themis::utils::global_path("sim_data/TestUncertainty/V_simdata_3601_ring_seed3+netcal_scan_average_scanavg_f1_tygtd.dat"),"HH");
  //Themis::data_visibility data(Themis::utils::global_path("sim_data/TestUncertainty/V_simdata_3601_ring_seed3+netcal_scan_average_scanavg_f1_t10mJy_tygtd.dat"),"HH");

  // Choose the model to compare
  // Themis::model_image_gaussian image;
  Themis::model_image_symmetric_gaussian image;
  //Themis::model_image_fixed_lightcurve image(static_image,Themis::utils::global_path("sim_data/TestLightCurve/light_curve.d"));
  

  //std::cerr << "Size of static image " << static_image.size() << '\n';
  std::cerr << "Size of image " << image.size() << '\n';
  
  std::vector<double> parameters;
  parameters.push_back(1.0);
  parameters.push_back(40.0/2.35 * M_PI/(180*3600e6) );
  image.generate_model(parameters);

  
  
  // Output some visibilities at different times
  for (size_t i=0; i<data.size(); ++i)
  {
    ///std::complex<double> Vs = static_image.visibility(data.datum(i),0.0);
    std::complex<double> Vs = image.visibility(data.datum(i),0.0);
    std::complex<double> Vv = image.visibility(data.datum(i),0.0);
    
    std::cout << std::setw(15) << data.datum(i).tJ2000-data.datum(0).tJ2000
	      << std::setw(15) << std::sqrt( data.datum(i).u*data.datum(i).u + data.datum(i).v*data.datum(i).v )/1.0e9
      	      << std::setw(15) << std::abs(Vs)
      	      << std::setw(15) << std::abs(Vv)
      	      << std::setw(15) << Vs.real()
      	      << std::setw(15) << Vs.imag()
      	      << std::setw(15) << Vv.real()
      	      << std::setw(15) << Vv.imag()
	      << std::endl;
  }
  

  // Create a likelihood
  //Themis::uncertainty_visibility_loose_change uncertainty;
  //Themis::uncertainty_visibility_power_change uncertainty;
  Themis::uncertainty_visibility_broken_power_change uncertainty;
  //parameters.push_back(0.0); // threshold
  //parameters.push_back(0.0); // frac
  parameters.push_back(0.00); // threshold
  parameters.push_back(0.00); // frac

  parameters.push_back(0.10); // zbl error
  parameters.push_back(2e9); // baseline break
  parameters.push_back(2.0); // long index
  parameters.push_back(5.0); // short index
  
  Themis::likelihood_visibility lklhd(data,image,uncertainty);

  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  //Themis::likelihood_optimal_complex_gain_visibility lklhd_g(data,image,station_codes,station_gain_priors);
  Themis::likelihood_optimal_complex_gain_visibility lklhd_g(data,image,uncertainty,station_codes,station_gain_priors);
  
  std::cerr << "Likelhood = " << lklhd(parameters) << '\n';
  std::cerr << "Likelhood w/ gains = " << lklhd_g(parameters) << '\n';

  lklhd.output_model_data_comparison("residuals.d");
  lklhd_g.output_model_data_comparison("residuals_gains.d");

  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

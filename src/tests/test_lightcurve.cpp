/*!
  \file tests/test_lightcurve.cpp
  \author Avery Broderick
  \date September 2020
  \test model_image_lightcurve implementation test
  \brief Reads in some data and a light curve and generates some output data
  
  \details TBD
*/

#include "data_visibility.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_fixed_lightcurve.h"
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
  Themis::data_visibility data(Themis::utils::global_path("sim_data/TestLightCurve/gaussian_test_variable.d"),"HH");

  // Choose the model to compare
  // Themis::model_image_gaussian image;
  Themis::model_image_symmetric_gaussian static_image;
  Themis::model_image_fixed_lightcurve image(static_image,Themis::utils::global_path("sim_data/TestLightCurve/light_curve.d"));
  

  std::cerr << "Size of static image " << static_image.size() << '\n';
  std::cerr << "Size of image " << image.size() << '\n';
  
  std::vector<double> parameters;
  parameters.push_back(1.0);
  parameters.push_back(40.0/2.35 * M_PI/(180*3600e6) );
  image.generate_model(parameters);

  
  // Output some visibilities at different times
  for (size_t i=0; i<data.size(); ++i)
  {
    std::complex<double> Vs = static_image.visibility(data.datum(i),0.0);
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

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

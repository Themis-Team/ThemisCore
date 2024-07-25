/*! 
  \file tests/visibility_test.cpp
  \authors Avery E. Broderick
  \date May, 2018
  
  \brief Tests the visibility construction for simple models.

  \details
  Test to check the construction of visibility-based quantities for simple models (gaussians, crescents).
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "utils.h"

#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in visibility amplitude data
  Themis::data_visibility_amplitude VMdata(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  VMdata.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  VMdata.add_data(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  VMdata.add_data(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  VMdata.add_data(Themis::utils::global_path("eht_data/VM_2009_097.d"));


  // Read in closure phases data
  Themis::data_closure_phase CPdata(Themis::utils::global_path("eht_data/CP_2009_093.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2009_096.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2009_097.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2011_088.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2011_090.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2011_091.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2011_094.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2012_081.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2013_080.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2013_081.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2013_082.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2013_085.d"));
  CPdata.add_data(Themis::utils::global_path("eht_data/CP_2013_086.d"));


  // Choose the model to compare
  Themis::model_image_asymmetric_gaussian miag;
  Themis::model_image_crescent mic;
  Themis::model_ensemble_averaged_scattered_image scmiag(miag);
  Themis::model_ensemble_averaged_scattered_image scmic(mic);

  miag.use_numerical_visibilities();
  mic.use_numerical_visibilities();


  std::vector<double> miag_params(4);
  miag_params[0] = 1.0;
  miag_params[1] = 30.0 * 1e-6/3600. * M_PI/180.0;
  miag_params[2] = 0.7;
  miag_params[3] = 1.3;


  std::vector<double> mic_params(5);
  mic_params[0] = 1.0;
  mic_params[1] = 30.0 * 1e-6/3600. * M_PI/180.0;
  mic_params[2] = 0.1;
  mic_params[3] = 0.1;
  mic_params[4] = 2.1;

  miag.generate_model(miag_params);
  mic.generate_model(mic_params);
  scmiag.generate_model(miag_params);
  scmic.generate_model(mic_params);
  
  
  // Generate visibility amplitude values
  std::ofstream vmout("visibility_test_VMdata.d");
  for (size_t i=0; i<VMdata.size(); ++i)
    vmout << std::setw(15) << VMdata.datum(i).u
	  << std::setw(15) << VMdata.datum(i).v
	  << std::setw(15) << miag.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::setw(15) << mic.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::setw(15) << scmiag.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::setw(15) << scmic.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::endl;

  // Generate closure phase values
  std::ofstream cpout("visibility_test_CPdata.d");
  for (size_t i=0; i<CPdata.size(); ++i)
    cpout << std::setw(15) << CPdata.datum(i).u1
	  << std::setw(15) << CPdata.datum(i).v1
	  << std::setw(15) << CPdata.datum(i).u2
	  << std::setw(15) << CPdata.datum(i).v2
	  << std::setw(15) << CPdata.datum(i).u3
	  << std::setw(15) << CPdata.datum(i).v3
	  << std::setw(15) << miag.closure_phase(CPdata.datum(i),0.0)
	  << std::setw(15) << mic.closure_phase(CPdata.datum(i),0.0)
	  << std::endl;

  //Finalize MPI
  MPI_Finalize();
  return 0;
}

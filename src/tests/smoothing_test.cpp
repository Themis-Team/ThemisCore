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
#include "model_image_crescent.h"
#include "model_image_smooth.h"
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
  Themis::model_image_crescent cr;
  Themis::model_image_smooth smcr(cr);
  cr.use_analytical_visibilities();




  std::vector<double> cr_params(5);
  cr_params[0] = 1.0;
  cr_params[1] = 30.0 * 1e-6/3600. * M_PI/180.0;
  cr_params[2] = 0.1;
  cr_params[3] = 0.1;
  cr_params[4] = 2.1;

  std::vector<double> smcr_params(8);
  smcr_params[0] = 1.0;
  smcr_params[1] = 30.0 * 1e-6/3600. * M_PI/180.0;
  smcr_params[2] = 0.1;
  smcr_params[3] = 0.1;
  smcr_params[4] = 2.1;
  smcr_params[5] = 2.0 * 1e-6/3600. * M_PI/180.0;
  smcr_params[6] = 2.0 * 1e-6/3600. * M_PI/180.0;
  smcr_params[7] = 0;


  cr.generate_model(cr_params);
  smcr.generate_model(smcr_params);
  
  // Generate visibility amplitude values
  std::ofstream vmout("smoothing_test_VMdata.d");
  for (size_t i=0; i<VMdata.size(); ++i)
    vmout << std::setw(15) << VMdata.datum(i).u
	  << std::setw(15) << VMdata.datum(i).v
	  << std::setw(15) << cr.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::setw(15) << smcr.visibility_amplitude(VMdata.datum(i),0.0)
	  << std::endl;
  std::cerr << "CP test" << smcr.closure_phase(CPdata.datum(0), 0) << std::endl;
  // Generate closure phase values
  std::ofstream cpout("smoothing_test_CPdata.d");
  for (size_t i=0; i<CPdata.size(); ++i)
    cpout << std::setw(15) << CPdata.datum(i).u1
	  << std::setw(15) << CPdata.datum(i).v1
	  << std::setw(15) << CPdata.datum(i).u2
	  << std::setw(15) << CPdata.datum(i).v2
	  << std::setw(15) << CPdata.datum(i).u3
	  << std::setw(15) << CPdata.datum(i).v3
	  << std::setw(15) << cr.closure_phase(CPdata.datum(i),0.0)
	  << std::setw(15) << smcr.closure_phase(CPdata.datum(i),0.0)
	  << std::endl;

  std::cerr << "Here 2\n";
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

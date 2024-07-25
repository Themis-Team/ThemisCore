#include "data_visibility_amplitude.h"
#include "model_image_multi_asymmetric_gaussian.h"
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

  // Use the Challenge2 data as test positions
  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/Challenge2/Ch2_VMs.d"));
  Themis::data_closure_phase dCP(Themis::utils::global_path("sim_data/Challenge2/Ch2_CPs.d"));

  
  // Create model
  size_t NG=2;
  Themis::model_image_multi_asymmetric_gaussian image(NG);


  // Set some parameters
  double uas2r = 1e-6/3600*M_PI/180.;
  std::vector<double> p;
  p.push_back(1.0); // I 
  p.push_back(uas2r * 10); // sigma
  p.push_back(0.2); // A
  p.push_back(0.1*M_PI); // phi
  p.push_back(0.0); // x
  p.push_back(0.0); // y

  p.push_back(0.2); // I 
  p.push_back(uas2r * 5); // sigma
  p.push_back(0.6); // A
  p.push_back(0.4*M_PI); // phi
  p.push_back(uas2r * 40); // x
  p.push_back(uas2r * 20); // y

  p.push_back(0.0); // xi

  image.use_numerical_visibilities();
  image.generate_model(p);


  std::ofstream imout("image_test.txt");
  std::vector< std::vector<double> > alpha, beta, I;
  image.get_image(alpha,beta,I);
  for (size_t i=0; i<alpha.size(); ++i) {
    for (size_t j=0; j<alpha[i].size(); ++j)
      imout << std::setw(15) << alpha[i][j]
	    << std::setw(15) << beta[i][j]
	    << std::setw(15) << I[i][j]
	    << '\n';
    imout << '\n';
  }


  std::ofstream vmout("visibility_amplitude_test.txt");
  for (size_t j=0; j<dVM.size(); ++j)
    vmout << std::setw(15) << dVM.datum(j).u
	  << std::setw(15) << dVM.datum(j).v
	  << std::setw(15) << image.visibility_amplitude(dVM.datum(j),0)
	  << '\n';
  vmout << "\n\n";

  image.use_analytical_visibilities();
  for (size_t j=0; j<dVM.size(); ++j)
    vmout << std::setw(15) << dVM.datum(j).u
	  << std::setw(15) << dVM.datum(j).v
	  << std::setw(15) << image.visibility_amplitude(dVM.datum(j),0)
	  << '\n';


  std::ofstream cpout("closure_phase_test.txt");
  image.use_numerical_visibilities();
  for (size_t j=0; j<dCP.size(); ++j)
    cpout << std::setw(15) << dCP.datum(j).u1
	  << std::setw(15) << dCP.datum(j).v1
	  << std::setw(15) << dCP.datum(j).u2
	  << std::setw(15) << dCP.datum(j).v2
	  << std::setw(15) << dCP.datum(j).u3
	  << std::setw(15) << dCP.datum(j).v3
	  << std::setw(15) << image.closure_phase(dCP.datum(j),0)
	  << '\n';
  cpout << "\n\n";

  image.use_analytical_visibilities();
  for (size_t j=0; j<dCP.size(); ++j)
    cpout << std::setw(15) << dCP.datum(j).u1
	  << std::setw(15) << dCP.datum(j).v1
	  << std::setw(15) << dCP.datum(j).u2
	  << std::setw(15) << dCP.datum(j).v2
	  << std::setw(15) << dCP.datum(j).u3
	  << std::setw(15) << dCP.datum(j).v3
	  << std::setw(15) << image.closure_phase(dCP.datum(j),0)
	  << '\n';


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

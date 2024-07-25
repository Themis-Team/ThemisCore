/*!
  \file analyses/Challenge2/generate_multigaussian_test_data.cpp
  \author Avery E. Broderick
  \date Mar 2018
  \brief Generates test data comprised of multiple Gaussians, modeling elements of challenge 2 for testing the development of the multi-Gaussian model.

  \details Generates visibility amplitude and closure phase test data for 1-4 Gaussian components.  This is produced completely within Themis, and thus provides an internal test of the multi-Gaussian model.
*/

#include "data_visibility_amplitude.h"
#include "model_image_multigaussian.h"
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

  // Set data parameters
  std::vector<double> Icomp(0), sigma(0), x(0), y(0);
  double u2rad = M_PI/180./3600./1e6;
  
  double xshift = 0; //10*u2rad;
  double yshift = 0; //10*u2rad;

  //  Gaussian 1
  Icomp.push_back(1.0);
  sigma.push_back(15*u2rad);
  //x.push_back(-30*u2rad);
  //y.push_back(-15*u2rad);
  x.push_back(-0*u2rad+xshift);
  y.push_back(-0*u2rad+yshift);
  //  Gaussian 2
  Icomp.push_back(0.5);
  sigma.push_back(10*u2rad);
  x.push_back(40*u2rad+xshift);
  y.push_back(0*u2rad+yshift);
  /*
  //  Gaussian 3
  Icomp.push_back(0.7);
  sigma.push_back(50*u2rad);
  x.push_back(-30*u2rad+xshift);
  y.push_back(20*u2rad+yshift);
  //  Gaussian 4
  Icomp.push_back(0.2);
  sigma.push_back(10*u2rad);
  x.push_back(-5*u2rad+xshift);
  y.push_back(-30*u2rad+yshift);
  */

  std::vector<double> parameters(0);
  for (size_t j=0; j<Icomp.size(); ++j)
  {
    parameters.push_back(Icomp[j]);
    parameters.push_back(sigma[j]);
    parameters.push_back(x[j]);
    parameters.push_back(y[j]);
  }
  parameters.push_back(0.0);

  // Generate a model
  Themis::model_image_multigaussian model(Icomp.size());
  model.use_analytical_visibilities();
  //model.use_numerical_visibilities();
  model.generate_model(parameters);

  for (size_t j=0; j<parameters.size(); ++j)
    std::cout << "parameters[" << j << "]= " << parameters[j] << std::endl;


  // Generate data for a random sampling of baselines
  Themis::Ran2RNG rng(0);
  Themis::GaussianRandomNumberGenerator<Themis::Ran2RNG> grng(1);
  std::ofstream vmout("VM_Ngauss_data.d");
  vmout << '#' 
	<< std::setw(14) << "nsrc"
	<< std::setw(15) << "nyear"
	<< std::setw(15) << "nday"
	<< std::setw(15) << "ntime"
	<< std::setw(15) << "nbase"
	<< std::setw(15) << "nu"
	<< std::setw(15) << "nv"
	<< std::setw(15) << "namp"
	<< std::setw(15) << "err"
	<< "\n#\n";
  for (size_t j=0; j<800; ++j)
  {
    double u = (2*rng.rand()-1.)*6e9;
    double v = (2*rng.rand()-1.)*6e9;
    
    Themis::datum_visibility_amplitude d(u,v,0,0);

    double err = 0.01;
    double V = model.visibility_amplitude(d,0) + err*grng.rand();

    vmout << std::setw(15) << "TEST"
	  << std::setw(15) << 2018
	  << std::setw(15) << 1
	  << std::setw(15) << 1200
	  << std::setw(15) << "RN"
	  << std::setw(15) << u/1e6
	  << std::setw(15) << v/1e6
	  << std::setw(15) << V
	  << std::setw(15) << err
	  << '\n';
  }


  // Generate data for a random sampling of triangles
  std::ofstream cpout("CP_Ngauss_data.d");
  cpout << '#' 
	<< std::setw(14) << "SRC"
	<< std::setw(15) << "Year"
	<< std::setw(15) << "Day"
	<< std::setw(15) << "Hour"
	<< std::setw(15) << "Triangle"
	<< std::setw(15) << "u1"
	<< std::setw(15) << "v1"
	<< std::setw(15) << "u2"
	<< std::setw(15) << "v2"
	<< std::setw(15) << "CP"
	<< std::setw(15) << "err"
	<< "\n#\n";
  for (size_t j=0; j<400; ++j)
  {
    double u1 = (2*rng.rand()-1.)*6e9;
    double v1 = (2*rng.rand()-1.)*6e9;
    double u2 = (2*rng.rand()-1.)*6e9;
    double v2 = (2*rng.rand()-1.)*6e9;

    Themis::datum_closure_phase d(u1,v1,u2,v2,0,0);

    double err = 0.01;
    double CP = model.closure_phase(d,0) + err*grng.rand();

    cpout << std::setw(15) << "TEST"
	  << std::setw(15) << 2018
	  << std::setw(15) << 1
	  << std::setw(15) << 1200
	  << std::setw(15) << "RNG"
	  << std::setw(15) << u1/1e6
	  << std::setw(15) << v1/1e6
	  << std::setw(15) << u2/1e6
	  << std::setw(15) << v2/1e6
	  << std::setw(15) << CP
	  << std::setw(15) << err
	  << '\n';
  }


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

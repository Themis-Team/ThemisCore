/*!
  \file grtrans_image.cpp
  \author Roman Gold
  \date June 2017
  \brief Test model for grtrans image

  \details This example shows how to choose parameters and run the
  external radiative transfer code "GRTRANS" [Dexter 2016] and read in the data

  \bug File format ('raw binary' format) not correctly recognized

*/
#define VERBOSITY (0)


#include "model_image_grtrans.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <mpi.h>

int main(int argc,char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  std::vector<double> p;

  // <I>=?Jy , <LP>/<I>=?
  p.push_back(1.57e+15); // mdot
  p.push_back(60.*M_PI/180.); // inclination angle
  p.push_back(0.25); // mu_val electron temperature related
  p.push_back(0.9375); // spin
  p.push_back(-0.5); // Azimuthal viewing (i.e. position) angle

  Themis::model_image_grtrans GRTRANS_IMAGE;
  // GRTRANS_IMAGE.use_numerical_visibilities();
  GRTRANS_IMAGE.generate_model(p);
  std::vector<std::vector<double> > aN, bN, IN;
  GRTRANS_IMAGE.get_image(aN,bN,IN);


  /*
  std::cerr << a.size() << " " << b.size() << " " << I.size() << '\n';
  std::cerr << aN.size() << " " << bN.size() << " " << IN.size() << '\n';
  std::cout << "\n\n";
  for (size_t j=0; j<aN.size(); ++j)
    for (size_t k=0; k<aN[j].size(); ++k)
      std::cout << std::setw(15) << aN[j][k]
		<< std::setw(15) << bN[j][k]
		<< std::setw(15) << IN[j][k]
		<< '\n';
  std::cout << "\n\n" << std::endl;;
  */


  for (int i=0; i<10; ++i) {
    double u = 4.e9*i/9.;
    Themis::datum_visibility_amplitude d(u,0.0,1.0,0.1);
    double VA = 0., VB = 0.;
    VA = GRTRANS_IMAGE.visibility_amplitude(d,0);
    VB = GRTRANS_IMAGE.visibility_amplitude(d,0);
    // VA = GAUSSIAN.visibility_amplitude(d,0);
    // VB = GAUSSIAN_NUM.visibility_amplitude(d,0);
    std::cout << std::setw(15) << u
      << std::setw(15) << VA
	    << std::setw(15) << VB
	    << std::endl;
  }


  MPI_Finalize();

  return 0;
}

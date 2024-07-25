/*!
  \file astroray_image.cpp
  \author Roman Gold
  \date June 2017
  \brief Test model for astroray image
  
  \details This example demonstrates how to choose parameters and run the external radiative transfer code ASTRORAY, obtain an image and read it back into THEMIS
*/
#define VERBOSITY (0)


#include "model_image_astroray.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>


int main(int argc,char* argv[])
{
  MPI_Init(&argc, &argv);
  //int world_rank = MPI::COMM_WORLD.Get_rank();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  std::vector<double> p;

  // <I>=2.5Jy , <LP>/<I>=3.5%
  p.push_back(1.e5); // rhonor
  p.push_back(60.*M_PI/180.); // inclination angle
  p.push_back(10.); // Te_jet_par
  p.push_back(0.); // Azimuthal viewing (i.e. position) angle

  // best-fit for MAD-disk from https://arxiv.org/abs/1601.05550
  // p.push_back(126198.9381); // rhonor
  // p.push_back(1.73); // inclination angle
  // p.push_back(10.); // Te_jet_par
  // p.push_back(0.); // Azimuthal viewing (i.e. position) angle

  Themis::model_image_astroray ASTRORAY_IMAGE;
  // ASTRORAY_IMAGE.use_numerical_visibilities();
  ASTRORAY_IMAGE.generate_model(p);
  std::vector<std::vector<double> > aN, bN, IN;
  ASTRORAY_IMAGE.get_image(aN,bN,IN);


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
    VA = ASTRORAY_IMAGE.visibility_amplitude(d,0);
    VB = ASTRORAY_IMAGE.visibility_amplitude(d,0);
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

/*!
  \file gaussian_blob.cpp
  \author Avery Broderick
  \date May 2017
  \test Compare analytical to numerical visibilities in Gaussian model 

  \brief Check FFTs by comparing analytical to numerical visibilities
  in Gaussian model
 
  \details Make sure that FFTs are done correctly. For a Gaussian
  model visibilities are known analytical. In this test visibilities
  from two Gaussian models one with analytical and the other with
  numerical visibilities are compared against each other.
*/

#define VERBOSITY (0)

#include "model_image_gaussian.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  std::vector<double> p;
  p.push_back(2.85);
  p.push_back(43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.)))); // 43 muas
  p.push_back(43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))));
  p.push_back(0.0);

  Themis::model_image_gaussian GAUSSIAN;
  Themis::model_image_gaussian GAUSSIAN_NUM;
  GAUSSIAN_NUM.use_numerical_visibilities();

  GAUSSIAN.generate_model(p);
  GAUSSIAN_NUM.generate_model(p);


  Themis::datum_closure_phase d_CP(1e8, 1e9, 2e8, 2e9, 10., 1.);

  double CP;
  CP = GAUSSIAN.closure_phase(d_CP, 0);
  std::cerr << "Analytical: Generated CLOSURE PHASE:" << CP << std::endl;

  CP = GAUSSIAN_NUM.closure_phase(d_CP, 0);
  std::cerr << "Numerical:  Generated CLOSURE PHASE:" << CP << std::endl;


  std::vector<std::vector<double> > a, b, I;
  std::vector<std::vector<double> > aN, bN, IN;
  GAUSSIAN.get_image(a,b,I);
  GAUSSIAN.get_image(aN,bN,IN);

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


  for (int i=0; i<1000; ++i)
  {
    double u = 4.e9*i/999.;
    Themis::datum_visibility_amplitude d(u,0.0,1.0,0.1);
    
    double VA = 0., VB = 0.;
    VA = GAUSSIAN.visibility_amplitude(d,0);
    VB = GAUSSIAN_NUM.visibility_amplitude(d,0);
    
    std::cout << std::setw(15) << u 
	      << std::setw(15) << VA
	      << std::setw(15) << VB
	      << std::endl;
  }

  //Finalize MPI
  MPI_Finalize();

  return 0;
}

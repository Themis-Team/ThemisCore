#include "model_image_raster.h"
#include "model_image_upsample.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  double xmin=-50.0, xmax=45.0;
  size_t Nx=4;
  double ymin=20.0, ymax=70.0;
  size_t Ny=6;
  Themis::model_image_raster image(-50.0,45.0,4,20.0,70.0,6);

  std::vector<double> p;
  for (size_t j=0; j<Ny; ++j)
  {
    for (size_t i=0; i<Nx; ++i)
    {
      double x = (xmax-xmin)*double(i)/double(Nx-1) + xmin;
      double y = (ymax-ymin)*double(j)/double(Ny-1) + ymin;
      p.push_back( std::log(std::exp( -x*x/(2.*25.*25.))*(1.5+std::cos(2.*M_PI*y/30.0)) ) );

      std::cout << std::setw(15) << x
		<< std::setw(15) << y
		<< std::setw(15) << p[p.size()-1]
		<< std::setw(15) << std::exp(p[p.size()-1])
		<< std::setw(15) << "plist"
		<< std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  Themis::model_image_upsample image_up(image,8);
  image_up.generate_model(p);

  
  std::vector< std::vector<double> > a,b,I;


  image.get_image(a,b,I);
  for (size_t i=0; i<a.size(); ++i)
  {
    for (size_t j=0; j<a[i].size(); ++j)
      std::cout << std::setw(15) << a[i][j]
		<< std::setw(15) << b[i][j]
		<< std::setw(15) << I[i][j]
		<< std::setw(15) << "image"
		<< std::endl;
    std::cout << std::endl;
  }
  std::cout << std::endl;

  image_up.get_image(a,b,I);
  for (size_t i=0; i<a.size(); ++i)
  {
    for (size_t j=0; j<a[i].size(); ++j)
      std::cout << std::setw(15) << a[i][j]
		<< std::setw(15) << b[i][j]
		<< std::setw(15) << I[i][j]
		<< std::setw(15) << "upsamp"
		<< std::endl;
    std::cout << std::endl;
  }
  std::cout << std::endl;


  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

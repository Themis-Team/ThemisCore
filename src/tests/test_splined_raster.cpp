#include "model_image_splined_raster.h"
#include "model_image_upsample.h"
#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "utils.h"
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

  double uas2rad = 1e-6/3600. * M_PI/180.;

  double xmin=-50.0*uas2rad, xmax=50.0*uas2rad;
  double ymin=-50.0*uas2rad, ymax=50.0*uas2rad;
  size_t Nx=8, Ny=8;

  std::vector<double> means(Nx*Ny);
  // Starting point
  means[0] = 39.670175;
  means[1] = 40.64066;
  means[2] = 40.403132;
  means[3] = 37.964463;
  means[4] = 35.54215;
  means[5] = 39.985084;
  means[6] = 40.415874;
  means[7] = 35.069795;
  means[8] = 32.44248;
  means[9] = 41.996082;
  means[10] = 42.224759;
  means[11] = 43.531714;
  means[12] = 43.278534;
  means[13] = 42.64759;
  means[14] = 27.278992;
  means[15] = 36.660057;
  means[16] = 40.719575;
  means[17] = 38.362114;
  means[18] = 43.490529;
  means[19] = 39.301244;
  means[20] = 41.06569;
  means[21] = 43.586301;
  means[22] = 41.635805;
  means[23] = 39.033091;
  means[24] = 41.724684;
  means[25] = 41.61926;
  means[26] = 43.945397;
  means[27] = 43.524013;
  means[28] = 42.988082;
  means[29] = 44.061067;
  means[30] = 41.210049;
  means[31] = 36.88276;
  means[32] = 41.34857;
  means[33] = 39.854566;
  means[34] = 43.043505;
  means[35] = 43.903195;
  means[36] = 43.794025;
  means[37] = 42.977778;
  means[38] = 42.006194;
  means[39] = 40.606372;
  means[40] = 42.539072;
  means[41] = 41.006083;
  means[42] = 41.281379;
  means[43] = 40.059967;
  means[44] = 39.930256;
  means[45] = 39.749546;
  means[46] = 39.862371;
  means[47] = 38.554712;
  means[48] = 40.155517;
  means[49] = 40.48723;
  means[50] = 41.241379;
  means[51] = 40.441946;
  means[52] = 40.55777;
  means[53] = 35.827981;
  means[54] = 41.680467;
  means[55] = 36.877867;
  means[56] = 41.633227;
  means[57] = 36.379137;
  means[58] = 38.598873;
  means[59] = 41.383568;
  means[60] = 41.744979;
  means[61] = 39.618519;
  means[62] = 32.919115;
  means[63] = 34.066225;


  Themis::model_image_splined_raster image(xmin,xmax,Nx,ymin,ymax,Ny);

    
  std::vector< std::vector<double> > a,b,I;

  image.generate_model(means);
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


  image.generate_complex_visibilities();


  std::vector< std::vector<double> > u,v,V;
  image.get_visibility_amplitudes(u,v,V);

  for (size_t i=0; i<u.size(); ++i)
  {
    for (size_t j=0; j<u[i].size(); ++j)
    {
      Themis::datum_visibility_amplitude dtmp(u[i][j],v[i][j],0,0);
      double Vs=image.visibility_amplitude(dtmp,0);
      std::cout << std::setw(15) << u[i][j]
		<< std::setw(15) << v[i][j]
		<< std::setw(15) << V[i][j]
		<< std::setw(15) << Vs
		<< std::setw(15) << Vs/V[i][j]
		<< std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


  std::vector< std::vector< std::complex<double> > > Vc;
  image.get_visibilities(u,v,Vc);

  for (size_t i=0; i<u.size(); ++i)
  {
    for (size_t j=0; j<u[i].size(); ++j)
    {
      Themis::datum_visibility dtmp(u[i][j],v[i][j],0,0);
      std::complex<double> Vs=image.visibility(dtmp,0);
      std::cout << std::setw(15) << u[i][j]
		<< std::setw(15) << v[i][j]
		<< std::setw(15) << Vc[i][j].real()
		<< std::setw(15) << Vc[i][j].imag()
		<< std::setw(15) << Vs.real()
		<< std::setw(15) << Vs.imag()
		<< std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}

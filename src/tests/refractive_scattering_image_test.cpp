/*!
  \file tests/parameterized_scattering_model_test.cpp
  \author Paul Tiede
  \date May 2019
  \test Test of model_image_refractive_scattering image generation.
  \details Creates an ensemble blurred, and refractively scattered image of a symmetric gaussian.
*/
#include "data_visibility.h"
#include "model_image_sum.h"
#include "model_image_gaussian.h"
#include "model_image_crescent.h"
#include "model_image_refractive_scattering.h"
#include "utils.h"
#include "random_number_generator.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <string>

#include <mpi.h>

int main(int argc, char* argv[])
{

  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  size_t nModes = 32;
  //Now create the image
  Themis::model_image_gaussian g1, g2, g1_num, g2_num;
  g1_num.use_numerical_visibilities();
  g2_num.use_numerical_visibilities();
  std::vector<Themis::model_image*> imlist, imlist_num;
  imlist.push_back(&g1);
  imlist.push_back(&g2);
  
  imlist_num.push_back(&g1_num);
  imlist_num.push_back(&g2_num);

  Themis::model_image_sum im(imlist), im_num(imlist_num);


  double uas2rad = 1e-6/3600.0*M_PI/180.0;
  //Create a 128x128 refractive screen
  Themis::model_image_refractive_scattering scattered_image_num(im_num, nModes, 0);
  Themis::model_image_refractive_scattering scattered_image_an(im, nModes, 0);
  scattered_image_num.set_image_resolution(128);
  scattered_image_an.set_image_resolution(128);
  scattered_image_num.set_screen_size(150.0*uas2rad);
  scattered_image_an.set_screen_size(150.0*uas2rad);
  
  
  //Set parameters
  size_t nsrc = im_num.size();
  std::vector<double> parameters(nsrc+nModes*nModes-1,0.0);
  //Gaussian params
  parameters[0] = 1.0;
  parameters[1] = 5.0*uas2rad;
  parameters[2] = 5.0*uas2rad;
  parameters[3] = 0.0;
  parameters[4] = -10.0*uas2rad;
  parameters[5] = -10.0*uas2rad;
  //Gaussian 2
  parameters[6] = 1.0;
  parameters[7] = 5.0*uas2rad;
  parameters[8] = 5.0*uas2rad;
  parameters[9] = 0.0;
  parameters[10] = 15.0*uas2rad;
  parameters[11] = 5.0*uas2rad;
  
  
  //Now fill the screen
  Themis::Ran2RNG rng(42);
  std::ofstream screen_out("2gauss_screen_parameters_test.dat");
  screen_out << "# nModes="<< nModes << std::endl;
  //parameters[nsrc] = 25;
  for ( size_t i = 0; i < nModes*nModes-1; i++ ){
    //parameters[nsrc+i] = rng.rand();
    screen_out << parameters[i+nsrc] << std::endl;
  }

  scattered_image_num.generate_model(parameters);
  scattered_image_an.generate_model(parameters);
  scattered_image_num.generate_complex_visibilities();
  scattered_image_an.generate_complex_visibilities();


  //Now lets get the source image first and output it
  std::vector<std::vector<double> > alpha,beta,Iea_num,Ia_num,Iea_an,Ia_an;
  //scattered_image.get_unscattered_image(alpha,beta,Isrc);
  //std::cout << "Isrc size " << Isrc.size() << std::endl;
  scattered_image_num.get_ensemble_average_image(alpha,beta,Iea_num);
  scattered_image_num.get_image(alpha,beta,Ia_num);
  scattered_image_an.get_ensemble_average_image(alpha,beta,Iea_an);
  scattered_image_an.get_image(alpha,beta,Ia_an);
  
  std::ofstream src_out("2gauss_images_test.dat"); 
  double dxdy = (alpha[1][1]-alpha[0][0])*(beta[1][1]-beta[0][0]);
  for ( size_t j = 0; j < alpha[0].size(); j++)
    for ( size_t i = 0; i < alpha.size(); i++)
    {
      src_out << std::setw(15) << alpha[alpha.size()-1-i][j]
              << std::setw(15) << beta[i][j]
              //<< std::setw(15) << Isrc[i][j]*dxdy
              << std::setw(15) << Iea_num[i][j]*dxdy
              << std::setw(15) << Ia_num[i][j]*dxdy
              << std::setw(15) << Iea_an[i][j]*dxdy
              << std::setw(15) << Ia_an[i][j]*dxdy << std::endl;
    }
  src_out.close();
  
  std::ofstream vis_out("2gauss_visibilities_test.dat"); 
  std::vector<std::vector<double> > u,v;
  std::vector<std::vector<std::complex<double> > > V_num, V_an;
  
  scattered_image_num.get_visibilities(u,v,V_num);
  scattered_image_an.get_visibilities(u,v,V_an);
  for ( size_t j = 0; j < u[0].size(); j++)
    for ( size_t i = 0; i < u.size(); i++)
    {
      
      Themis::datum_visibility tmp(u[i][j],v[i][j],std::complex<double>(0.0,0.0),std::complex<double>(0.0,0.0),230e9,0.0,"","","");
      std::complex<double> Vsrc = im.visibility(tmp,0), Vsrc_num = im_num.visibility(tmp,0);
      vis_out << std::setw(15) << u[i][j]
              << std::setw(15) << v[i][j]
              << std::setw(15) << std::abs(V_num[i][j])
              << std::setw(15) << V_num[i][j].real()
              << std::setw(15) << V_num[i][j].imag()
              << std::setw(15) << std::abs(V_an[i][j])
              << std::setw(15) << V_an[i][j].real()
              << std::setw(15) << V_an[i][j].imag()
              << std::setw(15) << std::abs(Vsrc_num)
              << std::setw(15) << Vsrc_num.real()
              << std::setw(15) << Vsrc_num.imag()
              << std::setw(15) << std::abs(Vsrc)
              << std::setw(15) << Vsrc.real()
              << std::setw(15) << Vsrc.imag() << std::endl;
    }
  vis_out.close();

  MPI_Finalize();

  return 0;
  

}

/*!
  \file model_image_astroray.cpp
  \author Roman Gold
  \date  June, 2017
  \brief Implements driver to ASTRORAY executable
  \details Run the ASTRORAY code externally and read the image as an image_model_object
*/

#include "model_image_astroray.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <mpi.h>

namespace Themis {

model_image_astroray::model_image_astroray()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "Creating model_image_astroray in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_astroray in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}

void model_image_astroray::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  // See code snippet in [model_image_astroray_tmp.cpp] taken from ASTRORAY [see imaging.cpp] 

  // read from binary input file (produced externally by ASTRORAY
  // std::ifstream astroray_image_input_file (("astroray_image.dat").c_str(), std::ios::in|std::ios::binary);
  std::ifstream astroray_image_input_file ("astroray_image.dat", std::ios::in|std::ios::binary);
  
  typedef double (*para)[20];          // array of parameters
  para params = (para) new double[20];

  // read parameters from header - fixed size variable
  astroray_image_input_file.read(reinterpret_cast<char *>(params), 20*sizeof(double));
  
  const int nxy = 100; // parameters[2]; // nr of pixels per dimension

  typedef double (*array_astroray)[nxy+1][nxy+1][5];                       //array of intensities 
  array_astroray IQUV_astroray = (array_astroray) new double[nxy+1][nxy+1][5];
  
  astroray_image_input_file.read(reinterpret_cast<char *>(IQUV_astroray), 5*(nxy)*(nxy)*sizeof(double));  // read image data 4 Stokes parameters + 1 image diagnostic

  astroray_image_input_file.close();

  _rhonor = parameters[0];
  _th = parameters[1];
  _Te_jet_par = parameters[2];
  _dphi = parameters[3];


  /*********************
   * LAUNCH ASTRORAY : */

  std::stringstream launch_cmd;
  launch_cmd<<"export LSB_JOBINDEX=-1;time ../ASTRORAY_main"; // First part, Set "case" environment variable, call executable
  launch_cmd<<" 1 1 1 4"; // standard ASTRORAY command line arguments
  launch_cmd<<" "<<_rhonor<<" "<<_th<<" "<<_dphi<<" "<<_Te_jet_par; // command line arguments to ASTRORAY
  std::cout<<"LAUNCHING: "<<launch_cmd.str()<<std::endl;
  
  int launch_cmd_return = system(launch_cmd.str().c_str());
  if ( launch_cmd_return==-1 ) {
    std::cout << "[model_image_astroray.cpp]: ASTRORAY launch command returned -1 (FAILED)" << std::endl;
    std::cout << "BETTER DIE..." << std::endl;
    exit(-1);
  }

  std::string path2astroray="~/codes/astroray/";
  std::string copy_command="cp ~/codes/astroray/thermal/";

  std::stringstream ASTRORAY_OUTPUT_FILENAME;
  ASTRORAY_OUTPUT_FILENAME<<"shotimag93.75th"<<int(_th*100)<<"f230fn5500case-1_100.dat";
  // copy_command.append("/shotimag93.75th173f230fn5500case-1_100.dat astroray_image.dat");
  // copy_command.append("/shotimag93.75th104f230fn5500case-1_100.dat astroray_image.dat");
  copy_command.append(ASTRORAY_OUTPUT_FILENAME.str().c_str());
  std::stringstream ASTRORAY_THEMIS_FILENAME;

  
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  ASTRORAY_THEMIS_FILENAME << " astroray_image_rank" << world_rank << ".dat";
  // ASTRORAY_THEMIS_FILENAME << " astroray_image_rank" << MPI::COMM_WORLD.Get_rank() << ".dat";
  // ASTRORAY_OUTPUTFILENAME.append(MPI_RANK.str() );
  // ASTRORAY_OUTPUTFILENAME.append(".dat");
  copy_command.append(ASTRORAY_THEMIS_FILENAME.str());
  int copy_command_return = system(copy_command.c_str());
  // system("cp " << path2astroray.c_str() << "/poliresa93.75th173fn5500hi.dat astroray_sed.dat");

  if ( copy_command_return==-1 ) {
    std::cout << "[model_image_astroray.cpp]: Copy command returned -1 (FAILED)" << std::endl;
    std::cout << "BETTER DIE..." << std::endl;
    exit(-1);
  }



  /*
  std::cout << "model_image_astroray::generate_image : "
      	    << std::setw(4) << world_rank
      	    << std::setw(15) << _Itotal
      	    << std::setw(15) << _sigma_alpha
      	    << std::setw(15) << _sigma_beta
      	    << std::endl;
  */

  // double Inorm = _Itotal/(_sigma_alpha*_sigma_beta*2*M_PI);

  // Allocate if necessary
  if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(nxy))
  {
    alpha.resize(nxy);
    beta.resize(nxy);
    I.resize(nxy);
  }
  
  for (size_t j=0; j<alpha.size(); j++)
  {
    if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(nxy))
    {
      alpha[j].resize(nxy,0.0);
      beta[j].resize(nxy,0.0);
      I[j].resize(nxy,0.0);
    }
  }

  // Fill array with new image
  for (size_t j=0; j<alpha.size(); j++)
  {
    for (size_t k=0; k<alpha[j].size(); k++)
    {
      // 2DO: Set correctly consistent with ASTRORAY
      alpha[j][k] = ((double(j)-0.5*double(nxy)+0.5)* 15. *2.0/double(nxy));
      beta[j][k] = ((double(k)-0.5*double(nxy)+0.5)* 15. *2.0/double(nxy));

      I[j][k] = (*IQUV_astroray)[j][k][0]; // I-only for now

    }
  }

} // end of generate_image()


double model_image_astroray::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
    return ( model_image::visibility_amplitude(d,acc) );
} // end of visibility_amplitude()


};

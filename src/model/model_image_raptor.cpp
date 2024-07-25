/*!
  \file model_image_raptor.cpp
  \author Roman Gold, Jordy Davelaar, Thomas Bronzwaer
  \date  Dec, 2017
  \brief Implements driver to RAPTOR executable
  \details Run the RAPTOR code externally and read the image as an image_model_object
*/

#include "model_image_raptor.h"

#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <iomanip>
#include <mpi.h>

namespace Themis {

model_image_raptor::model_image_raptor()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "Creating model_image_raptor in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_raptor in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}

void model_image_raptor::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  // read from ASCII input file (produced externally by RAPTOR)
 
  static int onlyonetime = 1;

  const int nxy = 64; // parameters[2]; // nr of pixels per dimension

  //"usage: raptor theta freq filename Munit trat_j trat_d \n"
  _th = parameters[0];
  _Munit = parameters[1];
  _trat_j = parameters[2];
  _trat_d = parameters[3];
  _position_angle = parameters[4];

  /*********************
   * LAUNCH RAPTOR : */

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::stringstream launch_cmd;
  launch_cmd<<"./raptor_rank"<<world_rank;

  if (onlyonetime) { // MAKE ONE EXECUTABLE FOR EACH RANK BUT ONLY ONCE
    onlyonetime = 0; // only once
    std::stringstream exec_copy_stream;
    exec_copy_stream << "cp raptor " << launch_cmd.str();
    std::string exec_copy_string = exec_copy_stream.str();
    int status=system(exec_copy_string.c_str());
    if (status) {
      std::cout<<"copy of executable failed"<<std::endl;
      exit(1);
    }
  }

  std::stringstream RAPTOR_THEMIS_FILENAME;
  RAPTOR_THEMIS_FILENAME << "raptor_image_rank" << world_rank << ".dat";

  // LAUNCH RAPTOR
  launch_cmd<<" model.in /data/BHAC3D/data2500.blk "<< RAPTOR_THEMIS_FILENAME.str()<<" "<<_Munit<<" "<< _th<<" "<< _trat_j<<" "<< _trat_d << " 1>/dev/null 2>/dev/null"; // command line arguments to RAPTOR
  std::cout<<"LAUNCHING: "<<launch_cmd.str()<<std::endl;
  
  int launch_cmd_return = system(launch_cmd.str().c_str());
  if ( launch_cmd_return==-1 ) {
    std::cout << "[model_image_raptor.cpp]: RAPTOR launch command returned -1 (FAILED)" << std::endl;
    std::cout << "BETTER DIE..." << std::endl;
    exit(-1);
  }



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

  std::ifstream RAPTOR_THEMIS_FILE;
  std::string RAPTOR_THEMIS_FILENAME_STR=RAPTOR_THEMIS_FILENAME.str();
  RAPTOR_THEMIS_FILE.open(RAPTOR_THEMIS_FILENAME.str().c_str());

  if(RAPTOR_THEMIS_FILE.is_open())
     {
      std::string line;
      int a=0, b=0;
      while(std::getline(RAPTOR_THEMIS_FILE, line))
	{
	  std::stringstream strs(line);
	  double dummy;
	  strs  >> dummy;
	  I[a][b]=dummy;
	  a++;
	  if(a%nxy==0){
		b++;
		a=0;
	  }
	}

    }
  RAPTOR_THEMIS_FILE.close();

  // Fill array with new image
  for (size_t j=0; j<alpha.size(); j++)
  {
    for (size_t k=0; k<alpha[j].size(); k++)
    {
      // 2DO: Set correctly consistent with RAPTOR
      alpha[j][k] = ((double(j)-0.5*double(nxy)+0.5)* 40./double(nxy));
      beta[j][k] = ((double(k)-0.5*double(nxy)+0.5)* 40./double(nxy));
      // I[j][k] = (*IQUV_raptor)[j][k][2]; // I-only for now

    }
  }

} // end of generate_image()


double model_image_raptor::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
    return ( model_image::visibility_amplitude(d,acc) );
} // end of visibility_amplitude()


};

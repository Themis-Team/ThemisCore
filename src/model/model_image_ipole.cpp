/*!
  \file model_image_ipole.cpp
  \author Roman Gold, Monika Moscibrodzka, Shan-Shan Zhao
  \date  Nov, 2018
  \brief Implements driver to IPOLE executable
  \details Run the IPOLE code externally and read the image as an image_model_object
*/

#include "model_image_ipole.h"

#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <iomanip>
#include <mpi.h>


namespace Themis {

model_image_ipole::model_image_ipole()
{
 // int world_rank;
 // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
 // std::cout << "Creating model_image_ipole in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_ipole in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}

//image resolution setting
void model_image_ipole::set_image_resolution(int Nray)
{
  _Nray = Nray;
    
}

//field of view setting
void model_image_ipole::set_FOV(double FOV)
{
  _FOVx = FOV;
  _FOVy = FOV; 
}

void model_image_ipole::set_fit_one_param(bool fit_onepar, int param_num)
{   
   _fit_onepar=fit_onepar;
   _param_num=param_num;
}

void model_image_ipole::image_file_name(std::string im_filename)
{
   _im_filename=im_filename;
}

void model_image_ipole::set_tag(std::string tag)
{
   _tag=tag;
}


void model_image_ipole::set_Rlow(double Rlow)
{
  _trat_j=Rlow;
}

void model_image_ipole::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  // read from ASCII input file (produced externally by IPOLE)
 
  static int onlyonetime = 1;
  int nxy=_Nray;
  std::stringstream IPOLE_THEMIS_FILENAME;
  bool IF_LAUNCH_IPOLE=true;

  //fix value
  _th = 60.;
  _Munit = 3.e18;
  _trat_d=3.;
  if (_trat_j<0.1||_trat_j>100){
     std::cout<<"ERR: R_low is out of range (0.1,100), R_low="<<_trat_j<<std::endl;
     std::cout<<"Must use: model_image_ipole::set_Rlow(double Rlow). "<<std::endl;
     exit(1);    
  }
  
  if (_fit_onepar)
  {
     if (_param_num==1){
        _th = parameters[0];}
     if (_param_num==2){
        _Munit = parameters[0];}
     if (_param_num==3){
        _trat_d = parameters[0];}
     if (_param_num==4){
        if (_im_filename.size()>0){
            IPOLE_THEMIS_FILENAME<<_im_filename;
            IF_LAUNCH_IPOLE=false;}
        else{
            std::cout<<"ERR: NO ipole image filename is declared. "<<std::endl;
            std::cout<<"Must use: model_image_ipole::image_file_name(std::string im_filename). "<<std::endl;
            exit(1);}}
  }
  else
  {
      _th = parameters[0];
      _Munit = parameters[1];
      _trat_d = parameters[2];
   }

  if (IF_LAUNCH_IPOLE){

    /*********************
     * LAUNCH IPOLE : */

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::stringstream launch_cmd,dump_filename,dump_filename_mpi;
    // ./ipole // Monika's ipole
    launch_cmd<<"time ./ipole_"<<_tag<<"_rank"<<world_rank;
    dump_filename<<"dump_00001500.h5";
    dump_filename_mpi<<"dump_00001500"<<world_rank<<".h5";
    // new git ipole version from afd-il repo
    // ./ipole -par roman.par --nx=32 --ny=32 --fovx=100 --fovy=100
    // launch_cmd<<"./ipole_"<<_tag<<"_rank"<<world_rank<<" -par roman.par --nx=32 --ny=32 --fovx=100 --fovy=100";

    if (onlyonetime) { // MAKE ONE EXECUTABLE FOR EACH RANK BUT ONLY ONCE
    // if (false) { // MAKE ONE EXECUTABLE FOR EACH RANK BUT ONLY ONCE

      onlyonetime = 0; // only once
      std::stringstream exec_copy_stream;
      exec_copy_stream << "cp ipole ./ipole_"<<_tag<<"_rank"<<world_rank;
      std::string exec_copy_string = exec_copy_stream.str();
      int status=system(exec_copy_string.c_str());
      std::stringstream dump_copy_stream;
      dump_copy_stream<<"cp "<<dump_filename.str()<<" "<<dump_filename_mpi.str();
      std::string dump_copy_string = dump_copy_stream.str();
      int status_dump=system(dump_copy_string.c_str());
      // std::cout<<exec_copy_string.c_str()<<" system "<<status<<std::endl;
      if (status+status_dump) {
          std::cout<<"copy of executable or dump failed "<<exec_copy_string.c_str()<<" system "<<status<<std::endl;
//        exit(1);
          std::cout<<"WARN: copy of executable failed..rank: "<<world_rank<<std::endl;
       }
     }

  
    IPOLE_THEMIS_FILENAME << "ipole_image_"<<_tag<<"_rank" << world_rank << ".dat";

    // LAUNCH IPOLE
    // modified Monika's ipole version
    // launch_cmd<<" "<<_th<<" "<< 230e9 <<" "<<"HARM3D.001500.h5"<<" "<<_Munit<<" "<<_trat_j<<" "<< _trat_d << " "<< IPOLE_THEMIS_FILENAME.str() << " 1>/dev/null 2>/dev/null"; // command line arguments to IPOLE

    // _FOVx _FOVy are in rads ipole's fovx/fovy are in muas
    double fovx=_FOVx / M_PI*180. *3600.*1e6;
    double fovy=_FOVy / M_PI*180. *3600.*1e6;

    // use distance? dsource in ipole is in pc 16.9Mpc for M87
    // also note that ipole sets MBH 6.2e9
    double MBH=6.2e9;    // HARDWIRED
    // double sigma_cut=1.; // HARDWIRED (can't be varied as parameter in ipole)

    // drive IPOLE via command line arguments
    launch_cmd<<" -par roman.par "                  // par file with defaults
	      <<" --unpol "                         // unpolarized
              <<" --nx="<<nxy<<" --ny="<<nxy        // nr of pixels
              <<" --fovx="<<fovx<<" --fovy="<<fovy  // field of view
	      <<" --MBH="<<MBH                      // BH mass (fixed for now)
              <<" --thetacam="<<_th                 // inclination
	      <<" --dump="<<dump_filename_mpi.str() // GRMHD snapshot
	      <<" --M_unit="<<_Munit                // Accretion rate (unit)
              // trat_j=R_low trat_d=R_high => trat_small=R_low? trat_large=R_high?
	      <<" --trat_small="<<_trat_j           // R_low
	      <<" --trat_large="<< _trat_d          // R_high
	      // <<" --sigmacut="<< sigma_cut          // magnetization cut on emissivity
	      << " --outfile="<< IPOLE_THEMIS_FILENAME.str();  
              // <<" 1>/dev/null 2>/dev/null"; // suppress ipole's stdout/stderr 

    std::cout<<"LAUNCHING: "<<launch_cmd.str()<<"..."<<std::endl;

    int launch_cmd_return = system(launch_cmd.str().c_str());
    if ( launch_cmd_return==1 ) {
      std::cout << "[model_image_ipole.cpp]: IPOLE launch command returned 1 (FAILED)" << std::endl;
      std::cout << "BETTER DIE..." << std::endl;
      exit(1);
    }

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

  // system("h5dump -d unpol -o image.txt -y image.h5"); // Use h5dump to convert hdf5 to txt file
  // system("h5dump -d unpol -o image.txt -y "<<IPOLE_THEMIS_FILENAME.str().c_str());

  double FOVx=_FOVx;
  double FOVy=_FOVy;

  std::ifstream IPOLE_THEMIS_FILE;
  IPOLE_THEMIS_FILE.open(IPOLE_THEMIS_FILENAME.str().c_str());

  if(IPOLE_THEMIS_FILE.is_open())
     {
      std::string line;
      while(std::getline(IPOLE_THEMIS_FILE, line))
	{
	  std::stringstream strs(line);

	  // Monika's ipole
	  int a, b;
	  double dummy;
	  strs >> a >> b >> dummy;

          I[nxy-a-1][b]=dummy*double(nxy)*double(nxy)/FOVx/FOVy;
          //we flip the image from left to right and convert Jy/pix2 to Jy/rad2,
          //so that  ipole and Themis are consistent.
	  strs >> dummy >> dummy >> dummy >> dummy;
	}

    }
  IPOLE_THEMIS_FILE.close();

  // alpha and beta are in radians.
  for (size_t j=0; j<alpha.size(); j++)
  {
    for (size_t k=0; k<alpha[j].size(); k++)
    {
      alpha[j][k] = ((double(j)-0.5*double(nxy)+0.5)* FOVx/double(nxy));
      beta[j][k] = ((double(k)-0.5*double(nxy)+0.5)* FOVy/double(nxy));    
    }
  }

} // end of generate_image()



double model_image_ipole::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
    return ( model_image::visibility_amplitude(d,acc) );
}


};

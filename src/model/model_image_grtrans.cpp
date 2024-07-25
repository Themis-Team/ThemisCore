/*!
  \file model_image_grtrans.cpp
  \author Roman Gold
  \date  June, 2017
  \brief Implements driver to GRTRANS executable
  \details Run the GRTRANS code externally and read the image as an image_model_object
  \todo 
	   (1) READ-IN IMAGE DATA INTO THEMIS
           (2) CHECK EVERYTHING
  \warning NOT QUITE READY YET (see todo) 
*/

#include "model_image_grtrans.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mpi.h>

namespace Themis {



model_image_grtrans::model_image_grtrans()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "Creating model_image_grtrans in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_grtrans in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}



void model_image_grtrans::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{

  _mdot = parameters[0];
  _th = parameters[1];
  _muval = parameters[2];
  _spin = parameters[3];
  _phi0 = parameters[4];

  /* MODIFY PAR FILE inputs.in FOR GRTRANS */
  int status0 = system("cp inputs_fiducial.in inputs.in");

  // mdot
  int status1 = system( (std::string("sed -i \'s/ mdotmin=1.57e+15, / mdotmin=")+ dtos(_mdot) + std::string(", /g\' inputs.in") ).c_str() );
  int status2 = system( (std::string("sed -i \'s/ mdotmax=1.57e+15, / mdotmax=")+ dtos(_mdot) + std::string(", /g\' inputs.in") ).c_str() );
  // inclination
  int status3 = system( (std::string("sed -i \'s/ mumin=0.0, / mumin=")+ dtos(_th) + std::string(", /g\' inputs.in") ).c_str() );
  int status4 = system( (std::string("sed -i \'s/ mumax=0.0, / mumax=")+ dtos(_th) + std::string(", /g\' inputs.in") ).c_str() );
  // electron temperature
  int status5 = system( (std::string("sed -i \'s/ muval=0.25, / muval=")+ dtos(_muval) + std::string(", /g\' inputs.in") ).c_str() );
  // spin
  int status6 = system( (std::string("sed -i \'s/ spin=0.9375, / spin=")+ dtos(_spin) + std::string(", /g\' inputs.in") ).c_str() );
  // azimuthal viewing angle
  int status7 = system( (std::string("sed -i \'s/ phi0=-0.5, / phi0=")+ dtos(_phi0) + std::string(", /g\' inputs.in") ).c_str() );

  // do the same for other parameters ...

  if (status0+status1+status2+status3+status4+status5+status6+status7) {
    std::cout<<"[model_image_grtrans.cpp]: status of system() call to change grtrans parameters in parameter file input.in failed"<<std::endl;
  }


  /* MODIFY IN/OUT FILE files.in FOR GRTRANS */
  int status8 = system("cp files_fiducial.in files.in");
  if (status8!=0)
  {
    std::cout << "[model_image_grtrans.cpp]: copy fiducial filename configuration failed" << std::endl;
    exit(-1);
  }
  
  std::stringstream GRTRANS_OUTPUT_FILENAME;
  // MAKE IT THREAD SAFE: modify line in [file.in]:ofile="grtrans.out",

  
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  GRTRANS_OUTPUT_FILENAME<<"grtrans"<<"_rank"<<world_rank; // FITS format
  //GRTRANS_OUTPUT_FILENAME<<"grtrans"<<"_rank"<<MPI::COMM_WORLD.Get_rank()<<".out"; // FITS format
  

  int status9 = system((std::string("sed -i \'s/ ofile=\"grtrans.out\",/ ofile=\""+std::string(GRTRANS_OUTPUT_FILENAME.str())+"\",/g\' files.in")).c_str());
  if (status9!=0) {
    std::cout << "[model_image_grtrans.cpp]: substitution of filenames in configuration file failed" << std::endl;
    exit(-1);
  }




  /*********************
   * LAUNCH GRTRANS : */

  std::stringstream launch_cmd;
  // launch_cmd<<"time ../grtrans_ext/grtrans";
  launch_cmd<<"time grtrans";
  
  int launch_cmd_return = system(launch_cmd.str().c_str());
  if ( launch_cmd_return==-1 ) {
    std::cout << "[model_image_grtrans.cpp]: GRTRANS launch command returned -1 (FAILED)" << std::endl;
    std::cout << "BETTER DIE..." << std::endl;
    exit(-1);
  }



  // See code snippet in [model_image_grtrans_tmp.cpp] taken from GRTRANS [see imaging.cpp] 

  std::cout<<"READING IN "<<GRTRANS_OUTPUT_FILENAME.str()<<"..."<<std::endl;
  // read from binary input file (produced externally by GRTRANS)

  /*
  with open(self.ofile,'rb') as f:
    temp,nx,ny,nvals=np.fromfile(f,dtype='i4',count=4)
    temp=np.fromfile(f,dtype='i4',count=2)
    nkey=np.fromfile(f,dtype='i4',count=1)
    temp=np.fromfile(f,dtype='i4',count=2)
    keys=np.fromfile(f,dtype='f4',count=nkey)
    temp=np.fromfile(f,dtype='i4',count=2)
    field=np.fromfile(f,dtype='f4',count=2*nx*ny)
    ab=np.reshape(field,(nx*ny,2))
    temp=np.fromfile(f,dtype='i4',count=2)                          
    field=np.fromfile(f,dtype='f4',count=nvals*nx*ny)
    intens=np.reshape(field,(nx*ny,nvals))
                ivals=intens
                nu=keys[0]
    for x in f:
  temp,nx,ny,nvals=np.fromfile(x,dtype='i4',count=4)
	    temp=np.fromfile(x,dtype='i4',count=2)
	    nkey=np.fromfile(x,dtype='i4',count=1)
	    temp=np.fromfile(x,dtype='i4',count=2)
	    keys=np.fromfile(x,dtype='f4',count=nkey)
	    temp=np.fromfile(x,dtype='i4',count=2)
	    field=np.fromfile(x,dtype='f4',count=2*nx*ny)
	    ab=np.reshape(field,(nx*ny,2))
	    temp=np.fromfile(x,dtype='i4',count=2)
	    field=np.fromfile(x,dtype='f4',count=nvals*nx*ny)
	    intens=np.reshape(field,(nx*ny,nvals))
	    ivals=np.append(ivals,intens)
	    nu=nu.append(nu,keys[0])
  */


  // WHY does this not work?!
  // std::ifstream grtrans_image_input_file (GRTRANS_OUTPUT_FILENAME.str().c_str(), std::ios::in|std::ios::binary);

  std::fstream grtrans_image_input_file;
  grtrans_image_input_file.open (GRTRANS_OUTPUT_FILENAME.str().c_str(),  std::ios::binary | std::ios::out | std::ios::in);

  // const int nxy = 100;                                                      // nr of pixels per dimension
  // const int nvars=4;
  // typedef double (*array_grtrans)[nxy][nxy][nvars];                         // array of intensities 
  // array_grtrans IQUV_grtrans = (array_grtrans) new double[nxy][nxy][nvars];
  

  /************************
   * RG: THIS NEEDS WORK:
   *     COULD CALL PYTHON SCRIPT SHIPPED WITH GRTRANS AND OUTPUT ANOTHER FILE, BUT SEEMS BACKWARDS...
   * 
   // The binary output format is as follows:
   *
   // 3 integers: nx ny nvals
   // 1 integer: nkey
   // nkey floats: keyvals
   // 2*nx*ny floats: ab
   // nvals*nx*ny floats: ivals
   *
   *************************/

  // np.fromfile('grtrans_rank0.out',dtype=int32,count=10)
  //   Out[31]: 
  //   array([        12,        100,        100,          1,         12,
  // 		   4,          3,          4,          4, 1120403456], dtype=int32)


  int nx=0,ny=0,nvals=0,nkey=0,dummy=0;

  std::streampos begin_pos,end_pos; 

  // if (grtrans_image_input_file.is_open()) {

    grtrans_image_input_file.read((char *) &nkey,sizeof(int));
    grtrans_image_input_file.read((char *) &nx,sizeof(int));
    grtrans_image_input_file.read((char *) &ny,sizeof(int));
    grtrans_image_input_file.read((char *) &nvals,sizeof(int));

    grtrans_image_input_file.read((char *) &dummy,5*sizeof(int));

    

    // WHY DOES THIS NOT WORK (along with ifstream)? 
    // grtrans_image_input_file >> nx;
    // grtrans_image_input_file >> ny;
    // grtrans_image_input_file >> nvals;
    // grtrans_image_input_file >> nkey;

    std::cout<<"[model_image_grtrans.cpp]: "<<"nx="<<nx<<",ny="<<ny<<",nvals="<<nvals<<",nkey="<<nkey<<std::endl;
  
    // RG: PROBLEM: following ints need to be constant (except first) when used for array sizes... 
    const int nx_const = nx;
    const int ny_const = 100; // ny;
    const int nvals_const = 1; // nvals
    typedef double (*array_grtrans)[nx_const][ny_const][nvals];                         // array of intensities 
    array_grtrans keyvals_grtrans = (array_grtrans) new double[nkey];
    
    // array_grtrans ab_grtrans = (array_grtrans) new double[nx_const][ny_const][2];
    array_grtrans IQUV_grtrans = (array_grtrans) new double[nx_const][ny_const][nvals_const];
    
    // READ IN GRTRANS COORDINATES AND IMAGE DATA
    grtrans_image_input_file.read((char *) &keyvals_grtrans,nkey*sizeof(float)); // float/double?
    // std::cout<<"keyvals_grtrans="<<key_vals_grtrans<<std::endl;
    // SEG-faults
    // grtrans_image_input_file.read((char *) &ab_grtrans,2*nx_const*ny_const*sizeof(float)); // float/double?
    // grtrans_image_input_file.read((char *) &IQUV_grtrans,nvals_const*nx_const*ny_const*sizeof(float)); // float/double?
    //  }

  grtrans_image_input_file.close();



  // Allocate if necessary
  if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(nx_const))
  {
    alpha.resize(nx_const);
    beta.resize(ny_const);
    I.resize(nx_const);
  }
  for (size_t j=0; j<alpha.size(); j++)
  {
    if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(nx_const))
    {
      alpha[j].resize(ny_const,0.0);
      beta[j].resize(nx_const,0.0);
      I[j].resize(ny_const,0.0);
    }
  }

  // Fill array with new image
  for (size_t j=0; j<alpha.size(); j++)
  {
    for (size_t k=0; k<alpha[j].size(); k++)
    {
      alpha[j][k] = ((double(j)-0.5*double(nx_const)+0.5)* 15. *2.0/double(ny_const));
      beta[j][k] = ((double(k)-0.5*double(nx_const)+0.5)* 15. *2.0/double(ny_const));

      I[j][k] = (*IQUV_grtrans)[j][k][0]; // I-only for now

    }
  }

} // end of generate_image()


double model_image_grtrans::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
    return ( model_image::visibility_amplitude(d,acc) );
} // end of visibility_amplitude()


std::string model_image_grtrans::dtos(double var)
{
  std::stringstream iss;
  iss << var;
  return iss.str();
}

};



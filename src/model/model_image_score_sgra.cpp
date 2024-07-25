/*!
  \file model_image_score_sgra.cpp
  \author Hung-Yi Pu, Paul Tiede
  \date  Oct, 2021
  \brief Implements SED-fitted RIAF model class.
  \details To be added
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "model_image_score_sgra.h"
#include "constants.h"
#include "utils.h"
#include "stop_watch.h"


namespace Themis {

  model_image_score_sgra::model_image_score_sgra(std::string image_file_name, std::string README_file_name, bool reflect_image, int window_function)
    : _comm(MPI_COMM_WORLD), _image_file_name(image_file_name), _README_file_name(README_file_name), _reflect_image(reflect_image)
  {
    /////////////////////////
    // Get the image particulars (masses, fov, etc.)
    std::ifstream rin(README_file_name.c_str());
    if (!rin.is_open())
    {
      std::cerr << "model_image_score_sgra: Could not open " << README_file_name << "\n";
      std::exit(1);
    }
    
    std::string stmp;
    rin.ignore(4096,'\n'); // Name
    rin.ignore(4096,'\n'); // Cadence
    rin >> stmp >> stmp >> _frequency;
    rin.ignore(4096,'\n');
    // Get the fov's
    double fovx, fovy;
    rin >> stmp >> stmp >> fovx;
    rin.ignore(4096,'\n');
    rin >> stmp >> stmp >> fovy;
    rin.ignore(4096,'\n');
    // Get the dimensions
    size_t Nx, Ny;
    rin >> stmp >> stmp >> Nx;
    rin.ignore(4096,'\n');
    rin >> stmp >> stmp >> Ny;
    rin.ignore(4096,'\n');

    rin.ignore(4096,'\n'); // Spin

    // Get the mass in Msun
    double M,D;
    rin >> stmp >> stmp >> M;
    rin.ignore(4096,'\n');
    // Get the distance in pc
    rin >> stmp >> stmp >> D;
    rin.ignore(4096,'\n');

    // M/D in uas
    _MoD = Themis::constants::G * M * Themis::constants::Msun / ( D * Themis::constants::pc * Themis::constants::c*Themis::constants::c ) * 180./M_PI * 3600 * 1e6;
    rin.close();

    std::cerr << std::setw(15) << M << std::setw(15) << D << std::setw(15) << Nx << std::setw(15) << Ny << std::setw(15) << fovx << std::setw(15) << fovy
    	      << std::setw(15) << _MoD << std::endl;

    
    ////////////////////////////
    // Read in the image (ONLY READS IN INTENSITY FOR NOW)
    std::ifstream iin(image_file_name.c_str());
    if (!iin.is_open())
    {
      std::cerr << "model_image_score_sgra: Could not open " << image_file_name << "\n";
      std::exit(1);
    }
    
    _I.resize(Nx);
    _alpha.resize(Nx);
    _beta.resize(Nx);
    for (size_t i=0; i<Nx; i++)
    {
      _I[i].resize(Ny);
      _alpha[i].resize(Ny);
      _beta[i].resize(Ny);
    }
    std::string dtmp;
    size_t ix,iy;
    double ix2rad = fovx/(Nx) * 1e-6 / 3600 * M_PI/180.;
    double iy2rad = fovy/(Ny) * 1e-6 / 3600 * M_PI/180.;

    //std::ofstream misdebug("mis_debug.txt");

    for (size_t i=0; i<Nx; i++)
      for (size_t j=0; j<Ny; j++)
      {
	iin >> ix >> iy;
     	_alpha[ix][iy] = ix2rad * ( ix - 0.5*Nx );
        _beta[ix][iy] = iy2rad * ( iy - 0.5*Nx );
	if (reflect_image)
	  iin >> dtmp;
	else{
	  iin >> dtmp;
          _I[ix][iy] = std::stod(dtmp);
        }
	iin.ignore(4096,'\n');

	/*
	misdebug << std::setw(15) << ix
		 << std::setw(15) << iy
		 << std::setw(15) << dtmp
		 << std::endl;
	*/
      }


    // Window I to avoid potential problems with ringing
    // 0 is no windowing
    if (window_function==0)
    {
      std::cerr << "WARNING: Applying no window to the input image, may produce ringing from image mask.\n";
    }
    else if (window_function==1) // Cosine
    {
      std::cerr << "WARNING: Applying a cosine window to the input image.\n";
      for (size_t i=0; i<Nx; i++)
	for (size_t j=0; j<Ny; j++)
	  _I[i][j] *= std::sin(2*M_PI*double(i)/double(Nx-1))*std::sin(2*M_PI*double(j)/double(Ny-1));
    }
    else if (window_function==2) // Hann
    {
      std::cerr << "WARNING: Applying a Hann window to the input image.\n";
      for (size_t i=0; i<Nx; i++)
	for (size_t j=0; j<Ny; j++)
	  _I[i][j] *= std::pow(std::sin(M_PI*double(i)/double(Nx-1))*std::sin(M_PI*double(j)/double(Ny-1)),2);
    }
    else if (window_function==3) // Blackman
    {
      std::cerr << "WARNING: Applying a Blackman window to the input image.\n";
      double a0 = 7938./18608.;
      double a1 = 9240./18608.;
      double a2 = 1430./18608.;
      double x,y;
      for (size_t i=0; i<Nx; i++)
	for (size_t j=0; j<Ny; j++)
	{
	  x = 2*M_PI*double(i)/double(Nx-1);
	  y = 2*M_PI*double(j)/double(Ny-1);
	  _I[i][j] *= (a0 - a1*std::cos(x) + a2*std::cos(2*x)) * (a0 - a1*std::cos(y) + a2*std::cos(2*y));
	}
    }
    else if (window_function==-3) // Symmetric Blackman
    {
      std::cerr << "WARNING: Applying a cylindrically symmetric Blackman window to the input image.\n";
      double a0 = 7938./18608.;
      double a1 = 9240./18608.;
      double a2 = 1430./18608.;
      double fx,fy, r;
      for (size_t i=0; i<Nx; i++)
	for (size_t j=0; j<Ny; j++)
	{
	  fx = (double(i)-0.5*(Nx-1))/double(Nx-1);
	  fy = (double(j)-0.5*(Ny-1))/double(Ny-1);
	  r = std::sqrt(fx*fx+fy*fy);

	  _I[i][j] *= ( r>0.5 ? 0.0 :  (a0 - a1*std::cos(2.0*M_PI*r) + a2*std::cos(4.0*M_PI*r)) );
	}
    }
    else if (window_function==-10) // Radial Bump Function
    {
      std::cerr << "WARNING: Applying a cylindrically bump function window to the input image.\n";
      double fx,fy, r;
      for (size_t i=0; i<Nx; i++)
	for (size_t j=0; j<Ny; j++)
	{
	  fx = (double(i)-0.5*(Nx-1))/double(Nx-1) / 0.5;
	  fy = (double(j)-0.5*(Ny-1))/double(Ny-1) / 0.5;
	  r = std::sqrt(fx*fx+fy*fy);

	  _I[i][j] *= (r<1.0 ? std::exp( - 1.0/(1.0-r*r) ) : 0.0);
	}
    }
    else 
    {
      std::cerr << "ERROR: Unrecognized window option " << window_function << ".  Options are 0-3.\n";
      std::exit(1);
    }


    std::ofstream misdebug("mis_debug.txt");
    for (size_t i=0; i<Nx; i++)
      for (size_t j=0; j<Ny; j++)
	misdebug << std::setw(15) << i
		 << std::setw(15) << j
		 << std::setw(15) << _I[i][j]
		 << std::endl;



    // Renorm I to 1 Jy
    double Itotal=0.0;
    for (size_t i=0; i<Nx; i++)
      for (size_t j=0; j<Ny; j++)
	Itotal += _I[i][j];

    // WHY IS THIS NOT NEEDED?
    // Itotal *= (fovx/(Nx)) * (fovy/(Ny));

    std::cerr << "Itotal = " << Itotal << std::endl;
    
    double renormalization_factor = 1.0/( Itotal*ix2rad*iy2rad) ;
    //double renormalization_factor =
    //
    //1.0/( Itotal );
    for (size_t i=0; i<Nx; i++)
      for (size_t j=0; j<Ny; j++)
	_I[i][j] *= renormalization_factor;

    // Finished!
    iin.close();
    _generated_model = true;


    
    // Get tabulated set of visibilities
    model_image::generate_complex_visibilities();
  }

  std::string model_image_score_sgra::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_score_sgra " << _MoD << " " << _image_file_name << " " << _README_file_name << " " << _reflect_image;
    return tag.str();
  }
  
  void model_image_score_sgra::generate_model(std::vector<double> parameters)
  {
    _mod = parameters[0]/_MoD; // Rescale factor, parameters[1] is M/D in uas
    _position_angle = parameters[1]; // position angle in radians
  }

  std::complex<double> model_image_score_sgra::visibility(datum_visibility& d, double acc)
  {
    // Create a rescaled datum_visibility object
    datum_visibility dtmp(d.u*_mod,d.v*_mod,d.V,d.err,d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    return (model_image::visibility(dtmp,acc));
  }

  double model_image_score_sgra::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    //datum_visibility tmp(d.u,d.v,std::complex<double>(d.V,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility_amplitude tmp(d.u*_mod,d.v*_mod,d.V,d.err,d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);

    //return ( std::abs(visibility(tmp,acc)) );
    return ( model_image::visibility_amplitude(tmp,acc));
  }

  double model_image_score_sgra::closure_phase(datum_closure_phase& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station1,d.Source);
    std::complex<double> V123 = visibility(tmp1,acc)*visibility(tmp2,acc)*visibility(tmp3,acc);

    return ( std::imag(std::log(V123))*180.0/M_PI );
  }



  double model_image_score_sgra::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
    datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0.0,0.0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);

    double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
    return ( V1234 );
  }


  
  void model_image_score_sgra::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
  }
}

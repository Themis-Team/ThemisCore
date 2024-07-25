/*!
  \file model_image_vrt2_pmap.cpp
  \author Avery Broderick
  \date  July, 2017
  \brief Implements the asymmetric model_image_vrt2_pmap image class.
  \details To be added
*/

#include "model_image_vrt2_pmap.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace Themis {

  model_image_vrt2_pmap::model_image_vrt2_pmap(double M, double D)
    : _Mcm(M), _Dcm(D), _pmap_file_name_set(false)
  {
  }

  model_image_vrt2_pmap::model_image_vrt2_pmap(std::string pmap_file_name, double M, double D)
    : _Mcm(M), _Dcm(D), _pmap_file_name(pmap_file_name), _pmap_file_name_set(true)
  {
  }

  void model_image_vrt2_pmap::set_pmap_file(std::string pmap_file_name)
  {
    _pmap_file_name=pmap_file_name;
    _pmap_file_name_set=true;
  }

  std::string model_image_vrt2_pmap::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_vrt2_pmap " << _pmap_file_name << " " << _Mcm << " " << _Dcm;
    return tag.str();
  }

  void model_image_vrt2_pmap::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    if (!_pmap_file_name_set)
    {
      std::cerr << "model_image_vrt2_pmap::generate_image: A pmap file name must be set prior to generating image.\n";
      std::exit(1);
    }
  
    std::ifstream pmap(_pmap_file_name);
    if (!pmap.is_open())
    {
      std::cerr << "model_image_vrt2_pmap::generate_image: Could not open " << _pmap_file_name << '\n';
      std::exit(1);
    }

    std::string stmp;
    
    double xmin, xmax, ymin, ymax;
    size_t Nx, Ny;
    
    pmap >> stmp >> xmin >> xmax >> Nx;
    pmap >> stmp >> ymin >> ymax >> Ny;

    pmap.ignore(4096,'\n'); // End line
    pmap.ignore(4096,'\n'); // Kill header
    
    
    double alphamin = xmin * _Mcm/_Dcm;
    double alphamax = xmax * _Mcm/_Dcm;
    double betamin  = ymin * _Mcm/_Dcm;
    double betamax  = ymax * _Mcm/_Dcm;
    double dalpha = (alphamax-alphamin)/(int(Nx)-1);
    double dbeta  = (betamax-betamin)/(int(Ny)-1);
    
    // To go from Jy/px to Jy/steradian
    double intensity_renormalization = 1.0/(dalpha*dbeta);
    
    // Allocate if necessary
    if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(Nx))
    {
      alpha.resize(Nx);
      beta.resize(Nx);
      I.resize(Nx);
    }
    for (size_t j=0; j<alpha.size(); j++)
    {
      if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(Ny))
      {
	alpha[j].resize(Ny,0.0);
	beta[j].resize(Ny,0.0);
	I[j].resize(Ny,0.0);
      }
    }
    
    // Fill array with new image
    int ix, iy;
    double Ival;
    for (size_t j=0; j<alpha.size(); j++)
    {
      for (size_t k=0; k<alpha[j].size(); k++)
      {
	pmap >> ix  >> iy >> Ival; // Read through the intensity, add polarization later
	pmap.ignore(4096,'\n'); // Kill rest of the line

	alpha[ix][iy] = double(ix)*dalpha + alphamin;
	beta[ix][iy] = double(iy)*dbeta  + betamin;
	I[ix][iy] = intensity_renormalization * Ival;
      }
    }
  }
};

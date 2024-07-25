/*!
  \file model_image_raster.cpp
  \author Avery Broderick
  \date  October, 2017
  \brief Implements the model_image_raster image class.
  \details To be added
*/

#include "model_image_raster.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace Themis {

  model_image_raster::model_image_raster(double xmin, double xmax, size_t Nx, double ymin, double ymax, size_t Ny)
    : _xmin(xmin), _xmax(xmax), _Nx(Nx), _ymin(ymin), _ymax(ymax), _Ny(Ny), _size(_Nx*_Ny), _defined_raster_grid(false)
  {
  }

  std::string model_image_raster::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_raster " << _xmin << " " << _xmax << " " << _Nx << " " << _ymin << " " << _ymax << " " << _Ny;
    
    return tag.str();
  }

  void model_image_raster::generate_model(std::vector<double> parameters)
  {
    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      _current_parameters = parameters;
      
      // Generate the image using the user-supplied routine
      generate_image(parameters,_I,_alpha,_beta);
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }

  void model_image_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    // Allocate if necessary
    if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(_Nx))
    {
      alpha.resize(_Nx);
      beta.resize(_Nx);
      I.resize(_Nx);
      _defined_raster_grid=false;
      for (size_t j=0; j<alpha.size(); j++)
      {
	if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(_Ny))
        {
	  alpha[j].resize(_Ny,0.0);
	  beta[j].resize(_Ny,0.0);
	  I[j].resize(_Ny,0.0);
	}
      }
    }

    //if (!_defined_raster_grid)
    {
      double dx = (_xmax-_xmin)/(int(_Nx)-1);
      double dy = (_ymax-_ymin)/(int(_Ny)-1);
    
      // Fill array with new image
      for (size_t j=0; j<alpha.size(); j++)
      {
	for (size_t k=0; k<alpha[j].size(); k++)
        {
	  alpha[j][k] = double(j)*dx + _xmin;
	  beta[j][k] = double(k)*dy  + _ymin;
	}
      }
      //_defined_raster_grid = true;
    }

    
    // Fill array with new image
    size_t k=0;
    for (size_t j=0; j<_Ny; j++)
      for (size_t i=0; i<_Nx; i++)
	I[i][j] = std::exp(parameters[k++]);

    /*
    double xC=0.0, yC=0.0;
    size_t k=0;
    I[0][0] = I[_Nx-1][0] = 0.0;
    for (size_t i=1; i<_Nx-1; i++)
    {
      I[i][0] = std::exp(parameters[k++]);
      xC += alpha[i][0]*I[i][0];
      yC += beta[i][0]*I[i][0];
    }

    for (size_t j=1; j<_Ny; j++)
      for (size_t i=0; i<_Nx; i++)
      {
	I[i][j] = std::exp(parameters[k++]);
	xC += alpha[i][j]*I[i][j];
	yC += beta[i][j]*I[i][j];
      }

    double detab = alpha[0][0]*beta[_Nx-1][0] - alpha[_Nx-1][0]*beta[0][0];
    I[0][0] = -( beta[_Nx-1][0]*xC - alpha[_Nx-1][0]*yC )/detab;
    I[_Nx-1][0] = -( - beta[0][0]*xC + alpha[0][0]*yC )/detab;
    */

    /*
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank==0)
    {
      std::cout << "\n\n";
      for (size_t j=0; j<_Ny; j++)
	for (size_t i=0; i<_Nx; i++)
	  std::cout << std::setw(15) << alpha[i][j]
		    << std::setw(15) << beta[i][j]
		    << std::setw(15) << I[i][j]
		    << std::setw(15) << "Foo"
		    << std::endl;
      std::cout << "\n\n";
    }
    */
    
  }
};

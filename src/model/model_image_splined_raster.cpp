/*!
  \file model_image_splined_raster.cpp
  \author Avery Broderick
  \date  October, 2017
  \brief Implements the model_image_splined_raster image class.
  \details To be added
*/

#include "model_image_splined_raster.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace Themis {

  model_image_splined_raster::model_image_splined_raster(double xmin, double xmax, size_t Nx, double ymin, double ymax, size_t Ny, double a)
    : _xmin(xmin), _xmax(xmax), _Nx(Nx), _ymin(ymin), _ymax(ymax), _Ny(Ny), _size(_Nx*_Ny), _defined_raster_grid(false), _tpdx(2.*M_PI*(_xmax-_xmin)/(_Nx-1)), _tpdy(2.*M_PI*(_ymax-_ymin)/(_Ny-1)), _a(a), _use_analytical_visibilities(false)
  {
  }

  void model_image_splined_raster::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }

  void model_image_splined_raster::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = true;
  }

  std::string model_image_splined_raster::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_splined_raster " << _xmin << " " << _xmax << " " << _Nx << " " << _ymin << " " << _ymax << " " << _Ny << " " << _a;
    
    return tag.str();
  }

  void model_image_splined_raster::generate_model(std::vector<double> parameters)
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

  void model_image_splined_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
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
      
    if (_defined_raster_grid==false)
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
      _defined_raster_grid=true;
    }

    
    // Fill array with new image
    size_t k=0;
    for (size_t j=0; j<_Ny; j++)
      for (size_t i=0; i<_Nx; i++)
	I[i][j] = std::exp(parameters[k++]);

  }



  std::complex<double> model_image_splined_raster::visibility(datum_visibility& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      std::complex<double> V(0.0,0.0);
      for (size_t i=0; i<_Nx; ++i)
	for (size_t j=0; j<_Ny; ++j)
	  V += _I[i][j] * std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (d.u*_alpha[i][j]+d.v*_beta[i][j]) );
      return ( cubic_spline_kernel(d.u,d.v)*V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]) );
    }
    else
      return ( cubic_spline_kernel(d.u,d.v)*model_image::visibility(d, acc) );
  }

  double model_image_splined_raster::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp(d.u,d.v,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      return std::abs(visibility(tmp,acc));
    }
    else
      return ( cubic_spline_kernel(d.u,d.v)*model_image::visibility_amplitude(d, acc) );
  }

  double model_image_splined_raster::closure_phase(datum_closure_phase& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
      datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station1,d.Source);
      std::complex<double> V123 = visibility(tmp1,acc)*visibility(tmp2,acc)*visibility(tmp3,acc);
      
      return ( std::imag(std::log(V123))*180.0/M_PI );
    }
    else
      return ( model_image::closure_phase(d,acc) );
  }



  double model_image_splined_raster::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
      datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
      datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);
      
      double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
      return ( V1234 );
    }
    else
      return ( model_image::closure_amplitude(d,acc) );
  }


  double model_image_splined_raster::cubic_spline_kernel_1d(double k) const
  {
    if (std::fabs(k)<1e-2)
      //return 1.0 - (2.0*_a-1.0)*(k*k)/15.0 + (16.0*_a+1.0)*(k*k*k*k)/560.0;
      return 1.0 - (2.0*_a+1.0)*(k*k)/15.0 + (16.0*_a+1.0)*(k*k*k*k)/560.0;
    else
    {
      double sk=std::sin(k), ck=std::cos(k);
      double c2k=ck*ck-sk*sk;

      double G = -4.0*sk*(2.0*_a*ck+(4.0*_a+3.0))/(k*k*k) + 12.0*(_a*(1.0-c2k)+2.0*(1.0-ck))/(k*k*k*k);

      return G;
    }
  }

  double model_image_splined_raster::cubic_spline_kernel(double u, double v) const
  {
    return cubic_spline_kernel_1d(u*_tpdx)*cubic_spline_kernel_1d(v*_tpdy);
  }

};

/*!
  \file model_polarized_image_adaptive_splined_raster.cpp
  \author Avery Broderick
  \date  March, 2020
  \brief Implements the model_polarized_image_splined_raster image class.
  \details To be added
*/

#include "model_polarized_image_adaptive_splined_raster.h"
#include "utils.h"
#include <iostream>
#include <iomanip>

namespace Themis {

  model_polarized_image_adaptive_splined_raster::model_polarized_image_adaptive_splined_raster(size_t Nx, size_t Ny, double a)
    : _Nx(Nx), _Ny(Ny), _size(4*_Nx*_Ny+3), _defined_raster_grid(false), _a(a), _use_analytical_visibilities(true)
  {
  }

  void model_polarized_image_adaptive_splined_raster::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }

  void model_polarized_image_adaptive_splined_raster::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = true;
  }

  void model_polarized_image_adaptive_splined_raster::use_exact_exp()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using std:exp in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = false;
  }

  void model_polarized_image_adaptive_splined_raster::use_fast_exp_approx()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using fast exponential approximation in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = true;
  }

  void model_polarized_image_adaptive_splined_raster::generate_model(std::vector<double> parameters)
  {
    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      _current_parameters = parameters;

      // Set the fov
      _xmin = -0.5*parameters[_size-3];
      _xmax =  0.5*parameters[_size-3];
      _ymin = -0.5*parameters[_size-2];
      _ymax =  0.5*parameters[_size-2];
      _tpdx = 2.*M_PI*(_xmax-_xmin)/(_Nx-1);
      _tpdy = 2.*M_PI*(_ymax-_ymin)/(_Ny-1); 
      _cpa = std::cos(parameters[_size-1]);
      _spa = std::sin(parameters[_size-1]);

      
      // Generate the image using the user-supplied routine
      generate_polarized_image(parameters,_I,_Q,_U,_V,_alpha,_beta);
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }

  void model_polarized_image_adaptive_splined_raster::generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    // Allocate if necessary
   if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(_Nx))
    {
      alpha.resize(_Nx);
      beta.resize(_Nx);
      I.resize(_Nx);
      Q.resize(_Nx);
      U.resize(_Nx);
      V.resize(_Nx);
      _defined_raster_grid=false;
      for (size_t j=0; j<alpha.size(); j++)
      {
	if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(_Ny))
        {
	  alpha[j].resize(_Ny,0.0);
	  beta[j].resize(_Ny,0.0);
	  I[j].resize(_Ny,0.0);
	  Q[j].resize(_Ny,0.0);
	  U[j].resize(_Ny,0.0);
	  V[j].resize(_Ny,0.0);
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
      //_defined_raster_grid=true;
      _defined_raster_grid=false;
    }

    
    // Fill array with new image
    size_t k=0, Npx=_Nx*_Ny;
    double m,EVPA,muV;
    for (size_t j=0; j<_Ny; j++)
      for (size_t i=0; i<_Nx; i++)
      {
	I[i][j] = std::exp(parameters[k]);
	m = std::exp(parameters[k + Npx]);
	EVPA = parameters[k+2*Npx];
	muV = parameters[k+3*Npx];

	Q[i][j] = m*std::cos(2.0*EVPA)*std::sqrt(1.0-muV*muV) * I[i][j];
	U[i][j] = m*std::sin(2.0*EVPA)*std::sqrt(1.0-muV*muV) * I[i][j];
	V[i][j] = m*muV * I[i][j];

	k+=1;
      }
  }



  void model_polarized_image_adaptive_splined_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
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
      //_defined_raster_grid=true;
      _defined_raster_grid=false;
    }

    
    // Fill array with new image
    size_t k=0;
    for (size_t j=0; j<_Ny; j++)
      for (size_t i=0; i<_Nx; i++)
	I[i][j] = std::exp(parameters[k++]);

  }

  std::string model_polarized_image_adaptive_splined_raster::model_tag() const
  {
    std::stringstream tag;
    tag << "model_polarized_image_adaptive_splined_raster " << _Nx << " " << _Ny << " " << _a
	<< " " <<  _modeling_Dterms;
    if (_modeling_Dterms)
      for (size_t j=0; j<_station_codes.size(); ++j)
	tag << " " << _station_codes[j];
    
    return tag.str();
  }
  

  std::vector< std::complex<double> > model_polarized_image_adaptive_splined_raster::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
  {
    if (_use_analytical_visibilities)
    {
      // Counter-rotate point
      double ur =  _cpa*d.u + _spa*d.v;
      double vr = -_spa*d.u + _cpa*d.v;
    
      std::complex<double> VI(0.0,0.0), VQ(0.0,0.0), VU(0.0,0.0), VV(0.0,0.0), phase_factor;

      if (_use_fast_exp_approx)
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	  {
	    phase_factor = utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) );
	    VI += _I[i][j] * phase_factor;
	    VQ += _Q[i][j] * phase_factor;
	    VU += _U[i][j] * phase_factor;
	    VV += _V[i][j] * phase_factor;
	  }
      }
      else
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	  {
	    phase_factor = std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );
	    VI += _I[i][j] * phase_factor;
	    VQ += _Q[i][j] * phase_factor;
	    VU += _U[i][j] * phase_factor;
	    VV += _V[i][j] * phase_factor;
	  }
      }
      
      // Apply cubic spline
      double spline_factor = cubic_spline_kernel(ur,vr) * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]);
      VI *= spline_factor;
      VQ *= spline_factor;
      VU *= spline_factor;
      VV *= spline_factor;

      // Convert to RR, LL, RL, LR
      std::vector< std::complex<double> > crosshand_vector(4);
      crosshand_vector[0] = VI+VV; // RR
      crosshand_vector[1] = VI-VV; // LL 
      crosshand_vector[2] = VQ+std::complex<double>(0.0,1.0)*VU; // RL
      crosshand_vector[3] = VQ-std::complex<double>(0.0,1.0)*VU; // LR

      // Apply Dterms
      apply_Dterms(d,crosshand_vector);
    
      return ( crosshand_vector );
    }
    else // NOT IMPLEMENTED
    {
      std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::crosshand_visiblities :"
		<< "       Numerical visibilities have not been properly implemented yet.\n"
		<< '\n';
      std::exit(1);
    }
  }


  std::complex<double> model_polarized_image_adaptive_splined_raster::visibility(datum_visibility& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      // Counter-rotate point
      double ur =  _cpa*d.u + _spa*d.v;
      double vr = -_spa*d.u + _cpa*d.v;
    
      std::complex<double> V(0.0,0.0);
      for (size_t i=0; i<_Nx; ++i)
	for (size_t j=0; j<_Ny; ++j)
	  V += _I[i][j] * std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );
      return ( cubic_spline_kernel(ur,vr)*V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]) );
    }
    else // NOT ROTATED
      return ( cubic_spline_kernel(d.u,d.v)*model_polarized_image::visibility(d, acc) );
  }

  double model_polarized_image_adaptive_splined_raster::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp(d.u,d.v,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      return std::abs(visibility(tmp,acc));
    }
    else
      return ( cubic_spline_kernel(d.u,d.v)*model_polarized_image::visibility_amplitude(d, acc) );
  }

  double model_polarized_image_adaptive_splined_raster::closure_phase(datum_closure_phase& d, double acc)
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
      return ( model_polarized_image::closure_phase(d,acc) );
  }



  double model_polarized_image_adaptive_splined_raster::closure_amplitude(datum_closure_amplitude& d, double acc)
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
      return ( model_polarized_image::closure_amplitude(d,acc) );
  }


  double model_polarized_image_adaptive_splined_raster::cubic_spline_kernel_1d(double k) const
  {
    if (std::fabs(k)<1e-2)
      return 1.0 - (2.0*_a-1.0)*(k*k)/15.0 + (16.0*_a+1.0)*(k*k*k*k)/560.0;
    else
    {
      double sk=std::sin(k), ck=std::cos(k);
      double c2k=ck*ck-sk*sk;

      double G = -4.0*sk*(2.0*_a*ck+(4.0*_a+3.0))/(k*k*k) + 12.0*(_a*(1.0-c2k)+2.0*(1.0-ck))/(k*k*k*k);

      return G;
    }
  }

  double model_polarized_image_adaptive_splined_raster::cubic_spline_kernel(double u0, double v0) const
  {
    return cubic_spline_kernel_1d(u0*_tpdx)*cubic_spline_kernel_1d(v0*_tpdy);
  }

};

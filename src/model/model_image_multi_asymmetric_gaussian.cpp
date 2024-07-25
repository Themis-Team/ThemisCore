/*!
  \file model_image_multi_asymmetric_gaussian.cpp
  \author Avery Broderick
  \date  October, 2018
  \brief Implements the multi-Asymmetric Gaussian image class originally motivated by model fitting challenge 3 within the MCFE WG.
  \details To be added
*/

#include "model_image_multi_asymmetric_gaussian.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>
#include <sstream>

namespace Themis {

  model_image_multi_asymmetric_gaussian::model_image_multi_asymmetric_gaussian(size_t N)
    : _N(N), _Icomp(N), _sigma_min(N), _sigma_maj(N), _phi(N), _x(N), _y(N), _use_analytical_visibilities(true)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_multi_asymmetric_gaussian in rank " << world_rank << std::endl;
    //std::cout << "Creating model_image_multi_asymmetric_gaussian in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
  }

  void model_image_multi_asymmetric_gaussian::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;
    //std::cout << "Using numerical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;

    _use_analytical_visibilities = false;
  }
  
  void model_image_multi_asymmetric_gaussian::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
    //std::cout << "Using analytical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
    
    _use_analytical_visibilities = true;
  }

  std::string model_image_multi_asymmetric_gaussian::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_multi_asymmetric_gaussian " << _N;
    return tag.str();
  }
  
  void model_image_multi_asymmetric_gaussian::generate_model(std::vector<double> parameters)
  {
    // Assumes the last parameter is the position angle, saves it and strips it off
    _position_angle = parameters.back();
    parameters.pop_back();

  
    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      _current_parameters = parameters;
      
      double sigma, A;
      for (size_t n=0,k=0; n<_N; n++)
      {
	_Icomp[n] = _current_parameters[k++];

	sigma = _current_parameters[k++];
	A = std::min(std::max(_current_parameters[k++],0.0),0.99);
	_sigma_min[n] = sigma/std::sqrt(1.0+A);
	_sigma_maj[n] = sigma/std::sqrt(1.0-A);

	_phi[n] = _current_parameters[k++];
	_x[n] = _current_parameters[k++];
	_y[n] = _current_parameters[k++];

	std::cerr << "mimag: I,s,A,phi,x,y"
		  << std::setw(15) << _Icomp[n]
		  << std::setw(15) << sigma
		  << std::setw(15) << A
		  << std::setw(15) << _phi[n]
		  << std::setw(15) << _x[n]
		  << std::setw(15) << _y[n]
		  << std::endl;
      }

      // Generate the image using the user-supplied routine
      if (_use_analytical_visibilities==false)
	generate_image(parameters,_I,_alpha,_beta);
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }
  

  void model_image_multi_asymmetric_gaussian::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    int Ntot = 128;
    double Ns = 4.0; // Number of sigmas to show in image

    // Allocate if necessary
    if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(Ntot))
    {
      alpha.resize(Ntot);
      beta.resize(Ntot);
      I.resize(Ntot);
    }
    for (size_t j=0; j<alpha.size(); j++)
    {
      if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(Ntot))
      {
        alpha[j].resize(Ntot,0.0);
        beta[j].resize(Ntot,0.0);
        I[j].resize(Ntot,0.0);
      }
    }

    // Find minimum/maximu based on 2 sigma + max shift
    double alpha_min = _x[0]-Ns*_sigma_maj[0];
    double alpha_max = _x[0]+Ns*_sigma_maj[0];
    double beta_min = _y[0]-Ns*_sigma_maj[0];
    double beta_max = _y[0]+Ns*_sigma_maj[0];

    for (size_t N_idx=1; N_idx<_N; N_idx++)
    {
      alpha_min = std::min(alpha_min,_x[N_idx]-Ns*_sigma_maj[N_idx]);
      alpha_max = std::max(alpha_max,_x[N_idx]+Ns*_sigma_maj[N_idx]);
      beta_min = std::min(beta_min,_y[N_idx]-Ns*_sigma_maj[N_idx]);
      beta_max = std::max(beta_max,_y[N_idx]+Ns*_sigma_maj[N_idx]);
    }

    // Fill array with new image
    for (size_t j=0; j<alpha.size(); j++)
    {
      for (size_t k=0; k<alpha[j].size(); k++)
      {
	// _sigma[0] only for now...
        alpha[j][k] = (double(j)*(alpha_max-alpha_min)/double(Ntot-1) + alpha_min);
        beta[j][k] = (double(k)*(beta_max-beta_min)/double(Ntot-1) + beta_min);

	for (size_t N_idx=0; N_idx<_N; N_idx++) {

	  double Inorm = _Icomp[N_idx]/(_sigma_min[N_idx]*_sigma_maj[N_idx]*2*M_PI);
	  double c = std::cos(_phi[N_idx]), s = std::sin(_phi[N_idx]);
	  double xmaj = s*(alpha[j][k]-_x[N_idx]) + c*(beta[j][k]-_y[N_idx]);
	  double xmin = c*(alpha[j][k]-_x[N_idx]) - s*(beta[j][k]-_y[N_idx]);
	  

	  I[j][k] += Inorm * std::exp( - 0.5 * ( (xmaj*xmaj)/(_sigma_maj[N_idx]*_sigma_maj[N_idx]) +  (xmin*xmin)/(_sigma_min[N_idx]*_sigma_min[N_idx]) ) );

	}

      }
    }
  } 


  std::complex<double> model_image_multi_asymmetric_gaussian::complex_visibility(double u, double v)
  {
    double ru = u*std::cos(_position_angle) + v*std::sin(_position_angle);
    double rv = -u*std::sin(_position_angle) + v*std::cos(_position_angle);

    //Reflection to have the image look the way it's seen in the sky
    ru *= -2.*M_PI; 
    rv *= 2.*M_PI;
      
    const std::complex<double> i(0.0,1.0);
    std::complex<double> exponent;
    std::complex<double> V=0.;
      
    double c,s, uvmaj,uvmin;
    for (size_t N_idx=0; N_idx<_N; N_idx++) {

      c = std::cos(_phi[N_idx]);
      s = std::sin(_phi[N_idx]);
      uvmaj = ru*s + rv*c;
      uvmin = ru*c - rv*s;

      exponent =  - i * (_x[N_idx]*ru + _y[N_idx]*rv) - 0.5*uvmaj*uvmaj*_sigma_maj[N_idx]*_sigma_maj[N_idx] - 0.5*uvmin*uvmin*_sigma_min[N_idx]*_sigma_min[N_idx];
      V += _Icomp[N_idx] * std::exp(exponent);

    }

    return ( V );
  }


  double model_image_multi_asymmetric_gaussian::closure_phase(datum_closure_phase& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      std::complex<double> V123 = complex_visibility(d.u1,d.v1)*complex_visibility(d.u2,d.v2)*complex_visibility(d.u3,d.v3);
      
      return ( std::imag(std::log(V123))*180.0/M_PI );
    }
    else
    {
      return ( model_image::closure_phase(d,acc) );
    }
  }



  double model_image_multi_asymmetric_gaussian::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double V1234 = std::abs((complex_visibility(d.u1,d.v1)*complex_visibility(d.u3,d.v3))/(complex_visibility(d.u2,d.v2)*complex_visibility(d.u4,d.v4)));
      
      return ( V1234 );
    }
    else
    {
      return ( model_image::closure_amplitude(d,acc) );
    }
  }

  double model_image_multi_asymmetric_gaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double VM = std::abs(complex_visibility(d.u, d.v));
            
      return ( VM );
    }
    
    else
    {
      return ( model_image::visibility_amplitude(d,acc) );
    }

  }

  std::complex<double> model_image_multi_asymmetric_gaussian::visibility(datum_visibility& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      return ( complex_visibility(d.u, d.v) );
    }
    
    else
    {
      return ( model_image::visibility(d,acc) );
    }

  }

  
};

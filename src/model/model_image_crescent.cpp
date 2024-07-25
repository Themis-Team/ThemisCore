/*!
  \file model_image_crescent.cpp
  \author Jorge A. Preciado, Hung-Yi Pu
  \date  June 2017
  \brief Implements Geometric Crescent Model Image class.
  \details To be added
*/

#include "model_image_crescent.h"
#include <iostream>
//#include <fstream>
#include <iomanip>
#include <complex>
#include <mpi.h>

namespace Themis {

  model_image_crescent::model_image_crescent()
    : _Nray(128), _use_analytical_visibilities(true)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_crescent in rank " << world_rank << std::endl;
  }

  void model_image_crescent::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }
  
  void model_image_crescent::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
    
    _use_analytical_visibilities = true;
  }

  void model_image_crescent::set_image_resolution(int Nray)
  {
    _Nray = Nray;
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "model_image_crescent: Rank " << world_rank << " using image resolution " << _Nray << std::endl;
  }
  

  void model_image_crescent::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    int Ntot = _Nray;
    double Rs = 2.0; // Number of Radius
    
    _Itotal = parameters[0];
    _Rp = parameters[1];
    _Rn = (1 - parameters[2]) * parameters[1];
    _a = (1 - parameters[3]) * parameters[1] * parameters[2];
    _b = 0;
    //_a = (1 - parameters[3]) * parameters[1] * parameters[2] * std::cos(parameters[4]);
    //_b = (1 - parameters[3]) * parameters[1] * parameters[2] * std::sin(parameters[4]);

    if (_use_analytical_visibilities==false)
    {

      double Inorm = _Itotal/(M_PI * (_Rp*_Rp - _Rn*_Rn));
    
      // Allocate if necessary
      if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(Ntot))
      {
        alpha.resize(Ntot);
        beta.resize(Ntot);
        I.resize(Ntot);
      }
      for (size_t j=0; j<alpha.size(); ++j)
      {
        if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(Ntot))
        {
          alpha[j].resize(Ntot,0.0);
          beta[j].resize(Ntot,0.0);
          I[j].resize(Ntot,0.0);
        }
      }
      
      // Fill array with new image
      for (size_t j=0; j<alpha.size(); ++j)
      {
        for (size_t k=0; k<alpha[j].size(); ++k)
        {
          alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_Rp*2.0*Rs/double(Ntot));
          beta[j][k]  = ((double(k)-0.5*double(Ntot)+0.5)*_Rp*2.0*Rs/double(Ntot));
          I[j][k] = 0.0;
          
          if (alpha[j][k]*alpha[j][k] + beta[j][k]*beta[j][k] <= _Rp*_Rp) {
            I[j][k] = 1.0;
          }
          
          if ( ((alpha[j][k]-_a)*(alpha[j][k]-_a) + (beta[j][k]-_b)*(beta[j][k]-_b)) <= _Rn*_Rn) {
            I[j][k] = 0.0;
          }
          
          I[j][k] *= Inorm;
        }
      }
    }
  }
  
  
  double model_image_crescent::BesselJ1(double x)
  {
    double ax,z;
    double xx,y,ans,ans1,ans2;

    if ((ax=std::fabs(x)) < 8.0) {
      y=x*x;
      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
        +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
        +y*(99447.43394+y*(376.9991397+y*1.0))));
      ans=ans1/ans2;
    } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-2.356194491;
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
        +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
        +y*(0.8449199096e-5+y*(-0.88228987e-6
        +y*0.105787412e-6)));
      ans=std::sqrt(0.636619772/ax)*(std::cos(xx)*ans1-z*std::sin(xx)*ans2);
      if (x < 0.0) ans = -ans;
    }
    return ans;

  }


  std::complex<double> model_image_crescent::complex_visibility(double u, double v)
  {
    double ru = u*std::cos(_position_angle) + v*std::sin(_position_angle);
    double rv = -u*std::sin(_position_angle) + v*std::cos(_position_angle);
    
    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;
      
    double k = std::sqrt(ru*ru + rv*rv) + 1.e-15*_Rp;
      
    const std::complex<double> i(0.0,1.0);
    std::complex<double> exponent;
    std::complex<double> V;
      
    exponent = - i * (_a*ru + _b*rv);
      
    V = 2.*_Itotal/(k *(_Rp * _Rp - _Rn * _Rn)) 
        * ( _Rp*BesselJ1(k * _Rp) - std::exp(exponent)*_Rn*BesselJ1(k*_Rn) );
    
    return ( V );
  }



  double model_image_crescent::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double VM = std::abs(complex_visibility(d.u, d.v));
            
      return ( VM );
    }
    
    else
    {
      // std::cout << "(d.u, d.v) = (" << d.u << ", " << d.v << ")" << std::endl;
      return ( model_image::visibility_amplitude(d,acc) );
    }
    
  }
  
  double model_image_crescent::closure_phase(datum_closure_phase& d, double acc)
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


  double model_image_crescent::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double V1 = std::abs(complex_visibility(d.u1,d.v1));
      double V2 = std::abs(complex_visibility(d.u2,d.v2));
      double V3 = std::abs(complex_visibility(d.u3,d.v3));
      double V4 = std::abs(complex_visibility(d.u4,d.v4));
      
      double V1234 = (V1*V3)/(V2*V4);

      return ( V1234 );
    }
    else
    {
      return ( model_image::closure_amplitude(d,acc) );
    }
  }

  std::complex<double> model_image_crescent::visibility(datum_visibility& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      return ( complex_visibility(d.u,d.v) );
    }
    else
    {
      return ( model_image::visibility(d,acc) );
    }
  }


};

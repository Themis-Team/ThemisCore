/*!
  \file model_image_xsringauss.cpp
  \author Jorge Alejandro Preciado-Lopez
  \date  November 2018
  \brief Implements the geometric nine-parameter eccentric slashed ring (xsringauss) model image class.
  \details To be added
*/

#include "model_image_xsringauss.h"
#include <iostream>
#include <iomanip>
#include <complex>
#include <mpi.h>

namespace Themis {

  model_image_xsringauss::model_image_xsringauss()
    : _Nray(128), _use_analytical_visibilities(true)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_xsringauss in rank " << world_rank << std::endl;
  }

  void model_image_xsringauss::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }
  
  void model_image_xsringauss::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
    
    _use_analytical_visibilities = true;
  }

  void model_image_xsringauss::set_image_resolution(int Nray)
  {
    _Nray = Nray;
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "model_image_xsringauss: Rank " << world_rank << " using image resolution " << _Nray << std::endl;
  }
  

  void model_image_xsringauss::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    int Ntot = _Nray;
    double Nr = 2.0; // Number of Radius
    double Ns = 4.0;

    /*
    _V0 = parameters[0];
    _Rext = parameters[1];
    _Rint = (1 - parameters[2]) * parameters[1];
    _d = parameters[3] * (_Rext - _Rint);
    _f = parameters[4];
    _sigma_alpha = parameters[5] * _Rext;
    _sigma_beta  = parameters[6] * _sigma_alpha;
    _gq = parameters[7];
    */

    // With enforced regions
    _V0 = std::max(1e-8,parameters[0]);
    _Rext = std::max(1e-20,parameters[1]);
    _Rint = std::min( std::max(1e-4,1-parameters[2]), 0.9999 ) * _Rext;
    _d = std::min(std::max(parameters[3],1e-4),0.9999) * (_Rext - _Rint);
    _f = std::min(std::max(parameters[4],1e-4),0.9999);
    _sigma_alpha = std::max(parameters[5],1e-4) * _Rext;
    _sigma_beta  = std::max(parameters[6],1e-4) * _sigma_alpha;
    _gq = std::min(std::max(parameters[7],1e-4),0.9999);

    if (_use_analytical_visibilities==false)
    {

      /*
      std::cout << " PARAMETERS PARAMETERS PARAMETERS PARAMETERS PARAMETERS PARAMETERS PARAMETERS" << std::endl;
      std::cout << " V0 = " << _V0 << std::endl;
      std::cout << " Re = " << _Rext << std::endl;
      std::cout << " Ri = " << _Rint << std::endl;
      std::cout << " d = "  << _d << std::endl;
      std::cout << " f = "  << _f << std::endl;
      std::cout << " a = "  << _sigma_alpha << std::endl;
      std::cout << " b = "  << _sigma_beta << std::endl;
      */
      
      //double Inorm =            (2.*_V0/M_PI) * 1./((1. + _f)*(_Rext*_Rext - _Rint*_Rint) - (1 - _f)*_d*_Rint*_Rint/_Rext);
      double Iring_norm = (1-_gq)*(2.*_V0/M_PI) * 1./((1. + _f)*(_Rext*_Rext - _Rint*_Rint) - (1 - _f)*_d*_Rint*_Rint/_Rext);
      double Igauss_norm = _gq*_V0/(2.*M_PI*_sigma_alpha*_sigma_beta);
      
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
    
      //double alpha_min = std::min(-Nr*_Rext,_d-_Rint-Ns*_sigma_alpha);
      //double alpha_max = std::max(Nr*_Rext,_d-_Rint+Ns*_sigma_alpha);
      double alpha_min = std::min(-Nr*_Rext,-Ns*_sigma_alpha);
      double alpha_max = std::max(Nr*_Rext,Ns*_sigma_alpha);
      double beta_min = std::min(-Nr*_Rext,-Ns*_sigma_beta);
      double beta_max = std::max(Nr*_Rext,Ns*_sigma_beta);

      /*
      std::cout << " ALPHA BETA ALPHA BETA ALPHA BETA ALPHA BETA ALPHA BETA ALPHA BETA" << std::endl;
      std::cout << " alpha_min = " << alpha_min << std::endl;
      std::cout << " alpha_max = " << alpha_max << std::endl;
      std::cout << " beta_min = " << beta_min << std::endl;
      std::cout << " beta_max = " << beta_max << std::endl;
      */
      
      // Fill array with new image
      for (size_t j=0; j<alpha.size(); ++j)
      {
	for (size_t k=0; k<alpha[j].size(); ++k)
        {
	  //alpha[j][k] = (double(j)-0.5*double(Ntot)+0.5)*_Rext*2.0*Nr/double(Ntot);
	  //beta[j][k]  = (double(k)-0.5*double(Ntot)+0.5)*_Rext*2.0*Nr/double(Ntot);
	  
	  alpha[j][k] = (double(j)-0.5*double(Ntot)+0.5)*2.0*(alpha_max-alpha_min)/double(Ntot);
	  beta[j][k] =  (double(k)-0.5*double(Ntot)+0.5)*2.0*(beta_max-beta_min)/double(Ntot);
	  
	  I[j][k] = 0.0;
	  
	  if (alpha[j][k]*alpha[j][k] + beta[j][k]*beta[j][k] <= _Rext*_Rext) {
	    //I[j][k] = 0.5 * (1 + alpha[j][k]/_Rext) + 0.5 * _f * (1 - alpha[j][k]/_Rext);
	    I[j][k] = 0.5 * (1 - alpha[j][k]/_Rext) + 0.5 * _f * (1 + alpha[j][k]/_Rext);
	  }
	  
	  if ( ((alpha[j][k]-_d)*(alpha[j][k]-_d) + (beta[j][k])*(beta[j][k])) <= _Rint*_Rint) {
	    I[j][k] = 0.0;
	  }
	  
	  I[j][k] *= Iring_norm;
	  
	  I[j][k] += Igauss_norm * std::exp( -0.5*std::pow((alpha[j][k]-(_d-_Rint))/_sigma_alpha,2) 
					     -0.5*std::pow(beta[j][k]/_sigma_beta,2) );
	  
	}
      }
    }  
  }
  
  
  double model_image_xsringauss::BesselJ0(double x)
  {
    double ax,z;
    double xx,y,ans,ans1,ans2;
    
    if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7 
        +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
      ans2=57568490411.0+y*(1029532985.0+y*(9494680.718 
        +y*(59272.64853+y*(267.8532712+y*1.0))));
      ans=ans1/ans2;
    }
    
    else {
      z=8.0/ax;
      y=z*z;
      xx=ax-0.785398164;
      ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
        +y*(-0.2073370639e-5+y*0.2093887211e-6))); 
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
        +y*(-0.6911147651e-5+y*(0.7621095161e-6
        -y*0.934945152e-7)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    }
    return ans; 
  }

  
  
  double model_image_xsringauss::BesselJ1(double x)
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


  double model_image_xsringauss::BesselJ2(double x)
  {
    return (2*BesselJ1(x)/x - BesselJ0(x));
  }
  
  
  std::complex<double> model_image_xsringauss::complex_visibility(double u, double v)
    {
      double ru = u*std::cos(_position_angle) + v*std::sin(_position_angle);
      double rv = -u*std::sin(_position_angle) + v*std::cos(_position_angle);
      
      ru *= -1.0; //Reflection to have the image look the way it's seen in the sky
        
      double k = 2.*M_PI*std::sqrt(ru*ru + rv*rv); //+ 1.e-15*_Rext;
      
      double H = (2./M_PI) * 1./((1. + _f)*(_Rext*_Rext - _Rint*_Rint) - (1 - _f)*_d*_Rint*_Rint/_Rext);
            
      const std::complex<double> i(0.0,1.0);
      std::complex<double> exponent,exponent_gauss;
      std::complex<double> Vring, Vgauss, V;
      double exponent_gauss_arg;
 
      exponent = -2.*M_PI*i*_d*ru ;
      
      /*
      Vring = (M_PI*H/k) * (1 + _f) * _Rext*BesselJ1(k*_Rext)
      - (M_PI*H/k) * ((1 + _f) + (1 - _f)*_d/_Rext) * std::exp(exponent)*_Rint*BesselJ1(k*_Rint)
      + (i*M_PI*H/(2.*k*k)) * 2.*M_PI*ru * (1 - _f) * ( _Rext*BesselJ0(k*_Rext) - _Rext*BesselJ2(k*_Rext) - 2.*BesselJ1(k*_Rext)/k )
      - (i*M_PI*H/(2.*k*k)) * 2.*M_PI*ru * (1 - _f) * ( _Rint*BesselJ0(k*_Rint) - _Rint*BesselJ2(k*_Rint) - 2.*BesselJ1(k*_Rint)/k ) * (_Rint/_Rext) * std::exp(exponent);
      */

      Vring = (M_PI*H/k) * (1 + _f) * _Rext*BesselJ1(k*_Rext)
      - (M_PI*H/k) * ((1 + _f) - (1 - _f)*_d/_Rext) * std::exp(exponent)*_Rint*BesselJ1(k*_Rint)
      - (i*M_PI*H/(2.*k*k)) * 2.*M_PI*ru * (1 - _f) * ( _Rext*BesselJ0(k*_Rext) - _Rext*BesselJ2(k*_Rext) - 2.*BesselJ1(k*_Rext)/k )
      + (i*M_PI*H/(2.*k*k)) * 2.*M_PI*ru * (1 - _f) * ( _Rint*BesselJ0(k*_Rint) - _Rint*BesselJ2(k*_Rint) - 2.*BesselJ1(k*_Rint)/k ) * (_Rint/_Rext) * std::exp(exponent);
      
      //exponent_gauss = - 2.0*M_PI*M_PI * ( ru*ru*(_sigma_alpha*_sigma_alpha) + rv*rv*(_sigma_beta*_sigma_beta) )
      //                 - 2.0*M_PI*i*(_d-_Rint)*ru ;
      exponent_gauss_arg = - 2.0*M_PI*M_PI * ( ru*ru*(_sigma_alpha*_sigma_alpha) + rv*rv*(_sigma_beta*_sigma_beta) );
      exponent_gauss = - 2.0*M_PI*i*(_d-_Rint)*ru ;                
      
      //Vgauss = std::exp( exponent_gauss );
      Vgauss = ( exponent_gauss_arg<-200.0 ? 0.0 : std::exp( exponent_gauss + exponent_gauss_arg ) );

      
      V = _V0*((1-_gq)*Vring + _gq*Vgauss);

      /*
      std::cout << " Vgauss = " << Vgauss << std::endl;
      std::cout << " V = " << V << std::endl;
      */

      return ( V );
    }
    
  
  double model_image_xsringauss::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      return ( std::abs(complex_visibility(d.u, d.v)) );
    }
    
    else
    {
      return ( model_image::visibility_amplitude(d,acc) );
    }
    
  }
  
  double model_image_xsringauss::closure_phase(datum_closure_phase& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      std::complex<double> V1 = complex_visibility(d.u1,d.v1);
      std::complex<double> V2 = complex_visibility(d.u2,d.v2);
      std::complex<double> V3 = complex_visibility(d.u3,d.v3);
      
      std::complex<double> B123 = V1*V2*V3;
            
      return ( std::imag(std::log(B123))*180.0/M_PI );
    }
    
    else
    {
      return ( model_image::closure_phase(d,acc) );
    }
  }


  double model_image_xsringauss::closure_amplitude(datum_closure_amplitude& d, double acc)
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
  
  
  std::complex<double> model_image_xsringauss::visibility(datum_visibility& d, double acc)
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

/*!
  \file model_circular_binary.cpp
  \author Roman Gold, Avery Broderick
  \date  July, 2017
  \brief Implements binary model based on two Gaussians.
  \details 

  \warning ...still under development...
  \todo ...still under development...
*/

#include "model_circular_binary.h"
#include "constants.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <algorithm>

namespace Themis {

  model_circular_binary::model_circular_binary()
  {
  }

  void model_circular_binary::generate_model(std::vector<double> parameters)
  {
    _I1 = parameters[0];
    _sigma1 = parameters[1];
    _alpha1 = parameters[2];
    _I2 = parameters[3];
    _sigma2 = parameters[4];
    _alpha2 = parameters[5];
    _M = parameters[6];
    _q = parameters[7];
    _R = parameters[8] * Themis::constants::pc;
    _D = parameters[9] * 1e6 * Themis::constants::pc;
    _Phi0 = parameters[10];
    _mu = parameters[11];
    _position_angle = parameters[12] * M_PI/180.0;

    _cosi = std::min(std::max(_mu,-1.0),1.0);
    _sini = std::sqrt(std::max(1.0-_cosi*_cosi,0.0));

    _Omega = std::sqrt(Themis::constants::G*_M /* * Themis::constants::Msun*/ /(_R*_R*_R));
    
  }

  double model_circular_binary::get_obs_time(double t_lab)
  {
    // double R_inM = _R / _M;
    // double Omega_inM = pow(R_inM, -1.5);
    // return (R_inM *_sini * std::sin(Omega_inM*t_lab + _Phi0) );

    // return (t_lab + _R*_sini/Themis::constants::c*std::sin(_Omega*t_lab + _Phi0) );
    return (_R *_sini * std::sin(_Omega*t_lab + _Phi0) );
  }

  // std::pair<double,double> model_circular_binary::get_lab_frame_time(double t_obs)
  double model_circular_binary::get_lab_frame_time(double t_obs)
  {
    // solve for t_lab via iteration
    const int max_iters = 10;
    const double tolerance = 1e-14; // typically 5 or 6 iterations necessary

    // double R_inM = _R / ( 1.5e5 * _M);


    // INITIAL GUESS DOES NOT SEEM TO MATTER

    // double delta = -_R/Themis::constants::c ; // initial guess (better delta = R/Themis::constants::c ? )
    // double delta = -R_inM ; // initial guess (better delta = R/Themis::constants::c ? )
    // double delta = _R/Themis::constants::c ; // initial guess (better delta = R/Themis::constants::c ? )
    double delta = 0.; // initial guess (better delta = R/Themis::constants::c ? )
    // double dtR = _R *_sini /Themis::constants::c;

    
    // std::cout<<std::setprecision(12)<<"_R ="<<_R<<" R_inM ="<<R_inM<<" _Omega ="<<_Omega<<" Given time in observer frame, t="<<std::setprecision(12)<<t_obs<<", get time in lab frame..."<<std::endl;

    for(int n=0; n<max_iters; n++) {

      // SOLVE

      // delta = dtR * std::sin(_Omega*(t_obs-delta)+_Phi0);
      // double res = t_obs-delta + dtR*std::sin(_Omega*(t_obs-delta)+_Phi0)-t_obs;

      delta = get_obs_time(t_obs + delta); // minimize delta := t_obs - t_lab
      double res = t_obs + delta - get_obs_time(t_obs + delta) - t_obs;


      // REPORT
      // std::cout<<"iteration ="<<n<<" t_obs ="<<t_obs<<" t_lab-t_obs ="<<delta<<" residual ="<<res<<std::endl;


      // BREAK
      if (fabs(res)<tolerance) {
	// std::cout<<"Hit residual floor. Algorithm converged."<<std::endl;
	// std::cout<<"t_obs ="<<t_obs<<" <=> t_lab ="<<t_obs-delta<<std::endl;
	break;
      }

    }

    return t_obs - delta; // return t_lab
    // return delta; // return difference between t_lab and t_obs
  }



  void model_circular_binary::get_orbital_solution(double t_lab, double x1[2], double& v1, double& v1r, double x2[2], double& v2, double& v2r)
  {
    // std::cout<<"YO start get_orbital_solution()"<<std::endl;

    // Primary:
    //  Position in radians (Omega t + Phi0 = 0 has the orbit at the maximum projected separation) 
    x1[0] = (_q/(1.0+_q))*(_R/_D) * std::cos(_Omega*t_lab+_Phi0);
    x1[1] = (_q/(1.0+_q))*(_R/_D)*_cosi * std::sin(_Omega*t_lab+_Phi0);
    //  Total velocity in cm/s
    v1 = (_q/(1.0+_q))*_R*_Omega;
    //  Line of sight velocity in cm/sm (right-handed orbit!)
    v1r = -(_q/(1.0+_q))*_R*_Omega*_sini * std::cos(_Omega*t_lab+_Phi0);

    // Secondary:
    //  Position in radians (Omega t + Phi0 = 0 has the orbit at the maximum projected separation) 
    x2[0] = -(1.0/(1.0+_q))*(_R/_D) * std::cos(_Omega*t_lab+_Phi0);
    x2[1] = -(1.0/(1.0+_q))*(_R/_D)*_cosi * std::sin(_Omega*t_lab+_Phi0);
    //  Total velocity in cm/s
    v2 = (1.0/(1.0+_q))*_R*_Omega;
    //  Line of sight velocity in cm/sm (right-handed orbit!)
    v2r = (1.0/(1.0+_q))*_R*_Omega*_sini * std::cos(_Omega*t_lab+_Phi0);

    // std::cout<<"YO end get_orbital_solution()"<<std::endl;
  }
  
  double model_circular_binary::doppler_factor(double v, double vr) const
  {
    // gamma ( 1 - beta.n )
    return ( (1.0-vr/Themis::constants::c)/std::sqrt(1.0-(v*v)/(Themis::constants::c*Themis::constants::c)) );
  }

  // Assumes u,v are in lambda and sigma is in radians
  double model_circular_binary::gaussian_visibility(double u, double v, double sigma) const
  {
    return ( std::exp(-2.0*M_PI*M_PI*(u*u+v*v)*sigma*sigma) );
  }
  
  std::complex<double> model_circular_binary::complex_visibility(double u, double v, double t)
  // double model_circular_binary::complex_visibility_time(double u, double v, double t)
  {
    // std::cout<<"YO start complex_visibility()"<<std::endl;

    // Get the orbital solutions
    double x1[2], x2[2], v1, v2, v1r, v2r;
    get_orbital_solution(get_lab_frame_time(t),x1,v1,v1r,x2,v2,v2r);
    
    const std::complex<double> i(0.0,1.0);

    // Get the contribution from the primary:
    std::complex<double> V1 = _I1                     // norm
      * std::pow(doppler_factor(v1,v1r),_alpha1+3.0)  // Doppler factors (shift & beaming)
      * gaussian_visibility(u,v,_sigma1)              // Shape of the primary
      * std::exp( 2.0*M_PI*i * (u*x1[0]+v*x1[1]) );   // Phase shift due to offset from center of mass
    
    // Get the contribution from the secondary:
    std::complex<double> V2 = _I2                     // norm
      * std::pow(doppler_factor(v2,v2r),_alpha2+3.0)  // Doppler factors (shift & beaming)
      * gaussian_visibility(u,v,_sigma2)              // Shape of the secondary
      * std::exp( 2.0*M_PI*i * (u*x2[0]+v*x2[1]) );   // Phase shift due to offset from center of mass

    // std::cout<<"YO end complex_visibility()"<<std::endl;
    return ( V1+V2 );
  }

  double model_circular_binary::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    // std::cout<<"YO start visibility_amplitude()"<<std::endl;

    double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);
    
    ru *= -1.0; //Reflection to have the image look the way it's seen in the sky
    
    double VM = std::abs(complex_visibility(ru, rv, d.tJ2000));

    if (std::isnan(VM)) {
	std::cerr<<"ERROR in model_circular_binary::visibility_amplitude() VM is nan..."<<std::endl;
      }

    // std::cout<<"YO end visibility_amplitude()"<<std::endl;    
    return ( VM );
  }

  double model_circular_binary::closure_phase(datum_closure_phase& d, double acc)
  {
    // std::cout<<"YO start closure_phase()"<<std::endl;

    // Obtain counter-rotated u,v coordinates
    double u[]={d.u1,d.u2,d.u3}, v[]={d.v1,d.v2,d.v3};
    double ru, rv;
    std::complex<double> V[3];
    double c=std::cos(_position_angle), s=std::sin(_position_angle);
    for (int j=0; j<3; ++j)
    {
      ru = u[j]*c + v[j]*s;
      rv = -u[j]*s + v[j]*c;

      ru *= -1.0; //Reflection to have the image look the way it's seen in the sky

      V[j] = complex_visibility(ru,rv,d.tJ2000);
    }
    // std::cout<<"YO end closure_phase()"<<std::endl;
    return ( std::imag(std::log(V[0]*V[1]*V[2]))*180.0/M_PI );
  }

  double model_circular_binary::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    // Obtain counter-rotated u,v coordinates
    double u[]={d.u1,d.u2,d.u3,d.u4}, v[]={d.v1,d.v2,d.v3,d.v4};
    double ru, rv, VM[4];
    double c=std::cos(_position_angle), s=std::sin(_position_angle);
    for (int j=0; j<4; ++j)
    {
      ru = u[j]*c + v[j]*s;
      rv = -u[j]*s + v[j]*c;

      ru *= -1.0; //Reflection to have the image look the way it's seen in the sky

      VM[j] = std::abs( complex_visibility(ru,rv,d.tJ2000) );
    }
    return ( (VM[0]*VM[2])/ (VM[1]*VM[3]) );
  }
};

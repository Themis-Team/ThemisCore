/*!
  \file model_image_mring_floor.cpp
  \author Paul Tiede
  \date  Mar 2021
  \brief Implements Slashed Ring Model Image class.
  \details To be added
*/

#include "model_image_mring_floor.h"
#include <iostream>
#include <iomanip>
#include <complex>
#include <mpi.h>
#include <limits>

namespace Themis {

  model_image_mring_floor::model_image_mring_floor(size_t order)
    : _order(order), _an(_order, 0.0), _bn(_order, 0.0), _Nray(128), _use_analytical_visibilities(true)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_mring_floor in rank " << world_rank << std::endl;
  }

  void model_image_mring_floor::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }
  
  void model_image_mring_floor::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
    
    _use_analytical_visibilities = true;
  }

  void model_image_mring_floor::set_image_resolution(int Nray)
  {
    _Nray = Nray;
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "model_image_mring_floor: Rank " << world_rank << " using image resolution " << _Nray << std::endl;
  }


void model_image_mring_floor::generate_model(std::vector<double> parameters)
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
  

  void model_image_mring_floor::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    int Ntot = _Nray;
    
    _V0 = parameters[0];
    _R = parameters[1];
    _floor = parameters[2];
    int m = 3;
    for ( size_t i = 0; i < _order; i++ ){
        double amp = parameters[m++];
        double phase = parameters[m++];
        _an[i] = amp*std::cos(phase);
        _bn[i] = amp*std::sin(phase);
    }

    if (_use_analytical_visibilities==false){
     
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
        double psize = 4*_R/(_Nray-1);
        double fov = 4*_R;
        double flux = 0.0;
        // Fill array with new image
        for (size_t j=0; j<alpha.size(); ++j)
        {
            for (size_t k=0; k<alpha[j].size(); ++k)
            {
                alpha[j][k] = -fov/2 + psize*j; 
                beta[j][k]  = -fov/2 + psize*k; 
        
                I[j][k] = 0.0;
        
        
                if ( std::fabs(((alpha[j][k])*(alpha[j][k]) + (beta[j][k])*(beta[j][k]))-_R) <= psize) {
                    I[j][k] += _V0*(1-_floor)/(2*M_PI*_R);
                }
                if (alpha[j][k]*alpha[j][k] + beta[j][k]*beta[j][k] < _R*_R)
                {
                    I[j][k] += _V0*_floor/(M_PI*_R*_R);
                }
            }
        }

        for ( size_t j = 0; j < alpha.size(); ++j ){
            for ( size_t k = 0; k < alpha[0].size(); ++k)
                I[j][k] *= _V0/flux;
        }
  
    }
  }
  
  
  double model_image_mring_floor::BesselJ0(double x)
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

  
  
  double model_image_mring_floor::BesselJ1(double x)
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


  double model_image_mring_floor::BesselJ(int n, double x)
  {
    //Define some constants to be used below
    const double ACC = 160.0;
    const int IEXP = std::numeric_limits<double>::max_exponent/2;
    

    if (n==0){
        return BesselJ0(x);
    }else if (n==1){
        return BesselJ1(x);
    }

    bool jsum;
    int j,k,m;
    double ax, bj, bjp, bjm, sum, tox, ans;

    //start the recursion
    ax = std::fabs(x); 
    if (ax*ax <= 8.0*std::numeric_limits<double>::min()){
        return 0.0;
    //start the recurrence from j1,j0
    }else if ( ax > double(n)){
        tox = 2.0/ax;
        bjm = BesselJ0(ax); //recursion down
        bj = BesselJ1(ax); //recursion current
        for ( j = 1; j < n; j++ ){
            bjp = j*tox*bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        ans = bj;
    } else{
        tox = 2.0/ax;
        m = 2*( (n + int(std::sqrt(ACC*n))/2));
        jsum = false;
        bjp = ans = sum = 0.0;
        bj = 1.0;

        for ( j = m; j > 0; j-- ){
            bjm = j*tox*bj - bjp;
            bjp = bj;
            bj = bjm;
            std::frexp(bj, &k);
            if ( k > IEXP ){
                bj = std::ldexp(bj, -IEXP);
                bjp = std::ldexp(bjp, -IEXP);
                ans = std::ldexp(ans, -IEXP);
                sum = std::ldexp(sum, -IEXP);
            }
            if (jsum){
                sum += bj;
            }
            jsum = !jsum;

            if (j == n){
                ans = bjp;
            }
        }

        sum = 2.0*sum - bj;
        ans /= sum;
    }


    return x < 0.0 &&(n & 1) ? -ans : ans;

  }
  
  
  std::complex<double> model_image_mring_floor::complex_visibility(double u, double v)
    {
      //double ru = u*std::cos(_position_angle) + v*std::sin(_position_angle);
      //double rv = -u*std::sin(_position_angle) + v*std::cos(_position_angle);
      
      //ru *= -1.0; //Reflection to have the image look the way it's seen in the sky
        
      double k = 2.*M_PI*std::sqrt(u*u + v*v)*_R + 1e-16;
      
      std::complex<double> vis(BesselJ0(k), 0.0); 
      const std::complex<double> im(0.0,1.0);

      double theta = std::atan2(u,v);
      for (size_t n = 1; n <= _order; ++n){
          double s = std::sin(n*theta);
          double c = std::cos(n*theta);
          vis += 2*(_an[n-1]*c - _bn[n-1]*s)*std::exp(std::complex<double>(0.0,M_PI/2.0*n))*BesselJ(n, k);
          //std::cerr << nn << "     vis: " << vis << std::endl;
      }

    
      std::complex<double> vdisk(2*BesselJ1(k)/(k), 0.0);

      return vis*_V0*(1-_floor) + _floor*vdisk;
    }
    
  
  double model_image_mring_floor::visibility_amplitude(datum_visibility_amplitude& d, double acc)
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
  
  double model_image_mring_floor::closure_phase(datum_closure_phase& d, double acc)
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


  double model_image_mring_floor::closure_amplitude(datum_closure_amplitude& d, double acc)
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
  
  
  std::complex<double> model_image_mring_floor::visibility(datum_visibility& d, double acc)
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

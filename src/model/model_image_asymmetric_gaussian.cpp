/*!
  \file model_image_asymmetric_gaussian.cpp
  \author Avery Broderick
  \date  June, 2017
  \brief Implements asymmetric Gaussian image class.
  \details To be added
*/

#include "model_image_asymmetric_gaussian.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <algorithm>
#include <cmath>

namespace Themis {

model_image_asymmetric_gaussian::model_image_asymmetric_gaussian()
  : _use_analytical_visibilities(true)
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Creating model_image_asymmetric_gaussian in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_asymmetric_gaussian in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}

void model_image_asymmetric_gaussian::use_numerical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;
  //std::cout << "Using numerical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
  _use_analytical_visibilities = false;
}

void model_image_asymmetric_gaussian::use_analytical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
  //std::cout << "Using analytical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;

  _use_analytical_visibilities = true;
}

void model_image_asymmetric_gaussian::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  int Ntot = 128;
  double Ns = 4.0;

  /*
  _Itotal = std::fabs(parameters[0]);

  parameters[1] = std::fabs(parameters[1]);

  //parameters[2] = std::max(0.1,std::min(0.9,parameters[2]));
  parameters[2] = std::min(std::max(parameters[2],0.0),0.99);

  _sigma_beta = parameters[1] * std::sqrt( 1.0 / (1.0-parameters[2]) ); // Major axis
  _sigma_alpha  = parameters[1] * std::sqrt( 1.0 / (1.0+parameters[2]) ); // Minor axis
  */


  const double p0 = parameters[0];
  const double p1 = parameters[1];
  const double p2 = parameters[2];

  _Itotal     = std::fabs(p0);
  _sigma_param = std::fabs(p1);
  _A_param     = std::min(std::max(p2,0.0),0.99);

  _dItotal_dp0 = (p0 >= 0.0 ? 1.0 : -1.0);
  _dsigma_dp1  = (p1 >= 0.0 ? 1.0 : -1.0);
  _dA_dp2      = ((p2 > 0.0 && p2 < 0.99) ? 1.0 : 0.0);

  _sigma_beta  = _sigma_param * std::sqrt(1.0 / (1.0 - _A_param)); // Major axis
  _sigma_alpha = _sigma_param * std::sqrt(1.0 / (1.0 + _A_param)); // Minor axis


  
  if (_use_analytical_visibilities==false)
  {
  
    double Inorm = _Itotal/(_sigma_alpha*_sigma_beta*2*M_PI);

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
    
    // Fill array with new image
    
    double exponent;
    
    for (size_t j=0; j<alpha.size(); j++)
    {
      for (size_t k=0; k<alpha[j].size(); k++)
      {
	alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_sigma_alpha*2.0*Ns/double(Ntot));
	beta[j][k] = ((double(k)-0.5*double(Ntot)+0.5)*_sigma_beta*2.0*Ns/double(Ntot));
	exponent = - 0.5 * ( alpha[j][k]*alpha[j][k]/(_sigma_alpha*_sigma_alpha)  + beta[j][k]*beta[j][k]/(_sigma_beta*_sigma_beta) );
	I[j][k] = Inorm * ( exponent<-200.0 ? 0.0 : std::exp(exponent) );
      }
    }
  }
}

double model_image_asymmetric_gaussian::closure_phase(datum_closure_phase& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    return 0.;
  }
  else
  {
    return ( model_image::closure_phase(d,acc) );
  }
}


void model_image_asymmetric_gaussian::visibility_and_derivatives(
    datum_visibility& d,
    std::complex<double>& V,
    std::complex<double> dV[4]) const
{
  const double cpa = std::cos(_position_angle);
  const double spa = std::sin(_position_angle);

  const double ru = -2.0*M_PI * ( d.u*cpa + d.v*spa );
  const double rv =  2.0*M_PI * (-d.u*spa + d.v*cpa );

  const double sa2 = _sigma_alpha * _sigma_alpha;
  const double sb2 = _sigma_beta  * _sigma_beta;

  const double expo = -0.5 * (ru*ru*sa2 + rv*rv*sb2);
  const double e = (expo < -200.0 ? 0.0 : std::exp(expo));

  V = std::complex<double>(_Itotal * e, 0.0);

  for (int k = 0; k < 4; ++k)
    dV[k] = std::complex<double>(0.0, 0.0);

  if (e == 0.0)
    return;

  // p0 = Itotal (through abs)
  dV[0] = std::complex<double>(_dItotal_dp0 * e, 0.0);

  // p1 = sigma (through abs)
  if (_sigma_param > 0.0)
  {
    const double dlogV_dsigma = -(ru*ru*sa2 + rv*rv*sb2) / _sigma_param;
    dV[1] = V * (_dsigma_dp1 * dlogV_dsigma);
  }

  // p2 = A (through clamp)
  {
    const double denom_p = std::max(1.0e-300, 1.0 + _A_param);
    const double denom_m = std::max(1.0e-300, 1.0 - _A_param);

    const double dlogV_dA =
        0.5 * ru*ru * sa2 / denom_p
      - 0.5 * rv*rv * sb2 / denom_m;

    dV[2] = V * (_dA_dp2 * dlogV_dA);
  }

  // p3 = PA
  {
    const double dlogV_dpa = ru * rv * (sa2 - sb2);
    dV[3] = V * dlogV_dpa;
  }
}
  

  
double model_image_asymmetric_gaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;

    double exponent = - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) );
    double VM = _Itotal * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );

    return ( VM );
  }
  else
  {
    return ( model_image::visibility_amplitude(d,acc) );
  }
}

double model_image_asymmetric_gaussian::closure_amplitude(datum_closure_amplitude& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    // Obtain counter-rotated u,v coordinates
    double u[]={d.u1,d.u2,d.u3,d.u4}, v[]={d.v1,d.v2,d.v3,d.v4};
    double ru, rv, exponent, VM[4];
    double c=std::cos(_position_angle), s=std::sin(_position_angle);
    for (int j=0; j<4; ++j)
    {
      ru = u[j]*c + v[j]*s;
      rv = -u[j]*s + v[j]*c;

      ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
      rv *= 2.*M_PI;

      exponent = - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) );
      VM[j] = (exponent<-200.0 ? 0.0 : std::exp( exponent ) );
    }
    return ( (VM[0]*VM[2])/ (VM[1]*VM[3]) );
  }
  else
  {
    return ( model_image::closure_amplitude(d,acc) );
  }
}

std::complex<double> model_image_asymmetric_gaussian::visibility(datum_visibility& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;

    double exponent = - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) );
    double VM = _Itotal * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );

    return ( std::complex<double>(VM,0) );
  }
  else
  {
    return ( model_image::visibility(d,acc) );
  }
}
  

};

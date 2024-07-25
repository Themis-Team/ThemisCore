 /*!
  \file model_image_symmetric_gaussian.cpp
  \author Avery Broderick
  \date  November, 2018
  \brief Implements symmetric Gaussian image class.
  \details To be added
*/

#include "model_image_symmetric_gaussian.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>

namespace Themis {

model_image_symmetric_gaussian::model_image_symmetric_gaussian()
  : _use_analytical_visibilities(true)
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Creating model_image_symmetric_gaussian in rank " << world_rank << std::endl;
}

void model_image_symmetric_gaussian::use_numerical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

  _use_analytical_visibilities = false;
}

void model_image_symmetric_gaussian::use_analytical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;

  _use_analytical_visibilities = true;
}

void model_image_symmetric_gaussian::generate_model(std::vector<double> parameters)
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

  
void model_image_symmetric_gaussian::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  int Ntot = 128;
  double Ns = 4.0;

  _Itotal = std::fabs(parameters[0]);
  _sigma = std::fabs(parameters[1]);

  if (_use_analytical_visibilities==false)
  {
    double Inorm = _Itotal/(_sigma*_sigma*2*M_PI);
    
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
	alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_sigma*2.0*Ns/double(Ntot));
	beta[j][k] = ((double(k)-0.5*double(Ntot)+0.5)*_sigma*2.0*Ns/double(Ntot));
	exponent = - 0.5 * ( alpha[j][k]*alpha[j][k]/(_sigma*_sigma)  + beta[j][k]*beta[j][k]/(_sigma*_sigma) );
	I[j][k] = Inorm * ( exponent<-200.0 ? 0.0 : std::exp(exponent) );
      }
    }
  }
}
  
double model_image_symmetric_gaussian::closure_phase(datum_closure_phase& d, double acc)
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

double model_image_symmetric_gaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;

    double exponent = - 0.5 * ( (ru*ru+rv*rv)*(_sigma*_sigma) );
    double VM = _Itotal * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );

    return ( VM );
  }
  else
  {
    return ( model_image::visibility_amplitude(d,acc) );
  }
}

double model_image_symmetric_gaussian::closure_amplitude(datum_closure_amplitude& d, double acc)
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

      exponent = - 0.5 * ( (ru*ru+rv*rv)*(_sigma*_sigma) );
      VM[j] = (exponent<-200.0 ? 0.0 : std::exp( exponent ) );
    }
    return ( (VM[0]*VM[2])/ (VM[1]*VM[3]) );
  }
  else
  {
    return ( model_image::closure_amplitude(d,acc) );
  }
}

std::complex<double> model_image_symmetric_gaussian::visibility(datum_visibility& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;

    double exponent = - 0.5 * ( (ru*ru+rv*rv)*(_sigma*_sigma) );
    double VM = _Itotal * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );

    return ( std::complex<double>(VM,0) );
  }
  else
  {
    return ( model_image::visibility(d,acc) );
  }
}
  

};

/*!
  \file model_image_themaussian.cpp
  \author Boris Georgiev
  \date  February, 2022
  \brief Implements Themaussian image class. A Themaussian is a sum of flux-ordered gaussians..
  \details To be added
*/

#include "model_image_themaussian.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>

namespace Themis {

model_image_themaussian::model_image_themaussian(size_t Num_gauss)
  : _Num_gauss(Num_gauss), _Itotal(_Num_gauss, 0.0), _sigma_alpha(_Num_gauss, 0.0), _sigma_beta(_Num_gauss, 0.0), _PA(_Num_gauss, 0.0), _xpos(_Num_gauss, 0.0), _ypos(_Num_gauss, 0.0), _use_analytical_visibilities(true)
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Creating model_image_themaussian in rank " << world_rank << std::endl;
  //std::cout << "Creating model_image_asymmetric_gaussian in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
}

void model_image_themaussian::use_numerical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;
  //std::cout << "Using numerical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
  _use_analytical_visibilities = false;
}

void model_image_themaussian::use_analytical_visibilities()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
  //std::cout << "Using analytical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;

  _use_analytical_visibilities = true;
}

void model_image_themaussian::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  //_Itotal = std::fabs(parameters[0]);
  _Itotal[0]=std::fabs(parameters[0]);
  _xpos[0]=parameters[4];
  _ypos[0]=parameters[5];
  for (size_t i=1; i<_Num_gauss; i++)
  {
    _Itotal[i]=std::fabs(_Itotal[i-1]*parameters[6*i]);
    _xpos[i]=_xpos[0]+parameters[4+6*i];
    _ypos[i]=_ypos[0]+parameters[5+6*i];
  }
  for (size_t i=0; i<_Num_gauss; i++)
  {
    //parameters[1] = std::fabs(parameters[1]);
    //parameters[2] = std::min(std::max(parameters[2],0.0),0.99);
    //_sigma_beta = parameters[1] * std::sqrt( 1.0 / (1.0-parameters[2]) ); // Major axis
    //_sigma_alpha  = parameters[1] * std::sqrt( 1.0 / (1.0+parameters[2]) ); // Minor axis
    parameters[1+6*i]=std::fabs(parameters[1+6*i]);
    parameters[2+6*i] = std::min(std::max(parameters[2+6*i],0.0),0.99);
    _sigma_beta[i]  = parameters[1+6*i] * std::sqrt( 1.0 / (1.0-parameters[2+6*i]) ); // Major axis
    _sigma_alpha[i] = parameters[1+6*i] * std::sqrt( 1.0 / (1.0+parameters[2+6*i]) ); // Minor axis

    _PA[i]=parameters[3+6*i];
  }
 
  int Ntot = 128;
  double Ns = 4.0;

  if (_use_analytical_visibilities==false)
  {
  
    //double Inorm = _Itotal/(_sigma_alpha*_sigma_beta*2*M_PI);
    std::vector<double> Inorm; 
    for (size_t i=0; i<_Num_gauss; i++)
    {
      Inorm[i]=_Itotal[i]/(_sigma_alpha[i]*_sigma_beta[i]*2*M_PI);

    }

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
	//alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_sigma_alpha*2.0*Ns/double(Ntot));
	//beta[j][k] = ((double(k)-0.5*double(Ntot)+0.5)*_sigma_beta*2.0*Ns/double(Ntot));
	//exponent = - 0.5 * ( alpha[j][k]*alpha[j][k]/(_sigma_alpha*_sigma_alpha)  + beta[j][k]*beta[j][k]/(_sigma_beta*_sigma_beta) );
	//I[j][k] = Inorm * ( exponent<-200.0 ? 0.0 : std::exp(exponent) );
	I[j][k]=0.0;
        for (size_t i=0; i<_Num_gauss; i++)
        {
	  alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_sigma_alpha[i]*2.0*Ns/double(Ntot));
	  beta[j][k] = ((double(k)-0.5*double(Ntot)+0.5)*_sigma_beta[i]*2.0*Ns/double(Ntot));
	  exponent = - 0.5 * ( alpha[j][k]*alpha[j][k]/(_sigma_alpha[i]*_sigma_alpha[i])  + beta[j][k]*beta[j][k]/(_sigma_beta[i]*_sigma_beta[i]) );
	  I[j][k] = I[j][k]+Inorm[i] * ( exponent<-200.0 ? 0.0 : std::exp(exponent) );
        }
      }
    }
  }
}


void model_image_themaussian::generate_model(std::vector<double> parameters)
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


std::complex<double> model_image_themaussian::complex_visibility(double u, double v)
{
  double VM_real = 0.0;
  double VM_imag = 0.0;
  for (size_t i=0; i<_Num_gauss; i++)
  {
    double ru = u*std::cos(_PA[i]) + v*std::sin(_PA[i]);
    double rv = -u*std::sin(_PA[i]) + v*std::cos(_PA[i]);

    ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    rv *= 2.*M_PI;

    double exponent = - 0.5 * ( ru*ru*(_sigma_alpha[i]*_sigma_alpha[i])  + rv*rv*(_sigma_beta[i]*_sigma_beta[i]) );
    double VM_current = _Itotal[i] * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );

    //There is a flip in _xpos due to convention, so that's why it has a minus sign. Like this, it is consistent
    //with model_image_sum
    double shift_factor=-2.*M_PI*(-u*_xpos[i]+v*_ypos[i]);
       
    //The only change is to multiply by Exp(-2 pi i (u x0+y v0))
    VM_real = VM_real + VM_current*std::cos(shift_factor);
    VM_imag = VM_imag + VM_current*std::sin(shift_factor);
  } 
    //return ( std::complex<double>(VM,0) ); 
    return ( std::complex<double>(VM_real,VM_imag) ); 

}

double model_image_themaussian::closure_phase(datum_closure_phase& d, double acc)
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

double model_image_themaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    return ( std::abs(complex_visibility(d.u, d.v)) );
    //double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
    //double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

    //ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    //rv *= 2.*M_PI;

    //double exponent = - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) );
    //double VM = _Itotal * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );
  }
  else
  {
    return ( model_image::visibility_amplitude(d,acc) );
  }
}

double model_image_themaussian::closure_amplitude(datum_closure_amplitude& d, double acc)
{
  if (_use_analytical_visibilities)
  {
    // Obtain counter-rotated u,v coordinates
    //double u[]={d.u1,d.u2,d.u3,d.u4}, v[]={d.v1,d.v2,d.v3,d.v4};
    //double ru, rv, exponent, VM[4];
    //double c=std::cos(_position_angle), s=std::sin(_position_angle);
    //for (int j=0; j<4; ++j)
    //{
    //  ru = u[j]*c + v[j]*s;
    //  rv = -u[j]*s + v[j]*c;

    //  ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
    //  rv *= 2.*M_PI;

    //  exponent = - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) );
    //  VM[j] = (exponent<-200.0 ? 0.0 : std::exp( exponent ) );
    //}
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
  
std::complex<double> model_image_themaussian::visibility(datum_visibility& d, double acc)
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


//std::complex<double> model_image_themaussian::visibility(datum_visibility& d, double acc)
//{
//  if (_use_analytical_visibilities)
//  {
//    double VM_real = 0;
//    double VM_imag = 0;
//    for (size_t i=0; i<_Num_gauss; i++)
//    {
//      double ru = d.u*std::cos(_PA[i]) + d.v*std::sin(_PA[i]);
//      double rv = -d.u*std::sin(_PA[i]) + d.v*std::cos(_PA[i]);
//
//      ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
//      rv *= 2.*M_PI;
//
//      double exponent = - 0.5 * ( ru*ru*(_sigma_alpha[i]*_sigma_alpha[i])  + rv*rv*(_sigma_beta[i]*_sigma_beta[i]) );
//      double VM_current = _Itotal[i] * (exponent<-200.0 ? 0.0 : std::exp( exponent ) );
//      double shift_factor=-2.*M_PI*(d.u*_xpos[i]+d.v*_ypos[i]);
//       
//      //The only change is to multiply by Exp(-2 pi i (u x0+y v0))
//      VM_real = VM_real + VM_current*std::cos(shift_factor);
//      VM_imag = VM_imag + VM_current*std::sin(shift_factor);
//    } 
//    //return ( std::complex<double>(VM,0) ); 
//    return ( std::complex<double>(VM_real,VM_imag) ); 
//  }
//  else
//  {
//    return ( model_image::visibility(d,acc) );
//  }
//}
//  

};

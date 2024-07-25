/*!
  \file model_image_gaussian.cpp
  \author Roman Gold, Avery Broderick
  \date  April, 2017
  \brief Implements the asymmetric Gaussian image class.
  \details To be added
*/

#include "model_image_gaussian.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>

namespace Themis {

  model_image_gaussian::model_image_gaussian()
    : _use_analytical_visibilities(true)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_image_gaussian in rank " << world_rank << std::endl;
    //std::cout << "Creating model_image_gaussian in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
  }

  void model_image_gaussian::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;
    //std::cout << "Using numerical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;

    _use_analytical_visibilities = false;
  }
  
  void model_image_gaussian::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
    //std::cout << "Using analytical visibilities in rank " << MPI::COMM_WORLD.Get_rank() << std::endl;
    
    _use_analytical_visibilities = true;
  }

  void model_image_gaussian::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    int Ntot = 128;
    double Ns = 4.0; // Number of sigmas

    _Itotal = parameters[0];
    _sigma_alpha = parameters[1];
    _sigma_beta = parameters[2];

    /*
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "model_image_gaussian::generate_image : "
  	    << std::setw(4) << world_rank
  	    << std::setw(15) << _Itotal
  	    << std::setw(15) << _sigma_alpha
  	    << std::setw(15) << _sigma_beta
  	    << std::endl;
    */

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
    for (size_t j=0; j<alpha.size(); j++)
    {
      for (size_t k=0; k<alpha[j].size(); k++)
      {
        alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_sigma_alpha*2.0*Ns/double(Ntot));
        beta[j][k] = ((double(k)-0.5*double(Ntot)+0.5)*_sigma_beta*2.0*Ns/double(Ntot));
        I[j][k] = Inorm * std::exp( - 0.5 * ( alpha[j][k]*alpha[j][k]/(_sigma_alpha*_sigma_alpha)  + beta[j][k]*beta[j][k]/(_sigma_beta*_sigma_beta) ));
      }
    }
  }

  double model_image_gaussian::closure_phase(datum_closure_phase& d, double acc)
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

  double model_image_gaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
      double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);
       
      ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
      rv *= 2.*M_PI;
      
      double VM = _Itotal * std::exp( - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) ));
      
      return ( VM );
    }
    else
    {
      return ( model_image::visibility_amplitude(d,acc) );
    }
  }


  std::complex<double> model_image_gaussian::visibility(datum_visibility& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
      double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);
       
      ru *= -2.*M_PI; //Reflection to have the image look the way it's seen in the sky
      rv *= 2.*M_PI;
      
      double VM = _Itotal * std::exp( - 0.5 * ( ru*ru*(_sigma_alpha*_sigma_alpha)  + rv*rv*(_sigma_beta*_sigma_beta) ));
      
      return ( std::complex<double>(VM,0) );
    }
    else
    {
      return ( model_image::visibility(d,acc) );
    }
  }

  
};

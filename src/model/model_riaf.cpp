/*!
  \file model_riaf.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements extended RIAF image+flux model class.
  \details To be added
*/

#include "model_riaf.h"
#include <cmath>

namespace Themis {

model_riaf::model_riaf(std::vector<double> frequencies, double M, double D)
  : _generated_model(false)
{
  for (size_t i=0; i<frequencies.size(); ++i)
    _riaf_images.push_back(new model_image_riaf(frequencies[i],M,D));
}

model_riaf::~model_riaf()
{
  for (size_t i=0; i<_riaf_images.size(); ++i)
    delete _riaf_images[i];
}

void model_riaf::set_image_resolution(int Nray)
{
  for (size_t i=0; i<_riaf_images.size(); ++i)
    _riaf_images[i]->set_image_resolution(Nray);
}
  
void model_riaf::set_mpi_communicator(MPI_Comm comm)
{
  for (size_t i=0; i<_riaf_images.size(); ++i)
    _riaf_images[i]->set_mpi_communicator(comm);
}

void model_riaf::generate_model(std::vector<double> parameters)
{
  _parameters = parameters;
  _generated_model = true;
}

double model_riaf::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _riaf_images[ii]->generate_model(_parameters);
    return ( _riaf_images[ii]->visibility_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_riaf::visibility_amplitude : Must generate model before visibility_amplitude\n";
    std::exit(1);
  }
}

double model_riaf::closure_phase(datum_closure_phase& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _riaf_images[ii]->generate_model(_parameters);
    return ( _riaf_images[ii]->closure_phase(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_riaf::visibility_amplitude : Must generate model before closure_phase\n";
    std::exit(1);
  }
}

double model_riaf::closure_amplitude(datum_closure_amplitude& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _riaf_images[ii]->generate_model(_parameters);
    return ( _riaf_images[ii]->closure_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_riaf::closure_amplitude : Must generate model before closure_amplitude\n";
    std::exit(1);
  }
}
  
double model_riaf::flux(datum_flux& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    //    _riaf_images[ii]->generate_model(_parameters);
    return ( _riaf_images[ii]->generate_flux_estimate(accuracy,_parameters) );
  } 
  else 
  {
    std::cerr << "model_riaf::visibility_amplitude : Must generate model before fluxes\n";
    std::exit(1);
  }
}

size_t model_riaf::find_frequency_index(double frequency) const
{
  size_t imin=0;
  double dfmin=0;
  for (size_t i=0; i<_riaf_images.size(); ++i)
  {
    double df = std::fabs(std::log(frequency/_riaf_images[i]->frequency()));
    if (i==0 || df<dfmin)
    {
      imin=i;
      dfmin=df;
    }
  }
  return imin;
}

};

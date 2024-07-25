/*!
  \file model_force_free_jet.cpp
  \author Paul Tiede
  \date  Sept, 2018
  \brief Implements extended force free jet image+flux model class.
  \details To be added
*/

#include "model_force_free_jet.h"
#include <cmath>

namespace Themis {

model_force_free_jet::model_force_free_jet(std::vector<double> frequencies)
  : _generated_model(false)
{
  for (size_t i=0; i<frequencies.size(); ++i)
    _ffj_images.push_back(new model_image_force_free_jet(frequencies[i]));
}

model_force_free_jet::~model_force_free_jet()
{
  for (size_t i=0; i<_ffj_images.size(); ++i)
    delete _ffj_images[i];
}

void model_force_free_jet::set_image_resolution(int Nray, int number_of_refines)
{
  for (size_t i=0; i<_ffj_images.size(); ++i)
    _ffj_images[i]->set_image_resolution(Nray, number_of_refines);
}
  
void model_force_free_jet::set_screen_size(double Rmax)
{
  for (size_t i=0; i<_ffj_images.size(); ++i)
    _ffj_images[i]->set_screen_size(Rmax);
}
void model_force_free_jet::set_mpi_communicator(MPI_Comm comm)
{
  for (size_t i=0; i<_ffj_images.size(); ++i)
    _ffj_images[i]->set_mpi_communicator(comm);
}

void model_force_free_jet::generate_model(std::vector<double> parameters)
{
  _parameters = parameters;
  _generated_model = true;
}

double model_force_free_jet::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _ffj_images[ii]->generate_model(_parameters);
    return ( _ffj_images[ii]->visibility_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_force_free_jet::visibility_amplitude : Must generate model before visibility_amplitude\n";
    std::exit(1);
  }
}

double model_force_free_jet::closure_phase(datum_closure_phase& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _ffj_images[ii]->generate_model(_parameters);
    return ( _ffj_images[ii]->closure_phase(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_force_free_jet::visibility_amplitude : Must generate model before closure_phase\n";
    std::exit(1);
  }
}

double model_force_free_jet::closure_amplitude(datum_closure_amplitude& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    _ffj_images[ii]->generate_model(_parameters);
    return ( _ffj_images[ii]->closure_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_force_free_jet::closure_amplitude : Must generate model before closure_amplitude\n";
    std::exit(1);
  }
}
  
double model_force_free_jet::flux(datum_flux& d, double accuracy)
{
  size_t ii=find_frequency_index(d.frequency);
  if (_generated_model)
  {
    //    _ffj_images[ii]->generate_model(_parameters);
    return ( _ffj_images[ii]->generate_flux_estimate(accuracy,_parameters) );
  } 
  else 
  {
    std::cerr << "model_force_free_jet::visibility_amplitude : Must generate model before fluxes\n";
    std::exit(1);
  }
}

size_t model_force_free_jet::find_frequency_index(double frequency) const
{
  size_t imin=0;
  double dfmin=0;
  for (size_t i=0; i<_ffj_images.size(); ++i)
  {
    double df = std::fabs(std::log(frequency/_ffj_images[i]->frequency()));
    if (i==0 || df<dfmin)
    {
      imin=i;
      dfmin=df;
    }
  }
  return imin;
}

};

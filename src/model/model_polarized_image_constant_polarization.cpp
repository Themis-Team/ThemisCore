/*!
  \file model_polarized_image_constant_polarization.cpp
  \author Avery E. Broderick
  \date  March, 2020
  \brief Header file for a polarized image model class with a constant polarization applied to a model image.
  \details To be added
*/

#include "model_polarized_image_constant_polarization.h"
#include <cmath>
#include <valarray>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <fstream>

#include <ctime>

namespace Themis {

  model_polarized_image_constant_polarization::model_polarized_image_constant_polarization(model_image& intensity_model)
    : _intensity_model(intensity_model)
  {
    _size = _intensity_model.size()+3;
  }

  model_polarized_image_constant_polarization::~model_polarized_image_constant_polarization()
  {
  }

  void model_polarized_image_constant_polarization::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    _intensity_model.generate_model(parameters);
    get_image(alpha,beta,I);
  }

  void model_polarized_image_constant_polarization::generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    generate_image(parameters,I,alpha,beta);

    Q=I;
    U=I;
    V=I;

    double smu = std::sqrt( 1.0 - _polarization_mu*_polarization_mu );

    double qfac = _polarization_fraction * std::cos(2.0*_polarization_EVPA) * smu;
    double ufac = _polarization_fraction * std::sin(2.0*_polarization_EVPA) * smu;
    double vfac = _polarization_fraction * _polarization_mu;

    for (size_t i=0; i<I.size(); ++i)
      for (size_t j=0; j<I[i].size(); ++j)
      {
	Q[i][j] *= qfac;
	U[i][j] *= ufac;
	V[i][j] *= vfac;
      }
  }

  void model_polarized_image_constant_polarization::generate_model(std::vector<double> parameters)
  {
    // Read and strip off Dterm parameters
    read_and_strip_Dterm_parameters(parameters);

    // Takes the polarization properties, and strip them off (in reverse order!)
    _polarization_mu = parameters.back();
    parameters.pop_back();
    _polarization_EVPA = parameters.back();
    parameters.pop_back();
    _polarization_fraction = parameters.back();
    parameters.pop_back();

    // Make intensity model
    _intensity_model.generate_model(parameters);
  }

  std::string model_polarized_image_constant_polarization::model_tag() const
  {
    std::stringstream tag;

    tag << "model_polarized_image_constant_polarization " << _modeling_Dterms;
    if (_modeling_Dterms)
      for (size_t j=0; j<_station_codes.size(); ++j)
	tag << " " << _station_codes[j];
    tag << "\n";
    tag << "SUBTAG START\n";
    tag << _intensity_model.model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }
  
  std::vector< std::complex<double> > model_polarized_image_constant_polarization::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
  {
    datum_visibility dI(d.u,d.v,0.0,1.0,d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    std::complex<double> VI = _intensity_model.visibility(dI,accuracy);

    double smu = std::sqrt(1.0 - _polarization_mu*_polarization_mu);
    
    double qfac = _polarization_fraction * std::cos(2.0*_polarization_EVPA) * smu;
    double ufac = _polarization_fraction * std::sin(2.0*_polarization_EVPA) * smu;
    double vfac = _polarization_fraction * _polarization_mu;

    std::complex<double> VQ = qfac*VI;
    std::complex<double> VU = ufac*VI;
    std::complex<double> VV = vfac*VI;

    // Convert to RR, LL, RL, LR
    std::vector< std::complex<double> > crosshand_vector(4);
    crosshand_vector[0] = VI+VV; // RR
    crosshand_vector[1] = VI-VV; // LL 
    crosshand_vector[2] = VQ+std::complex<double>(0.0,1.0)*VU; // RL
    crosshand_vector[3] = VQ-std::complex<double>(0.0,1.0)*VU; // LR
    
    // Apply Dterms
    apply_Dterms(d,crosshand_vector);
    
    return crosshand_vector;
  }

  std::complex<double> model_polarized_image_constant_polarization::visibility(datum_visibility& d, double accuracy)
  {
    return _intensity_model.visibility(d,accuracy);
  }

  double model_polarized_image_constant_polarization::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
  {
    return _intensity_model.visibility_amplitude(d,accuracy);
  }

  double model_polarized_image_constant_polarization::closure_phase(datum_closure_phase& d, double accuracy)
  {
    return _intensity_model.closure_phase(d,accuracy);
  }

  double model_polarized_image_constant_polarization::closure_amplitude(datum_closure_amplitude& d, double accuracy)
  {
    return _intensity_model.closure_amplitude(d,accuracy);
  }


  void model_polarized_image_constant_polarization::set_mpi_communicator(MPI_Comm comm)
  {
    _intensity_model.set_mpi_communicator(comm);
  }
  

};

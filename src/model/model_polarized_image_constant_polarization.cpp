/*!
  \file model_polarized_image_constant_polarization.cpp
  \author Avery E. Broderick
  \date  March, 2020
  \brief Header file for a polarized image model class with a constant polarization applied to a model image.
  \details To be added
*/

#include "model_polarized_image_constant_polarization.h"
#include "model_image_asymmetric_gaussian.h"
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



bool model_polarized_image_constant_polarization::analytic_asym_gaussian_supported() const
{
  return (dynamic_cast<const model_image_asymmetric_gaussian*>(&_intensity_model) != nullptr);
}

void model_polarized_image_constant_polarization::visibility_and_derivatives(
    datum_crosshand_visibilities& d,
    std::complex<double>* out,
    std::complex<double> deriv[7][4]) const
{
  for (int q = 0; q < 7; ++q)
    for (int h = 0; h < 4; ++h)
      deriv[q][h] = std::complex<double>(0.0, 0.0);

  const model_image_asymmetric_gaussian* ag =
    dynamic_cast<const model_image_asymmetric_gaussian*>(&_intensity_model);

  if (!ag)
  {
    out[0] = out[1] = out[2] = out[3] = std::complex<double>(0.0,0.0);
    return;
  }

  datum_visibility dI(d.u, d.v, 0.0, 1.0, d.frequency, d.tJ2000, d.Station1, d.Station2, d.Source);

  std::complex<double> VI;
  std::complex<double> dVI[4];
  ag->visibility_and_derivatives(dI, VI, dVI);

  const double pf  = _polarization_fraction;
  const double ev  = _polarization_EVPA;
  const double mu  = _polarization_mu;

  const double omm = std::max(1.0e-300, 1.0 - mu*mu);
  const double smu = std::sqrt(omm);

  const double c2 = std::cos(2.0*ev);
  const double s2 = std::sin(2.0*ev);

  const double qfac = pf * c2 * smu;
  const double ufac = pf * s2 * smu;
  const double vfac = pf * mu;

  out[0] = VI + vfac*VI;                                   // RR
  out[1] = VI - vfac*VI;                                   // LL
  out[2] = qfac*VI + std::complex<double>(0.0,1.0)*ufac*VI; // RL
  out[3] = qfac*VI - std::complex<double>(0.0,1.0)*ufac*VI; // LR

  // intensity-model params 0..3
  for (int k = 0; k < 4; ++k)
  {
    deriv[k][0] = (1.0 + vfac) * dVI[k];
    deriv[k][1] = (1.0 - vfac) * dVI[k];
    deriv[k][2] = (qfac + std::complex<double>(0.0,1.0)*ufac) * dVI[k];
    deriv[k][3] = (qfac - std::complex<double>(0.0,1.0)*ufac) * dVI[k];
  }

  // p4 = polarization fraction
  {
    const double dq = c2 * smu;
    const double du = s2 * smu;
    const double dv = mu;

    deriv[4][0] = dv * VI;
    deriv[4][1] = -dv * VI;
    deriv[4][2] = (dq + std::complex<double>(0.0,1.0)*du) * VI;
    deriv[4][3] = (dq - std::complex<double>(0.0,1.0)*du) * VI;
  }

  // p5 = EVPA
  {
    const double dq = pf * (-2.0*s2) * smu;
    const double du = pf * ( 2.0*c2) * smu;

    deriv[5][0] = std::complex<double>(0.0,0.0);
    deriv[5][1] = std::complex<double>(0.0,0.0);
    deriv[5][2] = (dq + std::complex<double>(0.0,1.0)*du) * VI;
    deriv[5][3] = (dq - std::complex<double>(0.0,1.0)*du) * VI;
  }

  // p6 = mu
  {
    const double dq = -(mu/omm) * qfac;
    const double du = -(mu/omm) * ufac;
    const double dv = pf;

    deriv[6][0] = dv * VI;
    deriv[6][1] = -dv * VI;
    deriv[6][2] = (dq + std::complex<double>(0.0,1.0)*du) * VI;
    deriv[6][3] = (dq - std::complex<double>(0.0,1.0)*du) * VI;
  }
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

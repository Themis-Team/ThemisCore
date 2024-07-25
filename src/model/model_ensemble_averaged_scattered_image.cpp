/*!
    \file model_ensemble_averaged_scattered_image.cpp  
    \author Roman Gold
    \date Apr 2017
    \brief Header file for ensemble averaged scattering interface.
    \details 
*/

#include "model_ensemble_averaged_scattered_image.h"
#include <cmath>

namespace Themis {

model_ensemble_averaged_scattered_image::model_ensemble_averaged_scattered_image(model_visibility_amplitude& model)
  : _model(model) 
{
}

void model_ensemble_averaged_scattered_image::generate_model(std::vector<double> parameters)
{
  _model.generate_model(parameters);
}

void model_ensemble_averaged_scattered_image::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}


  /*
void model_ensemble_averaged_scattered_image::get_unscattered_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  _model.get_image(alpha,beta,I);
}

void model_ensemble_averaged_scattered_image::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  I.clear(); alpha.clear(); beta.clear();
}
  */
  

double model_ensemble_averaged_scattered_image::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{

  // Major and minor scattering axes in mas cm^-2
  double major_FWHM_norm = 1.309;
  double minor_FWHM_norm = 0.64;

  // Position angle relative to East of the minor axis (Major axis is 78 degrees East of North)
//	double PAscat_minor_rad = 78.0 * M_PI/180.0; // If u goes from E to W as it increases, this is backwards!
  double PAscat_minor_rad = -78.0 * M_PI/180.0;	//in radians

  // The properly normalized and major and minor scattering sigmas (in rad)
  // the 1e3*... is to convert milliarcseconds to rads?
  double sigscat_major = (major_FWHM_norm/(2.0*std::sqrt(std::log(4.0)))) * std::pow((d.wavelength),2.0) / (1e3*3600.0*180.0/M_PI);
  double sigscat_minor = (minor_FWHM_norm/(2.0*std::sqrt(std::log(4.0)))) * std::pow((d.wavelength),2.0) / (1e3*3600.0*180.0/M_PI);
  // The associated sigmas in the u-v plane (in m)
  double uv_sigscat_major = 1.0/(2.0*M_PI*sigscat_major); //* lambda;		//we want it in units of lambda
  double uv_sigscat_minor = 1.0/(2.0*M_PI*sigscat_minor); //* lambda;		//we want it in units of lambda

  // Vij = V_in[i][j]*exp(-pow(uv_major/uv_sigscat_major,2.0)/2.0
  // 		       -pow(uv_minor/uv_sigscat_minor,2.0)/2.0)

  double uv_minor = d.u*std::cos(PAscat_minor_rad) + d.v * std::sin(PAscat_minor_rad);
  double uv_major = -d.u * std::sin(PAscat_minor_rad) + d.v*std::cos(PAscat_minor_rad) ;
  double scattering_kernel = std::exp(-0.5*std::pow(uv_major/uv_sigscat_major,2) - 0.5*std::pow(uv_minor/uv_sigscat_minor,2));

  // TODO:
  // Using interpolation class, bicubic, maybe replace with type 1
  // Final two parameters are extraneous and should be stripped out, relating to the
  // Galactic center scattering screen, which now is dealt with elsewhere
  return ( scattering_kernel*_model.visibility_amplitude(d, acc) );
}


  /*
double model_ensemble_averaged_scattered_image::closure_phase(datum_closure_phase& d, double acc)
{
  
  // TODO:
  // Final two parameters are extraneous and should be stripped out, relating to the
  // Galactic center scattering screen, which now is dealt with elsewhere
  // Not sure why it says "compute_closure_phase" and not simply "closure_phase", eh, like it 
  // does with visibility_amplitudes
  return ( _model.closure_phase(d,acc) );
}
  */


};

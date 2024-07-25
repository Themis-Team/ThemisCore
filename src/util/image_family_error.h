/*!
  \file image_family_error.h
  \author Avery E. Broderick
  \date  December, 2018
  \brief Header file for a utility class that generates an approximation of the systemtic error as measured by the variance among images.
*/

#ifndef Themis_image_family_systematics_generator_H_
#define Themis_image_family_systematics_generator_H_


#include "model_image_score.h"
#include "interpolator1D.h"
#include "data_visibility.h"
#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_closure_amplitude.h"

#include <vector>
#include <string>

namespace Themis {
  
  /*!
    \brief Defines a utility class that constructs and incorporates a theory error estimate from a library of images.

    \details Given a list of image file names appropriate for the model_image_score class, the variance of the visibility amplitudes are computed at a set of radial and radial locations.  These are then added in quadrature to the existing error budget for a given data object.  Note that these errors are static, i.e., they are set at construction time and do not vary subsequently.  

    \warning Only visibility amplitude and closure phase are implemented.  Others are trivial but not yet done.
    \todo Implement complex visibility and closure amplitudes.
*/
  class image_family_static_error {
   public:
    
    //! Defines an image_family_static_error object without an error esitmate yet.
    image_family_static_error();

    //! Defines an image_family_static_error object with an error estimate.  Identical to the default constructor followed by generate_error_estimates(...).
    image_family_static_error(std::vector<std::string> image_file_name_list, std::string README_file_name, std::vector<double> p, size_t Nr=128, size_t Nphi=64, double umax=10.0);

    //! Destructor
    ~image_family_static_error();

    //! Generates the error esitmates given a list of image file names, the README file name, a set of parameters appropriate for model_image_score.  Can modify the number of radial samples, azimuthal samples, and range of baseline lengths (in Glambda) sampled.
    void generate_error_estimates(std::vector<std::string> image_file_name_list, std::string README_file_name, std::vector<double> p, size_t Nr=128, size_t Nphi=64, double umax=10.0);

    //! Sets the error estimate to assume a constant visibility amplitude variance and fractional visibility amplitude variance.
    void use_constant_approximation();

    //! Sets the error estimate to assume an asymmetric, but radially varying, visibility amplitude variance and fractional visibility amplitude variance.
    void use_axisymmetric_approximation();

    //! Returns a data_visiblity object with the theory error estimate added in quadrature. (NOT IMPLEMENTED)
    Themis::data_visibility& data_visibility(data_visibility& d);

    //! Returns a data_visiblity_amplitude object with the theory error estimate added in quadrature.
    Themis::data_visibility_amplitude& data_visibility_amplitude(data_visibility_amplitude& d);

    //! Returns a data_closure_phase object with the theory error estimate added in quadrature.
    Themis::data_closure_phase& data_closure_phase(data_closure_phase& d);

    //! Returns a data_closure_amplitude object with the theory error estimate added in quadrature. (NOT IMPLEMENTED)
    Themis::data_closure_amplitude& data_closure_amplitude(data_closure_amplitude& d);
    
   private:
    int _approximation_type; // 0 - constant, 1 - axisymmetric, that's it.
    double _const_va_var, _const_va_frac_var;
    Interpolator1D _vis_amp_var;
    Interpolator1D _vis_amp_frac_var;

    Themis::data_visibility *_data_v;
    Themis::data_visibility_amplitude *_data_va;
    Themis::data_closure_phase *_data_cp;
    Themis::data_closure_amplitude *_data_ca;
  };

};

#endif

 /*!
  \file model_image_asymmetric_gaussian.h
  \author Avery Broderick & Hung-Yi Pu
  \date  December, 2017
  \brief Header file for a wavelength-dependent, asymmetric Gaussian image class, suitable for measuring the diffractive scattering Kernel of Sgr A*.
  \details To be added
*/

#ifndef Themis_MODEL_GAUSSIAN_SCATTERING_KERNEL_H_
#define Themis_MODEL_GAUSSIAN_SCATTERING_KERNEL_H_

#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "model_image_asymmetric_gaussian.h"

#include <vector>

namespace Themis {
  
  /*!
    \brief Defines an interface for a wavelength-dependent, asymmetric Gaussian image model for the semi-major/minor axes and position angle, all of which are assumed to have a power-law dependence on wavelength.  The pivot frequency may be set, but is 230GHz by default.

    \details The model assumes that the intrinsic source is a point source, implying that as \f$\lambda\rightarrow0\f$, the major/minor axes sizes vanish.  Note that no such condition applies for the position angle, which requires a zero-wavelength value.

  Parameter list:\n
    - parameters[0] ... Pivot frequency normalization of \f$\sigma_\alpha\f$
    - parameters[1] ... Power-law of \f$\lambda\f$ dependence of \f$\sigma_\alpha\f$

    - parameters[2] ... Pivot frequency normalization of \f$\sigma_\beta\f$
    - parameters[3] ... Power-law of \f$\lambda\f$ dependence of \f$\sigma_\beta\f$

    - parameters[4] ... Zero-wavelength position angle
    - parameters[5] ... Pivot frequency normalization of position angle (in model_image) in radians
    - parameters[6] ... Power-law of \f$\lambda\f$ dependence of position angle shift\n

    \warning Note that the \f$\sigma_{\alpha,\beta}\f$ are degenerate with the swapped configuration rotated by \f$\pi/2\f$ radians.  Thus, the orientation must limited to between \f$0\f$ and \f$\pi/2\f$.
  */
  class model_gaussian_scattering_kernel : public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
  {
    public:
    model_gaussian_scattering_kernel(std::vector<double> frequencies, double pivot_frequency=230e9);
    virtual ~model_gaussian_scattering_kernel() {};

    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return 7; };

    //! Currently simply saves model parameters. Repeat model production is prevented within the model_image_assymetric_gaussian objects.
    virtual void generate_model(std::vector<double> parameters);
       
    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
    virtual double closure_phase(datum_closure_phase& d, double acc);
      
    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_amplitude(datum_closure_amplitude& d, double acc);


    private:
    std::vector<double> _frequencies;
    std::vector< model_image_asymmetric_gaussian > _images;    

    size_t find_frequency_index(double frequency) const;

    std::vector<double> _parameters;
    bool _generated_model;

    double _pivot_frequency;

    std::vector<double> generate_asymmetric_gaussian_parameter_list(double frequency) const;
  };
  
};
#endif


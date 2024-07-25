/*!
  \file model_image_multi_asymmetric_gaussian.h
  \author Avery Broderick
  \date  October, 2018
  \brief Header file for the Multi-Asymmetric Gaussian image class originally motivated by the model fitting challenge 3 in the MCFE WG.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_MULTI_ASYMMETRIC_GAUSSIAN_H_
#define Themis_MODEL_IMAGE_MULTI_ASYMMETRIC_GAUSSIAN_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a multi-Asymmetric Gaussian image model.

  \details Each Gaussian image component is defined by a size, a
  flux normalization, and two coordinates location components. In
  addition there is a position angle (removed from the end of the
  parameter list by model_image.generate() prior to passing to
  model_image_multi_asymmetric_gaussians.generate_image().)

  Parameter list:\n
  - parameters[0] ... Total, integrated flux of first component in Jy.\n
  - parameters[1] ... Symmetrized std. dev. of first component in radians, i.e., \f$ \sigma = \sqrt{2} \sigma_m \sigma_M / \left( \sigma_M^2 + \sigma_m^2 \right)^{1/2} \f$
  - parameters[2] ... Asymmetry of first component, i.e., \f$ A = (\sigma_M^2-\sigma_m^2)/(\sigma_M^2+\sigma_m^2) \f$ 
  - parameters[3] ... Position angle of first component in radians\n
  - parameters[4] ... x position of first component in radians.\n
  - parameters[5] ... y position of first component in radians.\n
  - parameters[6(N-1)+0] ... Total, integrated flux of Nth component in Jy.\n
  - parameters[6(N-1)+1] ... Symmetrized std. dev. of Nth component in radians.\n
  - parameters[6(N-1)+2] ... Asymmetry of Nth component in radians.\n
  - parameters[6(N-1)+3] ... Position angle of Nth component in radians.\n
  - parameters[6(N-1)+4] ... x position of Nth component in radians.\n
  - parameters[6(N-1)+5] ... y position of Nth component in radians.\n
  - parameters[6N] ... Position angle (in model_image) in radians\n

*/
class model_image_multi_asymmetric_gaussian : public model_image
{
 private:

  //! Generates and returns rectalinear grid of intensities associated with the Gaussian image model in Jy/pixel located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

  //! \brief Calculates the complex visibility amplitude
  std::complex<double> complex_visibility(double u, double v);

 public:
  model_image_multi_asymmetric_gaussian(size_t N);
  virtual ~model_image_multi_asymmetric_gaussian() {};

  //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
  void use_numerical_visibilities();
  //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
  void use_analytical_visibilities();

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return (6*_N+1); };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;
  
  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);
  
  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
  virtual double closure_phase(datum_closure_phase& d, double acc);

  //! \brief Returns closure amplitude computed from the image given a 
  //! datum_closure_phase object, containing all of the accoutrements.  
  //! While this provides access to the actual data value, the two could 
  //! be separated if necessary.  Also takes an accuracy parameter with 
  //! the same units as the data, indicating the accuracy with which the 
  //! model must generate a comparison value.  Note that this can be 
  //! redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double acc);
    

 private:
  const size_t _N; //!< Nr of Gaussian components
  std::vector<double> _Icomp; //!< Internal total intensity.
  std::vector<double> _sigma_min; //!< Minor axis
  std::vector<double> _sigma_maj; //!< Major axis
  std::vector<double> _phi; //!< Position angle
  std::vector<double> _x; //!< X position
  std::vector<double> _y; //!< Y position
  // double position_angle; 

  bool _use_analytical_visibilities; //!< If true uses analytical visibility computation, if false use numerical visibilities.
  
};

};
#endif

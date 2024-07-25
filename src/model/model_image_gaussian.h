/*!
  \file model_image_gaussian.h
  \author Roman Gold, Avery Broderick
  \date  April, 2017
  \brief Header file for the Gaussian image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_GAUSSIAN_H_
#define Themis_MODEL_IMAGE_GAUSSIAN_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines an interface for an asymmetric Gaussian image
  model based on the model_image class.  This is an explicit example
  of a model_image object.

  \details The Gaussian image is defined by a pair of axes sizes, an
  overall flux normalization, and the position angle (removed from the
  end of the parameter list by model_image.generate() prior to passing
  to model_image_gaussian.generate_image().)

  Parameter list:\n
  - parameters[0] ... Total, integrated flux in Jy.\n
  - parameters[1] ... Std. dev. in the fiducial horizontal direction in radians.\n
  - parameters[2] ... Std. dev. in the fiducial vertical direction in radians.\n
  - parameters[3] ... Position angle (in model_image) in radians\n

  \warning Note that the two size parameters are not the semimajor and semiminor axes, leading to a natural degeneracy between swapping the axes sizes and rotating the image by pi/2 radians.
*/
class model_image_gaussian : public model_image
{
 private:
  //! Generates and returns rectalinear grid of intensities associated with the Gaussian image model in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
 public:
  model_image_gaussian();
  virtual ~model_image_gaussian() {};

  //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
  void use_numerical_visibilities();
  //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
  void use_analytical_visibilities();

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return 4; };


  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
  
  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
  virtual double closure_phase(datum_closure_phase& d, double acc);
  
  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const
  {
    return "model_image_gaussian";
  };


 private:
  double _Itotal; //!< Internal total intensity.
  double _sigma_alpha; //!< Std. dev. in fiducial horizontal direction.
  double _sigma_beta; //!< Std. dev. in fiducial vertical direction.

  bool _use_analytical_visibilities; //!< If true uses analytical visibility computation, if false use numerical visibilities.
  
};

};
#endif

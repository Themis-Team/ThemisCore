/*!
  \file model_image_upsample.h
  \author Avery Broderick
  \date May, 2019
  \brief Header file for the model_image_upsample image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_UPSAMPLE_H_
#define Themis_MODEL_IMAGE_UPSAMPLE_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a model image that is upsampled from that provided.  
  Interpolation options may be set to linear, bicubic, or bicubic_spline (default)
  The parameters are identical to those of the supplied model_image object.

  \warning Even upsampling factors appear to rotate the closure phases by 180 degrees for reasons that are currently unclear.  Odd upsampling factors do not share this problem.  bicubic_spline is not properly implemented.
*/
class model_image_upsample : public model_image
{
 private:
  //! Sets the image pixel values
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
 public:
  //! Constructs a model_image_upsample object. Takes a model_image object.
  model_image_upsample(model_image& image, size_t factor, std::string method="bicubic");

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _image.size(); };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);
  

 private:
  model_image& _image;
  size_t _factor;
  std::string _method;

  Themis::Interpolator2D _interp;
};

};
#endif

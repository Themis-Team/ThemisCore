/*!
  \file model_image_polynomial_variable.h
  \author Avery Broderick
  \date  March, 2019
  \brief Header file for the model_image_polynomial_variable class, which generates a variable model by permitting model parameters to vary with time.  All parameters are constrained to vary as polynomial functions of time, though the order of each may differ.  A reference time may be set.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_POLYNOMIAL_VARIABLE_H_
#define Themis_MODEL_IMAGE_POLYNOMIAL_VARIABLE_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a polynomial variable image model given an input image model.

  \details Each parameter of the input model_image is permitted to vary as a polynomial with time.  Different parameters may have different order polynomials.  Constant parameters should have order 0.  A reference time may be specified.

  Parameter list:\n
  - parameters[0] ... Model 1 parameters[0] 0th order term\n
  - parameters[1] ... Model 1 parameters[0] 1st order term\n
  ...\n
  - parameters[n1] ... Model 1 parameters[0] n1th order term\n
  - parameters[n1+1] ... Model 1 parameters[0] 0th order term\n
  ...\n
  - parameters[n1+n2] ... Model 1 parameters[0] n2th order term\n
  ...\n

  Note that for models that require substantial recomputation of the underlying image, this can be VERY inefficient.  It will recompute the model for EVERY data point.
*/
class model_image_polynomial_variable : public model_image
{
 private:
  //! Note that this is not defined because there is no uniform image size/resolution specification.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:

  //! Constructor.  Takes a model_image object, a vector of orders, and optional reference time.
  model_image_polynomial_variable(model_image& image, std::vector<int> order, double tref=0);

  //! Constructor.  Takes a model_image object, single order, and optional reference time.
  model_image_polynomial_variable(model_image& image, int order, double tref=0);

  //! Constructor.  Takes a pair of model_image objects.
  //model_image_polynomial_variable(model_image& image1, model_image& image2);
  virtual ~model_image_polynomial_variable() {};

  //! Access to set reference time after construction.
  void set_reference_time(double tref);

  //! Access to access reference time.
  double reference_time() const;
  
  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _new_size; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);


  //! A MANY-time generate function to construct the underlying model_image object for each datum.  Takes a vector of parameters.  Note that this will be called for EVERY data point, meaning that if the underlying model is not efficient (e.g., analytical) this will be VERY slow.
  void generate_image_model(double t);

  
  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
  virtual std::complex<double> visibility(datum_visibility& d, double acc);
  
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
  model_image& _image;
  size_t _new_size; //!< Number of parameters total
  std::vector<int> _orders;
  double _tref;

  std::vector<double> _image_parameters;
  
};

};
#endif

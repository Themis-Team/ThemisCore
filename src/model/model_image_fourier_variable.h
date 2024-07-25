/*!
  \file model_image_fourier_variable.h
  \author Avery Broderick et al.
  \date  March, 2021
  \brief Header file for the model_image_fourier_variable class, which generates a variable model by permitting model parameters to vary with time.  All parameters are constrained to vary as fourier series of time, though the order of each may differ.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_FOURIER_VARIABLE_H_
#define Themis_MODEL_IMAGE_FOURIER_VARIABLE_H_

#include "model_image.h"
#include "prior.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a fourier variable image model given an input image model.

  \details Each parameter of the input model_image is permitted to vary as a fourier with time.  Different parameters may have different order fouriers.  Constant parameters should have order 0.  A reference time may be specified.

  Parameter list:\n
  - parameters[0] ... Model 1 parameters[0] DC term (offset)\n
  - parameters[1] ... Model 1 parameters[0] cos coefficient of the half-wavelength term\n
  - parameters[2] ... Model 1 parameters[0] sin coefficient of the half-wavelength term\n
  - parameters[1] ... Model 1 parameters[0] cos coefficient of the full-wavelength term\n
  ...\n
  - parameters[n1+1] ... Model 1 parameters[0] DC term (offset)\n
  ...\n
  - parameters[n1+n2] ... Model 1 parameters[0] DC term (offset)\n
  ...\n

  Note that for models that require substantial recomputation of the underlying image, this can be VERY inefficient.  It will recompute the model for EVERY data point.
*/
class model_image_fourier_variable : public model_image
{
 private:
  //! Note that this is not defined because there is no uniform image size/resolution specification.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:

  //! Constructor.  Takes a model_image object, a vector of orders, and optional reference time.
  model_image_fourier_variable(model_image& image, std::vector<int> order, double tstart=0, double tend=86400);

  //! Constructor.  Takes a model_image object, single order, and optional reference time.
  model_image_fourier_variable(model_image& image, int order, double tstart=0, double tend=86400);

  //! Constructor.  Takes a pair of model_image objects.
  //model_image_fourier_variable(model_image& image1, model_image& image2);
  virtual ~model_image_fourier_variable() {};

  //! Access to set start/end time after construction.
  void set_start_time(double tstart);
  void set_end_time(double tend);

  //! Access to access start/end time.
  double start_time() const;
  double end_time() const;
  
  //! Set the underlying model image parameter priors that must be obeyed at all times.
  void set_priors(std::vector<prior_base*> priors);

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
  double _tstart, _tend;

  std::vector<double> _image_parameters;

  std::vector<prior_base*> _prior_ptrs;
  
};

};
#endif

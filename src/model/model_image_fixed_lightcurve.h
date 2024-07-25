/*!
  \file model_image_fixed_lightcurve.h
  \author Avery Broderick
  \date  September, 2020
  \brief Header file for the model_image_fixed_lightcurve class, which renormalizes the visibilities by a fixed time-variable normalization.  The provides a way to incorporate flux variability that does not impact image structure.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_FIXED_LIGHTCURVE_H_
#define Themis_MODEL_IMAGE_FIXED_LIGHTCURVE_H_

#include "model_image.h"
#include "interpolator1D.h"
#include <vector>
#include <string>

namespace Themis {

/*!
  \brief Defines a fixed light curve image model.

  \details Given a model_image, produces a model that is renormalized by a fixed light curve.  The parameter list and size is the same as input image.

  Parameter list:\n
  - parameters[0] ... Model parameters[0].\n
  ...\n
  - parameters[N-1] ... Model last parameter.\n

  Light curves may be provided either by a set of vectors or via a file name with time and dates as specified in the data files with associated fluxes:

  # Source Year   Day   Hour   Flux
  #
    M87    2017   101   14.3   1.35
  # M87    2017   101   14.5   1.40  # << This entry is commented
    M87    2017   101   14.7   1.55
  ...

  Only relative flux values matter if the given model_image has a flux normalization.
*/
class model_image_fixed_lightcurve : public model_image
{
 private:
  //! Note that this returns the image at the average light curve flux
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:

  //! Constructor.  Takes a model image and vector of times and fluxes.
  model_image_fixed_lightcurve(model_image& image, std::vector<double> t, std::vector<double> F);

  //! Constructor.  Takes a model image and vector of times and fluxes.
  model_image_fixed_lightcurve(model_image& image, std::string lightcurve_filename, std::string time_type="HH");

  //! Destructor
  virtual ~model_image_fixed_lightcurve() {};

  //! Set light curve
  void set_light_curve(std::vector<double> t, std::vector<double> F);

  //! Set light curve
  void set_light_curve(std::string light_curve_name, std::string time_type="HH");
  
  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _image.size(); };

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);

  
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


  //!\brief Sets the MPI communicator if the underlying images are parallelized
  virtual void set_mpi_communicator(MPI_Comm comm);
    
  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
  virtual std::string model_tag() const { return _image.model_tag(); };

  
 private:
  model_image& _image;
  Interpolator1D _light_curve_table;
  double _average_flux;
};

};
#endif

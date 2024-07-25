/*!
  \file model_symmetric_gaussian.h
  \author Avery Broderick
  \date  June, 2017
  \brief Header file for symmetric Gaussian image class.
  \details To be added
*/

#ifndef Themis_MODEL_SYMMETRIC_GAUSSIAN_H_
#define Themis_MODEL_SYMMETRIC_GAUSSIAN_H_

#include "model_visibility.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include <vector>
#include <complex>

namespace Themis {

/*!
  \brief Defines a symmetric Gaussian model based on the
  model_visibility_amplitude and model_closure_phase classes.

  \details The Gaussian image is defined by an axis size and
  overall flux normalization.  Note that this is not derived from 
  model_image because there is no meaning for the position angle.

  Parameter list:\n
  - parameters[0] ... Total, integrated flux in Jy.
  - parameters[1] ... Standard deviation in radians.

  \warning 
*/
  class model_symmetric_gaussian : public model_visibility, public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
{
 public:
  model_symmetric_gaussian();
  virtual ~model_symmetric_gaussian() {};

  //! A user-supplied function that returns the number of the parameters
  //! the model expects
  virtual inline size_t size() const { return 2; };

  //! A one-time generate function that permits model construction prior 
  //! to calling the visibility_amplitude, closure_phase, etc. for each 
  //! datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);
  
  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

  //! Returns visibility ampitudes in Jy computed from the image given a 
  //! datum_visibility_amplitude object, containing all of the accoutrements. 
  //! While this provides access to the actual data value, the two could 
  //! be separated if necessary.  Also takes an accuracy parameter with 
  //! the same units as the data, indicating the accuracy with which the 
  //! model must generate a comparison value.  Note that this is redefined 
  //! to accomodate the possibility of using the analytical computation.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

  /*! Returns closure phase in degrees computed from the image given a 
  datum_closure_phase object, containing all of the accoutrements. While 
  this provides access to the actual data value, the two could be separated 
  if necessary.  Also takes an accuracy parameter with the same units as 
  the data, indicating the accuracy with which the model must generate a 
  comparison value. Note that this is redefined since the closure phase of 
  Gaussian images is identically zero. */
  virtual double closure_phase(datum_closure_phase& d, double acc);
    
  /*! Returns closure amplitude computed from the image given a 
  datum_closure_phase object, containing all of the accoutrements.  While 
  this provides access to the actual data value, the two could be separated 
  if necessary.  Also takes an accuracy parameter with the same units as 
  the data, indicating the accuracy with which the model must generate a 
  comparison value.  Note that this can be redefined in child classes. */
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

 private:
  double _Itotal; //!< Internal total intensity.
  double _sigma;  //!< Std. dev. 
};

};
#endif

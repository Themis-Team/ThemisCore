/*!
  \file model_image_symmetric_gaussian.h
  \author Avery Broderick
  \date  November, 2018
  \brief Header file for the symmetric Gaussian image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_SYMMETRIC_GAUSSIAN_H_
#define Themis_MODEL_IMAGE_SYMMETRIC_GAUSSIAN_H_

#include "model_image.h"
#include <vector>
#include <complex>

namespace Themis {
  
  /*!
    \brief Defines an interface for a symmetric Gaussian image
    model based on the model_image class.  This is an explicit example
    of a model_image object.

    \details The Gaussian image is defined by the parameters in Broderick et al. (2011):
    an overall flux normalization, a measure of the total standard deviation, a measure
    of the symmetry, and the position angle (removed from the end of the parameter list 
    by model_image.generate() prior to passing to model_image_symmetric_gaussian.generate_image().)

  Parameter list:\n
    - parameters[0] ... Total, integrated flux in Jy.
    - parameters[1] ... \f$ \sigma \f$ in radians.

    \warning
  */
  class model_image_symmetric_gaussian : public model_image
  {
    private:
    //! Generates and returns rectalinear grid of intensities associated with the Gaussian image model in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
    virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
    
    public:
    model_image_symmetric_gaussian();
    virtual ~model_image_symmetric_gaussian() {};

    //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
    void use_numerical_visibilities();
    //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
    void use_analytical_visibilities();

    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return 2; };

    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
    virtual std::string model_tag() const
    {
      return "model_image_symmetric_gaussian";
    };

    //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);

    
  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
    virtual double closure_phase(datum_closure_phase& d, double acc);
      
    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_amplitude(datum_closure_amplitude& d, double acc);


    private:
    double _Itotal; //!< Internal total intensity.
    double _sigma; //!< Std. dev. in fiducial horizontal direction.

    bool _use_analytical_visibilities; //!< If true uses analytical visibility computation, if false use numerical visibilities.
    
  };
  
};
#endif

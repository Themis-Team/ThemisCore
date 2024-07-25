/*!
  \file model_image_themaussian.h
  \author Boris Georgiev
  \date  February, 2022
  \brief Header file for the Themaussian model image class. A Themaussian is a sum of flux-ordered asymmetric gaussians.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_THEMAUSSIAN_H_
#define Themis_MODEL_IMAGE_THEMAUSSIAN_H_

#include "model_image.h"
#include <vector>
#include <complex>

namespace Themis {
  
  /*!
    \brief Defines an interface for a Themaussian (a sum of flux-ordered asymmetric gaussians)
    model based on the model_image class.  This is an explicit example
    of a model_image object.

    \details The Gaussian image is defined by the parameters in Broderick et al. (2011):
    an overall flux normalization, a measure of the total standard deviation, a measure
    of the asymmetry, and the position angle (removed from the end of the parameter list 
    by model_image.generate() prior to passing to model_image_asymmetric_gaussian.generate_image().)
    For a Themaussian, the first Gaussian's flux is a parameter, and every subsequent Gaussian
    has a parameter of a fraction (0-1) of the previous Gaussian's flux.

    The equation for a gaussian is 
    \f$ B \exp [-a (x-x0)^2-2 b (x-x0) (y-y0)-c(y-y0)^2] \f$
    where
    \f$ a=\frac{\cos ^2\theta}{2 sm^2}+\frac{\sin ^2\theta}{2 sM} \f$
    \f$ b=\frac{\sin (2 \theta)}{4 sM^2}-\frac{\sin (2 \theta)}{4 sm^2} \f$
    \f$ c=\frac{\sin ^2\theta}{2 sm^2}+\frac{\cos ^2\theta}{2 sM^2} \f$
    with sM being the major axis and sm being minor. There is a transform of sm and sM into s and A:
    \f$ s = \sqrt{2} sm sM / \sqrt{sM^2 + sm^2} \f$
    \f$ A = \frac{sM^2-sm^2}{sM^2+sm^2} \f$ 
    with the reverse transform
    \f$ sm=s/\sqrt{1+A} \f$
    \f$ sM=s/\sqrt{1-A} \f$
    The gaussian has total flux F=2 pi sm sM B


  Parameter list:\n
    - parameters[0,6,12...] ... For 0, the integrated flux of the first gaussian in Jy. Each subsequent parameter is a ratio of the previous.\n
    - parameters[1,7,13...] ... \f$ \sigma = \sqrt{2} \sigma_m \sigma_M / \left( \sigma_M^2 + \sigma_m^2 \right)^{1/2} \f$
    - parameters[2,8,14...] ... \f$ A = (\sigma_M^2-\sigma_m^2)/(\sigma_M^2+\sigma_m^2) \f$ 
    - parameters[3,9,15...] ... Position angle (in model_image) in radians\n
    - parameters[4,10,16...] ... For 0, x location of first gaussian in radians. Each subsequent is the difference (x_i-x_0).\n
    - parameters[5,11,17...] ... For 0, y location of first gaussian in radians. Each subsequent is the difference (y_i-y_0).\n

    \warning Note that the two size parameters are not the semimajor and 
    semiminor axes, leading to a natural degeneracy between swapping the 
    axes sizes and rotating the image by pi/2 radians.
  */
  class model_image_themaussian : public model_image
  {
    private:
    //! Generates and returns rectalinear grid of intensities associated with the Gaussian image model in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
    virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
    
    public:
    model_image_themaussian(size_t Num_gauss);
    virtual ~model_image_themaussian() {};

    //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
    void use_numerical_visibilities();
    //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
    void use_analytical_visibilities();

      //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);

    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return 6*_Num_gauss; };

      //! \brief Calculates the complex visibility amplitude
      std::complex<double> complex_visibility(double u, double v);
 
    
  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
    virtual double closure_phase(datum_closure_phase& d, double acc);
      
    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_amplitude(datum_closure_amplitude& d, double acc);

    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
    virtual std::string model_tag() const
    {
      return "model_image_themaussian";
    };

    private:
    //double _Itotal; //!< Internal total intensity.
    //double _sigma_alpha; //!< Std. dev. in fiducial horizontal direction.
    //double _sigma_beta; //!< Std. dev. in fiducial vertical direction.
    
    size_t _Num_gauss;
    std::vector<double> _Itotal; 
    std::vector<double> _sigma_alpha; 
    std::vector<double> _sigma_beta; 
    std::vector<double> _PA; 
    std::vector<double> _xpos; 
    std::vector<double> _ypos; 
    
    bool _use_analytical_visibilities; //!< If true uses analytical visibility computation, if false use numerical visibilities.
    
  };
  
};
#endif

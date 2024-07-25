/*!
	\file model_image_xsringauss.h
	\author Jorge Alejandro Preciado-Lopez
	\date  November 2018
	\brief Header file for the geometric nine-parameter eccentric slashed ring (xsringauss) model image class.
	\details
*/

#ifndef THEMIS_MODEL_IMAGE_XSRINGAUSS_H_
#define THEMIS_MODEL_IMAGE_XSRINGAUSS_H_

#include "model_image.h"
#include <vector>

namespace Themis {

  /*!
    \brief Defines an interface for a Geometric Slashed Ring Model based on
    the model_image class. This is a simple example of a model_image object.
    
    \details The Geometric Slashed Rint Model is created by substracting out a
    disc from the inside of a larger disc, both of constant intensity.
    
    This model is defined by four parameters:
    - parameters[0]: The total integrated flux in Jy.
    - parameters[1]: \f$R \f$ = The overall size.
    - parameters[2]: \f$ \psi \f$ = The relative thickness.
    - parameters[3]: \f$ \epsilon \f$ = The eccentricity.
    - parameters[4]: \f$ f \f$ = The fading parameter.
    - parameters[5]: \f$ g_{ax} \f$ = The main axis of the FWHM ellipse in units of \f$R \f$
    - parameters[6]: \f$ a_{q} \f$ = The ellipse axial ratio
    - parameters[7]: \f$ g_{q} \f$ = The ratio of the Gaussian flux in the total flux
    - parameters[8]: \f$ \phi \f$ = The orientation (in model_image) in radians.
  */
  class model_image_xsringauss : public model_image
  {
   private:
    
    /*! \brief Generates and returns rectalinear grid of intensities 
      associated with the Crescent Image model in Jy/pixel located 
      at pixels centered on angular positions alpha and beta, both 
      specified in radians and aligned with a fiducial direction. 
      Note that the parameter vector has had the position removed.
    */
    virtual void generate_image(std::vector<double> parameters, 
				std::vector<std::vector<double> >& I, 
				std::vector<std::vector<double> >& alpha, 
				std::vector<std::vector<double> >& beta);
    
    
    //! \brief Calculates the Bessel Function J1(x)
    double BesselJ0(double x);
    
    //! \brief Calculates the Bessel Function J1(x)
    double BesselJ1(double x);
    
    //! \brief Calculates the Bessel Function J2(x)
    double BesselJ2(double x);
    
    //! \brief Calculates the complex visibility amplitude
    std::complex<double> complex_visibility(double u, double v);
    
    
  public:
    
    // Class constructor and destructor
    model_image_xsringauss();
    virtual ~model_image_xsringauss() {};
    
    
    //! \brief State switch to select numerically computed visibilities using the
    //! machinery in model_image. Once called, all future visibilities will
    //! be computed numerically until use_analytical_visibilities() is called.
    void use_numerical_visibilities();
    
    
    //! \brief Sets model_image_crescent to generate production images with
    //! resolution Nray x Nray. The default is 128x128, which is probably
    //! larger than required in practice.      
    void set_image_resolution(int Nray);
    
    
    //! \brief State switch to select analytically computed visibilities using the
    //! analytical Fourier transform of the Gaussian. Once called, 
    //! visibilities will be computed analytically until 
    //! use_numerical_visibilities() is called.
    void use_analytical_visibilities();
    
    
    //! \brief A user-supplied function that returns the number of the parameters
    //! the model expects
    virtual inline size_t size() const { return 9; };
      
    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
    virtual std::string model_tag() const
    {
      return "model_image_xsringauss";
    };
    
			
    //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
    
    
    //! \brief Returns visibility ampitudes in Jy computed from the image given a
    //! datum_visibility_amplitude object, containing all of the accoutrements.
    //! While this provides access to the actual data value, the two could
    //! be separated if necessary.  Also takes an accuracy parameter with the
    //! same units as the data, indicating the accuracy with which the model
    //! must generate a comparison value.  Note that this is redefined to 
    //! accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);
    
    
    //! \brief Returns closure phase in degrees computed from the image given a
    //! datum_closure_phase object, containing all of the accoutrements. 
    //! While this provides access to the actual data value, the two could
    //! be separated if necessary. Also takes an accuracy parameter with 
    //! the same units as the data, indicating the accuracy with which the
    //!  model must generate a comparison value. Note that this is redefined
    //! since the closure phase of Gaussian images is identically zero.
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
    double _V0;		//!< Internal total intensity.
    double _Rext;	//!< Radius of the larger disc.
    double _Rint;	//!< Radius of the smaller disc.
    double _d;    //!< Distance between centers of the two disks
    double _f;    //!< The fading parameter.
    double _sigma_alpha;		//!< Gaussian axis
    double _sigma_beta;		//!< Gaussian axis
    double _gq;		//!< Fraction of the Gaussian flux
    int _Nray;		//!< Image resolution
    bool _use_analytical_visibilities; //!< If true uses analytical visibility computation; if false uses numerical visibilities.
    
  };
};

#endif

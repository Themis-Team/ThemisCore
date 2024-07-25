/*!
  \file model_image_mring_floor.h
  \author Paul Tiede
  \date  March 2021
  \brief Header file for the geometric mring model.
  \details To be added.
*/

#ifndef THEMIS_MODEL_IMAGE_MRING_FLOOR_H_
#define THEMIS_MODEL_IMAGE_MRING_FLOOR_H_

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
      - parameters[2]: The fraction of flux in the disk floor.
      - parameters[3]: <em>R</em> = ring radius.
      - parameters[4,5]: coefficients for the first mode of the ring
      - parameters[6,7]: coefficients for the second mode of the ring
      - ....
      - parameters[2+2N-2,2+2N-1]: coefficients for the last N mode of the ring
  */
  class model_image_mring_floor : public model_image
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

      
      
      
    public:
      //! \brief Calculates the complex visibility amplitude
      std::complex<double> complex_visibility(double u, double v);
      //! \brief Calculates the integer order Bessel Function Jn(x)
      double BesselJ(int n, double x);
      
      // Class constructor and destructor
      //! order is the order of fourier mode expansion.
      model_image_mring_floor(size_t order);
      virtual ~model_image_mring_floor() {};
      
      
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

      //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
      virtual void generate_model(std::vector<double> parameters);
      
      
      
      //! \brief A user-supplied function that returns the number of the parameters
      //! the model expects
      virtual inline size_t size() const { return 3+_order*2; };
      
      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const
      {
        std::stringstream tag;
        tag << "model_image_mring_floor " << _order;
        return tag.str();
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
      size_t _order;
      double _V0;		//!< Internal total intensity.
      double _R;	//!< Radius of the larger disc.
      double _floor; //!< Disk percentage of flux
      std::vector<double> _an;	//!< Real components of the mring modes
      std::vector<double> _bn;    //!< Imaginaryc componenets of the mring modes
      int _Nray;		//!< Image resolution
      bool _use_analytical_visibilities; //!< If true uses analytical visibility computation; if false uses numerical visibilities.
    
  };
  
};

#endif

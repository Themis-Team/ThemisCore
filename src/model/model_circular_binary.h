/*!
  \file model_circular_binary.h
  \author Roman Gold, Avery Broderick
  \date  July, 2017
  \brief Header file for binary model class.
  \details This model is explicitly time dependent.
*/

#ifndef Themis_MODEL_CIRCULAR_BINARY_H_
#define Themis_MODEL_CIRCULAR_BINARY_H_

// #include "model_image.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include <vector>
#include <complex>

namespace Themis {

/*!  \brief Defines a binary model based of two Gaussians. This is an
  explicit model based on the model_visibility_amplitude,
  model_closure_phase, and model_closure_amplitude classes.

  \details The Gaussian blobs by their axis sizes and
  overall flux normalizations.  Note that this is not derived from 
  model_image because there is no meaning for the position angle.

  Parameter list:\n
  - parameters[0] ... Total, integrated flux in Jy for primary.
  - parameters[1] ... Standard deviation of the Gaussian in radians of primary.
  - parameters[2] ... Spectral index (\f$I_\nu \propto \nu^{-\alpha}\f$) of primary.
  - parameters[3] ... Total, integrated flux in Jy for secondary.
  - parameters[4] ... Standard deviation of the Gaussian in radians of secondary.
  - parameters[5] ... Spectral index (\f$I_\nu \propto \nu^{-\alpha}\f$) of secondary.
  - parameters[6] ... Total mass of binary in solar masses.
  - parameters[7] ... Binary mass ratio (secondary/primary).
  - parameters[8] ... Orbital separation in pc. 
  - parameters[9] ... Binary distance in Mpc.
  - parameters[10] .. Phase offset of binary in radians.
  - parameters[11] .. Cosine of the orbital angular momentum vector.
  - parameters[12] .. Position angle of the orbital angular momentum vector in degrees.

  \warning 
*/
  class model_circular_binary : public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
  {
  public:
    model_circular_binary();
    virtual ~model_circular_binary() {};
    
    //! A user-supplied function that returns the number of the parameters
    //! the model expects
    virtual inline size_t size() const { return 13; };
    
    //! A one-time generate function that permits model construction prior 
    //! to calling the visibility_amplitude, closure_phase, etc. for each 
    //! datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);
    
    //! Returns visibility amplitudes in Jy computed from the image given a 
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
    
    double _I1; //!< Primary total intensity
    double _sigma1; //!< Primary std dev
    double _alpha1; //!< Primary spectral index
    double _I2; //!< Secondary total intensity
    double _sigma2; //!< Secondary std dev
    double _alpha2; //!< Secondary spectral index
    double _M; //!< Total mass in Msun
    double _q; //!< Mass ratio
    double _R; //!< Orbital separation in cm
    double _D; //!< Distance to source in cm
    double _Phi0; //!< Orbital phase offset
    double _mu; //!< Cosine of inclination
    double _position_angle; //!< Position angle
    
    double _Omega; //!< Orbital angular velocity
    double _cosi, _sini;
    
    //! Computes the complex visibility for the model
    std::complex<double> complex_visibility(double u, double v, double t);
    
    //! Converts from lab time to the observer time, including the time delays
    double get_obs_time(double t_lab);
    //! Converts from observer time to the lab time, including the time delays
    double get_lab_frame_time(double t_obs);
    //! Computes and returns the locations, velocities, and line-of-sight velocities of the two binary components at a given lab time
    void get_orbital_solution(double t_lab, double x1[2], double& v1, double& v1r, double x2[2], double& v2, double& v2r);
    
    //! Doppler factor g = gamma*(1-beta.n)
    double doppler_factor(double v, double vr) const;

    //! Computes the visibility for a gaussian at (u,v) with size sigma and total intensity 1 Jy
    double gaussian_visibility(double u, double v, double sigma) const;
  };
};
#endif

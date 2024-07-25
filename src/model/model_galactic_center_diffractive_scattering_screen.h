/*!
  \file model_galactic_center_diffractive_scattering_screen.h   
  \author Paul Tiede 
  \date August 2019
  \brief Header file for diffractive scattering interface.
  \details Implements the diffractive scattering model for Sgr A*, following Psaltis 2018 and ehtim.
  The values for the scattering screen are taken from Johnson et al. 2018.
  For the scattering itself, there are three models for the wandering magnetic field.
  to the parameter list since we will fit out the explicit realization of the scattering screen./
*/

#ifndef Themis_MODEL_GALACTIC_CENTER_DIFFRACTIVE_SCATTERING_SCREEN_H_
#define Themis_MODEL_GALACTIC_CENTER_DIFFRACTIVE_SCATTERING_SCREEN_H_

#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "quadrature.h"
#include "interpolator1D.h"
#include "interpolator2D.h"
#include "utils.h"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <valarray>


namespace Themis {
  /*!
    \class model_galactic_center_diffractive_scattering_screen
    \author Paul Tiede
    \date Aug. 2019
    \brief Blurs the visibilities of the input model according to the diffractive scattering kernel with default values from Johnson 2018.
    \details Unlike the refractive scattering screen this does not increase the number of parameters since this is considered a deterministic process.
  */
  class model_galactic_center_diffractive_scattering_screen : public model_visibility_amplitude
  {

    public:
      model_galactic_center_diffractive_scattering_screen(
                                  model_visibility_amplitude& model, 
                                  double frequency=230e9,
                                  std::string scattering_model="dipole",
                                  double observer_screen_distance=2.82*3.086e21, double source_screen_distance=5.53*3.086e21,
                                  double theta_maj_mas_cm=1.38, double theta_min_ma_cm=0.703, double POS_ANG=81.9, 
                                  double scatt_alpha=1.38, double r_in=800e5, double r_out=1e20);

      virtual ~model_galactic_center_diffractive_scattering_screen() {};

      //! Size of the supplied model image, i.e., number of parameters expected.
      virtual inline size_t size() const { return _model.size(); };

      //! Takes model parameters and generates the model with the diffractive scattering.
      virtual void generate_model(std::vector<double> parameters);

      //! Redefines the visibility amplitude function since we need to multiply by the scattering kernel
      virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);
      //! Sets the mpi communicator for generating images
      virtual void set_mpi_communicator(MPI_Comm comm);
    
    
    private:

      //!< Model visibility amplitudes
      model_visibility_amplitude& _model;

      //const double _frequency; //!< frequency of observation in Hz
      const double _wavelength; //<! wavelength of observatoin in cm

      const std::string _scattering_model; //!< Which scattering model to use. Have three options, dipole, boxcar, vonMises
      //const double _observer_screen_distance; //<! Distance of observer to screen in cm.
      //const double _source_screen_distance; //<! Distance of source to screen in cm.
      
      const double _theta_maj_mas_cm, _theta_min_mas_cm; //!< Semi-major and minor axis length of Gaussian component of scattering screen in mas at refernce wavelength of 1 cm.
      //const double _POS_ANG; //!< Position angle of scattering Gaussian in degrees. Note we use the opposite convention of ehtim.
      const double _phi0; //!< (90-_POS_ANG)
      const double _M; //!< Magnification factor
      //const double _rF; //!< Fresnel scale

      const double _scatt_alpha; //!< Scattering turulence and dissipation scale power law parameter. Kolmogorov scattering has alpha=5/3
      const double _rin; //!< Inner scale for the turbulence in the scattering screen.
      //const double _rout; //!< Outer scale for the turbulence in the scattering screen.


      //Scattering special parameters
      //
      double _zeta0;
      double _Qbar;//!< Power spectrum normalization
      double _C_scatt_0;//!< Scattering coefficent for phase structure function
      double _kZeta; //!< Scattering kernel phase structure coefficient for different wandering magnetic field models

      double _Bmaj,_Bmin; //Bmaj,Bmin coefficients for scattering kernel in Psaltis et. al. 2018
      
      
      //!< Dphi function for the ensemble averaged image.
      double Dphi(double r, double phi) const;

      //!< Power spectrum of phase screen.
      double Q(double qx, double qy) const;

      //Class that defines the integrand for Bmaj and Bmin and P_phi
      class _P_phi : public Integrand
      {
       public:
         _P_phi(std::string axis,std::string model, double alpha, double phi0);
         inline void set_kZeta(double kZeta){_kZeta = kZeta; };
         double operator()(double x) const;
       private:
         const std::string _axis; //Whether using major or minor axis
         const std::string _model; //scattering model
         const double _alpha, _phi0;
         double _kZeta;
         double _angPow;
         double _phaseshift;
      };

      //!< Wandering magnetic field angular model.
      _P_phi _P_phi_func;
      //!< Normalization for wandering magnetic field model
      double _P_phi_norm;
      


  };

};

#endif


/*!
  \file model_riaf_wrapper.h
  \author Mansour Karami
  \date  February, 2018
  \brief Header file for extended RIAF image+flux model wrapper class.
  \details To be added
*/

#ifndef Themis_MODEL_RIAF_WRAPPER_H_
#define Themis_MODEL_RIAF_WRAPPER_H_

#include "model_image_riaf.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "model_flux.h"
#include "model_riaf.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

/*!
  \brief Defines an image and flux model based on the RIAF models employed in Broderick, Fish, Doeleman and Loeb (2011) and Broderick et al. (2016), permitting greater freedom to modify the underlying semi-analytic accretion flow structure. 

  \details Provides explicit implementation of the model_image and model_flux objects for the semi-analytic RIAF models.  There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_image_riaf.cpp for more.

Parameter list:
  - parameters[0] ... Black hole spin parameter (-1 to 1).
  - parameters[1] ... Cosine of the spin inclination relative to line of sight.
  - parameters[2] ... "Thermal" electron population density normalization in \f${\rm cm}^{-3}\f$.   
  - parameters[3] ... "Thermal" electron population density radial power law. // Fixed
  - parameters[4] ... "Thermal" electron population density \f$h/r\f$. 
  - parameters[5] ... "Thermal" electron population temperature normalization in K.  
  - parameters[6] ... "Thermal" electron population temperature radial power law. //Fixed 
  - parameters[7] ... Power-law electron population density normalization in \f${\rm cm}^{-3}\f$. 
  - parameters[8] ... Power-law electron population density radial power law. // Fixed 
  - parameters[9] ... Power-law electron population density \f$h/r\f$. // Same ah Thermal height 
  - parameters[10] .. Power-law electron population density spectral index. // Fixed 
  - parameters[11] .. Power-law electron population density minimum Lorentz factor. //Fixed 
  - parameters[12] .. Plasma beta. //Fixed 
  - parameters[13] .. Sub-keplerian parameter, i.e., sets \f$u_\phi/u_t =\f$ parameters[13]\f$\times\f$\<keplerian value\>.
  - parameters[14] .. Position angle (in model_image) in radians.

  \warning Requires the VRT2 library to be installed.
*/  
  class model_riaf_wrapper : public model_riaf
{
 public:
  //! Constructor to make a RIAF model. Takes a vector of frequencies (Hz), mass (cm), distance (cm).
  model_riaf_wrapper(std::vector<double> frequencies, double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  ~model_riaf_wrapper();

  //! Returns the number of the parameters the model expects
  inline size_t size() const { return 6; };
 
  
  //! Currently simply saves model parameters. Repeat model production is prevented within the model_image_riaf objects.
  void generate_model(std::vector<double> parameters);
 

  //void set_image_resolution(int Nray);

  //! Returns visibility ampitudes in Jy, computed numerically. The accuracy parameter is not used at present.
  //virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees. The accuracy parameter is not used at present.
  //virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure ampitudes, computed numerically. The accuracy parameter is not used at present.
  //virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  //! Returns flux in Jy computed from the model_image_riaf with a frequency nearest to that requested. The accuracy parameter, in Jy, is used to determine the resolution of the image from which to obtain the flux estimate.
  //virtual double flux(datum_flux& d, double accuracy);

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  //virtual void set_mpi_communicator(MPI_Comm comm);


  //void set_parameter_state(std::vector<size_t> fixed_parameters, std::vector<double> fixed_values);

 protected:

 private:
  //std::vector< model_image_riaf* > _riaf_images; //!< Vector of model_image_riaf objects constructed independently at each flux in the list passed to the constructor.

  //std::vector<double> _parameters; //!< Internal copy of parameters to pass to the various copies of model_image_riaf.

  //std::vector<size_t> _fixed_parameters, _changing_parameters;
  //std::vector<double> _fixed_values;
  //! Finds the nearest frequency in the list originally passed.
  //size_t find_frequency_index(double frequency) const;
  //size_t _size;

  //  bool _generated_model; //!< True if a model has been generated. Used to avoid repeated computation of the model at the same location.
};

};

#endif

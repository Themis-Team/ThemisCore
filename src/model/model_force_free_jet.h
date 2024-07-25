/*!
  \file model_force_free_jet.h
  \author Paul Tiede
  \date  Sept, 2018
  \brief Header file for extended force free image+flux model class.
  \details To be added
*/

#ifndef Themis_MODEL_RIAF_H_
#define Themis_MODEL_RIAF_H_

#include "model_image_force_free_jet.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "model_flux.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

/*!
  \brief Defines an image and flux model based on the force free jet models employed in Broderick and Loeb (2009).

  \details Provides explicit implementation of the model_image and model_flux objects for the force free jet models. There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_image_force_free_jet.cpp for more.

  Parameter list:\n
    - parameters[0]  ... Black hole mass in solar mass.
    - parameters[1]  ... Distance to black hole in kpc.
    - parameters[2]  ... Black hole spin parameter (-1 to 1).
    - parameters[3]  ... Cosine of the spin inclination relative to line of sight.
    - parameters[4]  ... jet radial power law index
    - parameters[5]  ... jet opening angle in degrees
    - parameters[6]  ... jet loading radius in M
    - parameters[7]  ... jet max Lorentz factor
    - parameters[8]  ... jet electron density normalization
    - parameters[9]  ... jet synchrotron spectral index
    - parameters[10]  ... jet synchrotron magnetic field normalization
    - parameters[11] ... synchrotron emitting electrons minimum lorentz factor
    - parameters[12] ... edge of disk and jet in terms of ISCO.
    - parameters[13] ... Position angle (in model_image) in radians.
 
  \warning Requires the VRT2 library to be installed.
*/  
class model_force_free_jet : public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude, public model_flux
{
 public:
  //! Constructor to make a RIAF model. Takes a vector of frequencies (Hz), mass (cm), distance (cm).
  model_force_free_jet(std::vector<double> frequencies);
  ~model_force_free_jet();

  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return 14; };

  //! Sets model_image_sed_fitted_riaf to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int NxRay, int number_of_refines=0);

  //! Sets image dimensions
  void set_screen_size( double Rmax );
 
  
  //! Currently simply saves model parameters. Repeat model production is prevented within the model_image_riaf objects.
  virtual void generate_model(std::vector<double> parameters);

  //! Returns visibility ampitudes in Jy, computed numerically. The accuracy parameter is not used at present.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees. The accuracy parameter is not used at present.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure ampitudes, computed numerically. The accuracy parameter is not used at present.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  //! Returns flux in Jy computed from the model_image_riaf with a frequency nearest to that requested. The accuracy parameter, in Jy, is used to determine the resolution of the image from which to obtain the flux estimate.
  virtual double flux(datum_flux& d, double accuracy);

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm);

 protected:

  // private:
  std::vector< model_image_force_free_jet* > _ffj_images; //!< Vector of model_image_riaf objects constructed independently at each flux in the list passed to the constructor.

  std::vector<double> _parameters; //!< Internal copy of parameters to pass to the various copies of model_image_riaf.

  //! Finds the nearest frequency in the list originally passed.
  size_t find_frequency_index(double frequency) const;

  bool _generated_model; //!< True if a model has been generated. Used to avoid repeated computation of the model at the same location.
};

};

#endif



/*!
  \file model_flux.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for flux model class.
  \details To be added
*/

#ifndef Themis_MODEL_FLUX_H_
#define Themis_MODEL_FLUX_H_

#include <vector>
#include "data_flux.h"
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for models that generate flux data appropriate for comparison with datum_flux.h.

  \details To facilitate model-data comparisons, here a fixed interface that may be assumed by likelihoods, etc., for generating model flux data is defined.  All models that produce flux data should be child classes of model_flux, thereby inheriting this interface.  The key function is flux which takes a datum_flux, providing access to all of the relevant accoutrements of a data point, and an accuracy parameter, and returns the model flux.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
*/
class model_flux
{
 public:
  model_flux();
  virtual ~model_flux();

  //! A user-supplied one-time generate function that permits model construction prior to calling the flux for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters) = 0;

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const = 0;

  //! A user-supplied function that returns the spectral flux density in Jy.  Takes a datum_flux to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
  virtual double flux(datum_flux& d, double accuracy) = 0;

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) { _comm=comm; };

 protected:
  MPI_Comm _comm;
};

};
#endif

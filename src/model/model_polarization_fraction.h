/*!
  \file model_fractional_polarization.h
  \author Roman Gold
  \date  June, 2018
  \brief Header file for polarization fraction model class.
  \details To be added
*/

#ifndef Themis_MODEL_POLARIZATION_FRACTION_H_
#define Themis_MODEL_POLARIZATION_FRACTION_H_

#include <vector>
#include "data_polarization_fraction.h"
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for models that generate visibility amplitude data appropriate for comparison with datum_polarization_fraction.h.

  \details To facilitate model-data comparisons, here a fixed interface that may be assumed by likelihoods, etc., for generating model visibility amplitude data is defined.  All models that produce visibility amplitude data should be child classes of model_polarization_fraction, thereby inheriting this interface.  The key function is polarization_fraction which takes a datum_polarization_fraction, providing access to all of the relevant accoutrements of a data point, and an accuracy parameter, and returns the model visibility amplitude.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
*/
class model_polarization_fraction 
{
 public:
  model_polarization_fraction();
  virtual ~model_polarization_fraction();

  //! A user-supplied one-time generate function that permits model construction prior to calling the polarization_fraction for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters) = 0;

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const = 0;

  //! A user-supplied function that returns the closure amplitudes.  Takes a datum_polarization_fraction to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
  virtual double polarization_fraction(datum_polarization_fraction& d, double accuracy) = 0;

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_polarization_fraction proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
    _comm=comm;
  };

 protected:
  MPI_Comm _comm;
};

};
#endif

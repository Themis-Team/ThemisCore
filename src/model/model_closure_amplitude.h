/*!
  \file model_closure_amplitude.h
  \author Avery E. Broderick
  \date  July, 2017
  \brief Header file for visibility ampitude model class.
  \details To be added
*/

#ifndef Themis_MODEL_CLOSURE_AMPLITUDE_H_
#define Themis_MODEL_CLOSURE_AMPLITUDE_H_

#include <vector>
#include "data_closure_amplitude.h"
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for models that generate visibility amplitude data appropriate for comparison with datum_closure_amplitude.h.

  \details To facilitate model-data comparisons, here a fixed interface that may be assumed by likelihoods, etc., for generating model visibility amplitude data is defined.  All models that produce visibility amplitude data should be child classes of model_closure_amplitude, thereby inheriting this interface.  The key function is closure_amplitude which takes a datum_closure_amplitude, providing access to all of the relevant accoutrements of a data point, and an accuracy parameter, and returns the model visibility amplitude.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
*/
class model_closure_amplitude 
{
 public:
  model_closure_amplitude();
  virtual ~model_closure_amplitude();

  //! A user-supplied one-time generate function that permits model construction prior to calling the closure_amplitude for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters) = 0;

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const = 0;

  //! A user-supplied function that returns the closure amplitudes.  Takes a datum_closure_amplitude to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy) = 0;

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_closure_amplitude proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
    _comm=comm;
  };

 protected:
  MPI_Comm _comm;
};

};
#endif

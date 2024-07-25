/*!
  \file model_visibility.h
  \author Avery E. Broderick
  \date  November, 2018
  \brief Header file for visibility model class.
  \details To be added
*/

#ifndef Themis_MODEL_VISIBILITY_H_
#define Themis_MODEL_VISIBILITY_H_

#include <vector>
#include <complex>
#include "data_visibility.h"
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for models that generate complex visibility data appropriate for comparison with datum_visibility.h.

  \details To facilitate model-data comparisons, here a fixed interface that may be assumed by likelihoods, etc., for generating model complex visibility data is defined.  All models that produce complex visibility data should be child classes of model_visibility, thereby inheriting this interface.  The key function is visibility which takes a datum_visibility, providing access to all of the relevant accoutrements of a data point, and an accuracy parameter, and returns the model complex visibility.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
*/
class model_visibility
{
 public:
  model_visibility();
  virtual ~model_visibility();

  //! A user-supplied one-time generate function that permits model construction prior to calling the visibility for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters) = 0;

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const = 0;


  //! Returns complex visibility in Jy computed from the image given a datum_visibility object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy) = 0;

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_visibility proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
    _comm=comm;
  };

 protected:
  MPI_Comm _comm;
};

};
#endif

/*!
  \file model_crosshand_visibilities.h
  \author Avery E. Broderick
  \date  March, 2020
  \brief Header file for crosshand visibilities model class.
  \details To be added
*/

#ifndef Themis_MODEL_CROSSHAND_VISIBILITIES_H_
#define Themis_MODEL_CROSSHAND_VISIBILITIES_H_

#include <vector>
#include <complex>
#include <iostream>
#include <cstdlib>
#include "data_crosshand_visibilities.h"
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for models that generate the full complex crosshand visibility data appropriate for comparison with datum_crosshand_visibilities_amplitude.h.

  \details To facilitate model-data comparisons, here a fixed interface that may be assumed by likelihoods, etc., for generating model complex crosshand visibilities data is defined.  All models that produce complex crosshand visibility data should be child classes of model_crosshand_visibilities, thereby inheriting this interface.  The key function is crosshand_visibilities which takes a datum_crosshand_visibilities_amplitude, providing access to all of the relevant accoutrements of a data point, and an accuracy parameter, and returns a vector of four complex numbers with the RR,LL,RL,LR complex visibilities in Jy.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
*/
class model_crosshand_visibilities
{
 public:
  model_crosshand_visibilities();
  virtual ~model_crosshand_visibilities();

  //! A user-supplied one-time generate function that permits model construction prior to calling the visibility_amplitude for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters) = 0;

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const = 0;



  virtual void set_data(const data_crosshand_visibilities& data) { (void)data; }

  virtual bool use_cached_exp() const { return false; }

  //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image
  //! given a datum_crosshand_visibilities object.  This is the original datum-only interface.
  virtual std::vector< std::complex<double> >
  crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy) = 0;

  //! Indexed helper. Default behavior is to fall back to the original datum-only interface.
  virtual std::vector<std::complex<double>>
  crosshand_visibilities(size_t d_idx,
                         datum_crosshand_visibilities& d,
                         double accuracy)
  {
    (void)d_idx;
    return crosshand_visibilities(d, accuracy);
  }

  virtual void fill_crosshand_visibilities(size_t d_idx,
                                           datum_crosshand_visibilities& d,
                                           double accuracy,
                                           std::complex<double>* out)
  {
    std::vector<std::complex<double>> tmp =
      crosshand_visibilities(d_idx, d, accuracy);
    out[0] = tmp[0];
    out[1] = tmp[1];
    out[2] = tmp[2];
    out[3] = tmp[3];
  }

  
  /*
  virtual void set_data(const data_crosshand_visibilities& data) { (void)data; }

  virtual bool use_cached_exp() const { return false; }

  virtual std::vector<std::complex<double>> crosshand_visibilities(size_t d_idx, datum_crosshand_visibilities& d, double accuracy)
  {
    (void)d_idx;
    return crosshand_visibilities(d, accuracy);
  }
  
  //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image given a datum_crosshand_visibilities_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::vector<std::complex<double>> crosshand_visibilities(size_t d_idx,
                                                                   datum_crosshand_visibilities& d,
                                                                   double accuracy)
  {
    (void)d_idx;
    (void)d;
    (void)accuracy;

    std::cerr
      << "ERROR: model_crosshand_visibilities::crosshand_visibilities(size_t, ...)\n"
      << "       base-class indexed fallback was called.\n"
      << "       A derived crosshand model is missing its indexed override.\n";
    std::exit(1);
  }

  virtual void fill_crosshand_visibilities(size_t d_idx,
                                           datum_crosshand_visibilities& d,
                                           double accuracy,
                                           std::complex<double>* out)
  {
    (void)d_idx;
    (void)d;
    (void)accuracy;
    (void)out;

    std::cerr
      << "ERROR: model_crosshand_visibilities::fill_crosshand_visibilities(...)\n"
      << "       base-class fill fallback was called.\n"
      << "       A derived crosshand model is missing its fill override.\n";
    std::exit(1);
  }

  virtual std::vector< std::complex<double> > crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy) = 0;

  virtual void fill_crosshand_visibilities(size_t d_idx,
                                         datum_crosshand_visibilities& d,
                                         double accuracy,
                                         std::complex<double>* out)
  {
    std::vector<std::complex<double>> tmp = crosshand_visibilities(d_idx, d, accuracy);
    out[0] = tmp[0];
    out[1] = tmp[1];
    out[2] = tmp[2];
    out[3] = tmp[3];
  }
    */


  
  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_crosshand_visibilities proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
    _comm=comm;
  };

 protected:
  MPI_Comm _comm;
};

};
#endif

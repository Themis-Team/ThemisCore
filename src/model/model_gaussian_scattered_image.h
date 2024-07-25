/*!
  \file model_gaussian_scattered_image.h   
  \author Roman Gold
  \date Apr 2017
  \brief Header file for ensemble averaged scattering interface.
  \details 
  \todo convert from image to visibility / more general model 
*/

#ifndef Themis_MODEL_ENSEMBLE_AVERAGE_SCATTERED_IMAGE_H_
#define Themis_MODEL_ENSEMBLE_AVERAGE_SCATTERED_IMAGE_H_

//#include "model_image.h"
#include "model_visibility_amplitude.h"
#include <vector>

namespace Themis {

  /*!
    \class model_gaussian_scattered_image
    \author Roman Gold
    \date Apr 2017
    \brief Defines the interface for models that generate scattered visibilities 
    \details Scattering implementation assumes we are in ensemble-averaged 
    (Gaussian blurring) regime
  */
  class model_gaussian_scattered_image : public model_visibility_amplitude
  {
  public:
    model_gaussian_scattered_image(model_visibility_amplitude& model, double pivot_frequency=230e9);
    virtual ~model_gaussian_scattered_image() {};
    
    //! Size of the supplied model image, i.e., number of parameters expected.
    virtual inline size_t size() const { return _model.size()+7; };
    
    //! Takes model parameters and generates 
    virtual void generate_model(std::vector<double> parameters);
    
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);
    
    //! Defines a set of processors provided to the model for parallel
    //! computation via an MPI communicator.  Only facilates code 
    //! parallelization if the model computation is parallelized via MPI.
    virtual void set_mpi_communicator(MPI_Comm comm);
    
  private:
    model_visibility_amplitude& _model;
    bool _generated_model;
    std::vector<double> _screen_params;
    double _pivot_frequency;
  };

};
#endif


/*!
  \file model_ensemble_averaged_parameterized_scattered_image.h   
  \author Avery Broderick & Hung-Yi Pu
  \date Jun 2018
  \brief Header file for ensemble averaged parameterized scattering interface.
  \details
  \todo convert from image to visibility / more general model 
*/

#ifndef Themis_MODEL_ENSEMBLE_AVERAGED_PARAMETERIZED_SCATTERED_IMAGE_H_
#define Themis_MODEL_ENSEMBLE_AVERAGED_PARAMETERIZED_SCATTERED_IMAGE_H_

//#include "model_image.h"
#include "model_visibility_amplitude.h"
#include <vector>

namespace Themis {

  /*!
    \class model_ensemble_averaged_parameterized_scattered_image
    \author Avery Broderick & Hung-Yi Pu
    \date Apr 2017
    \brief Defines a parameterized diffractive scattering model
    \details Scattering implementation assumes we are in ensemble-averaged 
    (Gaussian blurring) regime.  This appends a set of seven scattering model
    parameters, consisting of:\n
    - parameters[model.size+0] ... Major/minor axis at pivot frequency.
    - parameters[model.size+1] ... Frequency power law index of major/minor axis.
    - parameters[model.size+2] ... Minor/major axis at pivot frequency.
    - parameters[model.size+3] ... Frequency power law index of minor/major axis.
    - parameters[model.size+4] ... Position angle at pivot frequency.
    - parameters[model.size+5] ... Position angle frequency dependence normalization.
    - parameters[model.size+6] ... Position angle frequency power law index.
  */
  class model_ensemble_averaged_parameterized_scattered_image : public model_visibility_amplitude
  {
    
    private:
    
    public:
      model_ensemble_averaged_parameterized_scattered_image(model_visibility_amplitude& model, double pivot_frequency=230e9);
      virtual ~model_ensemble_averaged_parameterized_scattered_image() {};

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

      std::vector<double> _scattering_parameters;
      double _pivot_frequency;
  };

};
#endif


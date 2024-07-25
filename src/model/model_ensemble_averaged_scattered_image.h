/*!
  \file model_ensemble_averaged_scattered_image.h   
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
    \class model_ensemble_averaged_scattered_image
    \author Roman Gold
    \date Apr 2017
    \brief Defines the interface for models that generate scattered visibilities 
    \details Scattering implementation assumes we are in ensemble-averaged 
    (Gaussian blurring) regime
  */
  class model_ensemble_averaged_scattered_image : public model_visibility_amplitude
  {
    
    private:
    
      // virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

    public:
      model_ensemble_averaged_scattered_image(model_visibility_amplitude& model);
      virtual ~model_ensemble_averaged_scattered_image() {};

      //! Size of the supplied model image, i.e., number of parameters expected.
      virtual inline size_t size() const { return _model.size(); };

      //! Takes model parameters and generates 
      virtual void generate_model(std::vector<double> parameters);

      virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

      // virtual double closure_phase(datum_closure_phase& d, double accuracy);

      /*
        \brief Provide access to unscattered image.
      
        \details Provide access to *unscattered* image, note that the 
        scattered image is never defined, as it need not.
          
        \param alpha coordinate 1 in image plane
        \param beta coordinate 2 in image plane
        \param I Intensity in Jy
      */

      //void get_unscattered_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;

      //! Defines a set of processors provided to the model for parallel
      //! computation via an MPI communicator.  Only facilates code 
      //! parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
  
    private:
      model_visibility_amplitude& _model;
    
  };

};
#endif


/*!
  \file model_image_smooth.h   
  \author Avery Broderick & Paul Tiede
  \date Jun 2018
  \brief Header file for model_image_smooth.
  \details Smooths a model_image using a asymmetric Gaussian smoothing Kernel.
*/

#ifndef Themis_MODEL_IMAGE_SMOOTH_
#define Themis_MODEL_IMAGE_SMOOTH_

#include "model_image.h"
#include <vector>

namespace Themis {

  /*!
    \class model_image_smooth
    \author Avery Broderick & Paul Tiede
    \date Nov 2018
    \brief Defines a class that smooths a model image with some asymmetric smoothing Kernel
    \details This appends a set of smoothing parameters to the model_image
    parameters, consisting of:\n
    - parameters[model_size()-1+1] ... size of the gaussian \f$ \sigma = \sqrt{2} \sigma_m \sigma_M / \left( \sigma_M^2 + \sigma_m^2 \right)^{1/2} \f$
    - parameters[model_size()-1+2] ... asymmetry parameter \f$ A = (\sigma_M^2-\sigma_m^2)/(\sigma_M^2+\sigma_m^2) \f$ 
    - parameters[model_size()-1+3] ... rotation of Gaussian smoothing position angle (in model_image) in radians\n
  */
  class model_image_smooth : public model_image
  {
    
    private:
   //! Note that this is not defined yet becuase smoothing is applied in visibililty space directly.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

   
    public:
      model_image_smooth(model_image& model);
      virtual ~model_image_smooth() {};

      //! Size of the supplied model image, i.e., number of parameters expected.
      virtual inline size_t size() const { return _model.size()+3; };
    
      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const;

      //! Takes model parameters and generates 
      virtual void generate_model(std::vector<double> parameters);

      //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
    virtual std::complex<double> visibility(datum_visibility& d, double acc);
  
    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined since the closure phase of Gaussian images is identically zero.
  virtual double closure_phase(datum_closure_phase& d, double acc);


  //! \brief Returns closure amplitude computed from the image given a 
  //! datum_closure_phase object, containing all of the accoutrements.  
  //! While this provides access to the actual data value, the two could 
  //! be separated if necessary.  Also takes an accuracy parameter with 
  //! the same units as the data, indicating the accuracy with which the 
  //! model must generate a comparison value.  Note that this can be 
  //! redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double acc);

      //! Defines a set of processors provided to the model for parallel
      //! computation via an MPI communicator.  Only facilates code 
      //! parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
  
    private:
      model_image& _model;
      std::vector<double> _smoothing_parameters;
  };

};
#endif


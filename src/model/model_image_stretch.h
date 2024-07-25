/*!
  \file model_image_stretch.h   
  \author Paul Tiede
  \date April 2021
  \brief Header file for model_image_stretch.
  \details Smooths a model_image using a asymmetric Gaussian smoothing Kernel.
*/

#ifndef Themis_MODEL_IMAGE_STRETCH_
#define Themis_MODEL_IMAGE_STRETCH_

#include "model_image.h"
#include <vector>

namespace Themis {

  /*!
    \brief Defines a class that stretched a model image with some new axis ratio
    \details This appends a set of stretch parameters to the model_image
    parameters, consisting of:\n
    - parameters[model_size()-1+1] ... amount of stretch \f$ \tau = 1 - b/a \f$ where b is the semi-minor axis in radians and a semi-major
    - parameters[model_size()-1+2] ... rotation angle of the semi-major axis in radians measured east of north
  */
  class model_image_stretch : public model_image
  {
    
    private:
   //! Note that this is not defined yet because smoothing is applied in visibililty space directly.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

   
    public:
      model_image_stretch(model_image& model);
      virtual ~model_image_stretch() {};

      //! Size of the supplied model image, i.e., number of parameters expected.
      virtual inline size_t size() const { return _model.size()+2; };
    
      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const;

      //! Takes model parameters and generates 
      virtual void generate_model(std::vector<double> parameters);

      //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accommodate the possibility of using the analytical computation.
    virtual std::complex<double> visibility(datum_visibility& d, double acc);
  
    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this is redefined to accommodate the possibility of using the analytical computation.
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
      //! computation via an MPI communicator.  Only facilitates code 
      //! parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
  
    private:
      model_image& _model;
      std::vector<double> _smoothing_parameters;
  };

};
#endif


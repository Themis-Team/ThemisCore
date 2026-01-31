/*!
  \file model_image_sum.h
  \author Avery Broderick
  \date  November, 2018
  \brief Header file for the model_image_sum class, which sums different model images, generating a convenient way to create multi-component image models from single image components.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_SUM_H_
#define Themis_MODEL_IMAGE_SUM_H_

#include "model_image.h"
#include <vector>
#include <string>

namespace Themis {

/*!
  \brief Defines a summed image model.

  \details Each image component has its full complement of parameters, appended in order of the summation.  The resulting image is simply the sum of the intensity map from the each underlying object.  This sum is performed in the visibilty plane.

  Parameter list:\n
  - parameters[0] ... Model 1 parameters[0].\n
  ...\n
  - parameters[N1-1] ... Model 1 last parameter.\n
  - parameters[N1] ... Model 1 offset in x.\n
  - parameters[N1+1] ... Model 1 offset in y.\n
  - parameters[N1+2] ... Model 2 parameters[0].\n
  ...\n
  - parameters[N1+N2+1] ... Model 2 last parameter.\n
  - parameters[N1+N2+2] ... Model 2 offset in x.\n
  - parameters[N1+N2+3] ... Model 2 offset in y.\n
  ...\n
  - parameters[-1] ... Position angle (in model_image) in radians\n

Note that offsets for all components are included.  These are often not necessary, and can be set to fixed values via priors if needed.

*/
class model_image_sum : public model_image
{
 private:
  //! Note that this is not defined because there is no uniform image size/resolution specification.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:

  //! Constructor.  Takes a vector of pointers to model_image objects.
  model_image_sum(std::vector< model_image* > images, std::string offset_coordinates="Cartesian");

  //! Constructor.  Makes empty sum object, models can be added with add_model_image.
  model_image_sum(std::string offset_coordinates="Cartesian");

  //! Constructor.  Takes a pair of model_image objects.
  //model_image_sum(model_image& image1, model_image& image2);
  virtual ~model_image_sum() {};


  //! Add an image
  void add_model_image(model_image& image);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _size; };

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
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


  //!\brief Sets the MPI communicator if the underlying images are parallelized
  virtual void set_mpi_communicator(MPI_Comm comm);
    
  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
  virtual std::string model_tag() const;

  
 private:
  std::vector< model_image* > _images;
  std::string _offset_coordinates; //!< Choose coordinate system in which to specify offset ("Cartesian","polar")
  size_t _size; //!< Number of parameters total
  std::vector<double> _x; //!< x offset position in radians
  std::vector<double> _y; //!< y offset position in radians
  
};

};
#endif

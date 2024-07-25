/*!
  \file model_image_raster.h
  \author Avery Broderick
  \date October, 2017
  \brief Header file for the model_image_raster image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_RASTER_H_
#define Themis_MODEL_IMAGE_RASTER_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a raster image for image reconstruction.  This is
  an explicit example of a model_image object with variable parameter
  number.

  \details Generates a rastered pixel array, for which the parameters
  are the intensities at each pixel.  On construction the number of 
  pixels in the two directions are set, setting the resulting number
  of parameters.  Note that pixels (0,0) and (Nx-1,0) are reserved for
  weighting the center of light into the center.

  Parameter list:\n
  - parameters[0] ........ pixel 1,0
  - parameters[1] ........ pixel 2,0
  ...
  - parameters[Nx-2] ..... pixel Nx-3,0
  - parameters[Nx-1] ..... pixel 0,1
  - parameters[Nx] ....... pixel 1,1
  ...
  - parameters[Nx*Ny-1] .. pixel Nx-1,Ny-1

  \warning 
*/
class model_image_raster : public model_image
{
 private:
  //! Sets the image pixel values
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
 public:
  //! Constructs a model_image_raster object.  Takes the extents and number of pixels in each directions (xmin and xmax are the locations of the minimum and maximum pixel centers, etc.).
  model_image_raster(double xmin, double xmax, size_t Nx, double ymin, double ymax, size_t Ny);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _size; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);
  

 private:
  const double _xmin, _xmax;
  const size_t _Nx;
  const double _ymin, _ymax;
  const size_t _Ny;
  const size_t _size;

  bool _defined_raster_grid;
  
};

};
#endif

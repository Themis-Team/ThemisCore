/*!
  \file model_image_splined_raster.h
  \author Avery Broderick
  \date June, 2019
  \brief Header file for the model_image_splined_raster image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_SPLINED_RASTER_H_
#define Themis_MODEL_IMAGE_SPLINED_RASTER_H_

#include "model_image.h"
#include <vector>

namespace Themis {

/*!
  \brief Defines a splined raster image for image reconstruction.  The
  raster defined control points for an approximate spline, imposed in
  the u-v plane.

  \details Generates a rastered pixel array, for which the parameters
  are the intensities at each pixel.  On construction the number of 
  pixels in the two directions are set, setting the resulting number
  of parameters.  An approximate cubic spline is imposed with a control
  parameter \f$a\f$ that should usually be set to -0.5 (default) to -0.75 to 
  prevent excessive ringing.

  Parameter list:\n
  - parameters[0] ........ pixel 0,0
  - parameters[1] ........ pixel 1,0
  ...
  - parameters[Nx-1] ..... pixel Nx-1,0
  - parameters[Nx] ....... pixel 0,1
  - parameters[Nx+1] ..... pixel 1,1
  ...
  - parameters[Nx*Ny-1] .. pixel Nx-1,Ny-1
*/
class model_image_splined_raster : public model_image
{
 private:
  //! Sets the image pixel values
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
 public:
  //! Constructs a model_image_splined_raster object.  Takes the extents and number of pixels in each directions (xmin and xmax are the locations of the minimum and maximum pixel centers, etc.).
  model_image_splined_raster(double xmin, double xmax, size_t Nx, double ymin, double ymax, size_t Ny, double a=-0.5);

  //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
  void use_numerical_visibilities();
  //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
  void use_analytical_visibilities();

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _size; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);


  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  

 private:
  const double _xmin, _xmax;
  const size_t _Nx;
  const double _ymin, _ymax;
  const size_t _Ny;
  const size_t _size;
  bool _defined_raster_grid;


  const double _tpdx, _tpdy;
  const double _a;
  double cubic_spline_kernel_1d(double k) const;
  double cubic_spline_kernel(double u, double v) const;

  bool _use_analytical_visibilities;
};

};
#endif

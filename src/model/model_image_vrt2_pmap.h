/*!
  \file model_image_vrt2_pmap.h
  \author Avery Broderick
  \date  July, 2017
  \brief Header file for the model_image_vrt2_pmap image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_VRT2_PMAP_H_
#define Themis_MODEL_IMAGE_VRT2_PMAP_H_

#include "model_image.h"
#include "vrt2.h"
#include <string>
#include <vector>

namespace Themis {

/*!
  \brief Defines an interface for reading in a vrt2 image file (i.e.,
  pmap file). This is an explicit example of a model_image object.

  \details Reads in a preexisting polarizmation map output file (i.e.,
  pmap produced by the VRT2::PolarizationMap.output(<file>) function).
  Because this only accesses a single output file, usually this is 
  instantiated via the model_image_vrt2_library class, which specifies
  a collection of files.

  Parameter list:\n
  - parameters[0] ... Position angle (in model_image) in radians\n

  \warning 
*/
class model_image_vrt2_pmap : public model_image
{
 private:
  //! Reads in the pmap file specified and fills the rectalinear grid of intensities in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed (i.e., is size 0).
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
 public:
  //! Constructs a model_image_vrt2_pmap object.  Optionally takes a mass and distance (measured in cm).
  model_image_vrt2_pmap(double Mcm=VRT2::VRT2_Constants::M_SgrA_cm, double Dcm=VRT2::VRT2_Constants::D_SgrA_cm);
  //! Constructs a model_image_vrt2_pmap object.  Takes a pmap file name, setting the file initially.  Optionally takes a mass and distance (measured in cm). 
  model_image_vrt2_pmap(std::string pmap_file_name, double Mcm=VRT2::VRT2_Constants::M_SgrA_cm, double Dcm=VRT2::VRT2_Constants::D_SgrA_cm);
  virtual ~model_image_vrt2_pmap() {};

  //! Set the polarrization map file name
  void set_pmap_file(std::string pmap_file_name);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return 0; };
  
  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

 private:
  double _Mcm, _Dcm;
  std::string _pmap_file_name;

  bool _pmap_file_name_set; //!< True if a pmap file name has been set and visibilities may be computed.
  
};

};
#endif

/*!
  \file data_flux.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for datum and data objects containing fluxes.
  \details To be added
*/

#ifndef Themis_DATA_FLUX_H_
#define Themis_DATA_FLUX_H_

#include <string>
#include <vector>

namespace Themis {

/*! 
  \brief Defines a struct containing individual fluxes and ancillary information.

  \details Defines the atomic element of the flux data class.  This contains the flux with various additional accoutrements.  This will be passed to all objects that require a flux data value (e.g., likelihoods) and therefore can accrete additional elements but cannot rearrange elements internally to ensure backwards compatability.  Because data must never change during analysis, all elements are necessarily consts, requiring some minor gymnastics at initialization.

  \todo Define a standard for identifying additional flux information, e.g, bandwidths, instruments, etc.
*/
struct datum_flux
{
  datum_flux(double frequency, double Fnu, double err, double t=0, std::string Source="");

  // Visibility amplitude and error
  const double Fnu; //!< Measured flux value in Jy.
  const double err; //!< Error in measured flux Jy.

  // Frequency
  const double frequency; //!< Frequency in Hz.
  const double wavelength; //!< Wavelength in cm.
  
  // Timing -- always convert to time since Jan 1, 2000 in seconds
  const double tJ2000; //!< Time since Jan 1, 2000 in s, defaults to 0
  
  // Source
  const std::string Source; //!< Source identifier, defaults to \" \".
};

/*!
  \brief Defines a class containing a collection of flux datum objects with simple I/O.

  \details Collections of flux data are defined in data_flux, which includes simple I/O tools and provides access to a list of appropriately constructed datum_flux objects.

  \warning Currently assumes a fixed data file format.
  \todo Once data file formats crystalize, implement more generic or multi-format I/O options.
*/
class data_flux
{
 public:
  //! Defines a default, empty data_flux object.
  data_flux();
  //! Defines a default, data_flux filled by data in file \<file_name\>.
  data_flux(std::string file_name);
  //! Defines a default, data_flux filled by data in the collection of files in the vector \<file_name\>.
  data_flux(std::vector<std::string> file_name);

  //! Adds data from the file \<file_name\>.
  void add_data(std::string file_name);

  //! Adds data from a datum object
  void add_data(datum_flux& d);
  
  //! Returns the number of data points.
  inline size_t size() const { return _fluxes.size(); };

  //! Provides access to the atomic datum element.
  inline datum_flux& datum(size_t i) const { return (*_fluxes[i]); };

 private:
  std::vector< datum_flux* > _fluxes;
};

};

#endif

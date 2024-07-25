/*!
  \file data_closure_amplitude.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for datum and data objects containing closure amplitudes.
  \details To be added
*/

#ifndef Themis_DATA_CLOSURE_AMPLITUDE_H_
#define Themis_DATA_CLOSURE_AMPLITUDE_H_

#include <string>
#include <vector>

namespace Themis {

/*! 
  \brief Defines a struct containing individual closure amplitudes and ancillary information.

  \details Defines the atomic element of the closure amplitude data class.  This contains the closure amplitude with various additional accoutrements.  This will be passed to all objects that require a closure amplitude data value (e.g., likelihoods) and therefore can accrete additional elements but cannot rearrange elements internally to ensure backwards compatability.  Because data must never change during analysis, all elements are necessarily consts, requiring some minor gymnastics at initialization.  Note that here we define the closure amplitude is defined as \f$(|V_{12}| |V_{34}|) / (|V_{13}| |V_{24}|) \f$.

  \warning Currently forces u1+u2+u3+u4=0 and v1+v2+v3+v4=0.
  \todo Define a standard for identifying stations and sources.
*/
struct datum_closure_amplitude
{
  datum_closure_amplitude(double u1, double v1, double u2, double v2, double u3, double v3, double CA, double err, double frequency=230e9, double t=0, std::string Station1="", std::string Station2="", std::string Station3="", std::string Station4="", std::string Source="");

  // Visibility amplitude and error
  const double CA; //!< Closure amplitude value.
  const double err; //!< Closure amplitude error.

  // uv position, measured in lambda
  const double u1; //!< u position of station 1, measured in lambda.
  const double v1; //!< v position of station 1, measured in lambda.
  const double u2; //!< u position of station 2, measured in lambda.
  const double v2; //!< v position of station 2, measured in lambda.
  const double u3; //!< u position of station 3, measured in lambda.
  const double v3; //!< v position of station 3, measured in lambda.
  const double u4; //!< u position of station 4, measured in lambda.
  const double v4; //!< v position of station 4, measured in lambda.

  // Frequency
  const double frequency; //!< Frequency in Hz, defaults to 230 GHz.
  const double wavelength; //!< Wavelength in cm.
  
  // Timing -- always convert to time since Jan 1, 2000 in seconds
  const double tJ2000; //!< Time since Jan 1, 2000 in s, defaults to 0.
  
  // Stations
  const std::string Station1; //!< Station 1 identifier, defaults to "".
  const std::string Station2; //!< Station 2 identifier, defaults to "".
  const std::string Station3; //!< Station 3 identifier, defaults to "".
  const std::string Station4; //!< Station 4 identifier, defaults to "".

  // Source
  const std::string Source; //!< Source identifier, defaults to "".
};
  
  
/*!
  \brief Defines a class containing a collection of closure amplitude datum objects with simple I/O.

  \details Collections of closure amplitude data are defined in data_closure_amplitude, which includes simple I/O tools and provides access to a list of appropriately constructed datum_closure_amplitude objects.

  \warning Currently assumes a fixed data file format.
  \todo Once data file formats crystalize, implement more generic or multi-format I/O options.
*/
class data_closure_amplitude
{
 public:
  //! Defines a default, empty data_closure_amplitude object.
  data_closure_amplitude();
  //! Defines a default, data_closure_amplitude filled by data in file \<file_name\>.
  data_closure_amplitude(std::string file_name);
  //! Defines a default, data_closure_amplitude filled by data in the collection of files in the vector \<file_name\>.
  data_closure_amplitude(std::vector<std::string> file_name);

  //! Adds data from the file \<file_name\>.
  void add_data(std::string file_name);

  //! Adds data from a datum object
  void add_data(datum_closure_amplitude& d);

  //! Returns the number of data points.
  inline size_t size() const { return _closure_amplitudes.size(); };

  //! Provides access to the atomic datum element.
  inline datum_closure_amplitude& datum(size_t i) const { return (*_closure_amplitudes[i]); };

  private:
    std::vector< datum_closure_amplitude* > _closure_amplitudes;
};

};
#endif

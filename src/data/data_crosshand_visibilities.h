/*!
  \file data_crosshand_visibilities.h
  \author Avery E. Broderick
  \date  March, 2020
  \brief Header file for datum and data objects containing full RR, LL, RL and LR visibilities.
  \details To be added
*/

#ifndef Themis_DATA_CROSSHAND_VISIBILITIES_H_
#define Themis_DATA_CROSSHAND_VISIBILITIES_H_

#include <complex>
#include <string>
#include <vector>

namespace Themis {

/*! 
  \brief Defines a struct containing individual visibilities and ancillary information.

  \details Defines the atomic element of the visibility data class. This contains the RR, LL, RL, and LR visibilities with various additional accoutrements.  This will be passed to all objects that require a visibility data value (e.g., likelihoods) and therefore can accrete additional elements but cannot rearrange elements internally to ensure backwards compatability.  Because data must never change during analysis, all elements are necessarily consts, requiring some minor gymnastics at initialization.

  Within the datum_visibility class, station codes are arbitrary, but elsewhere may be assumed to correspond to either the one- or two-letter station codes identified in https://eventhorizontelescope.teamwork.com/#notebooks/97662:
  
  \image html images/eht_station_codes.png "List of one- and two-letter EHT station codes."

  The visibilities are pre-rotated by the the field rotation angle, and thus this must be folded into subsequent D-term calibration.
*/
struct datum_crosshand_visibilities
{
  datum_crosshand_visibilities(double u, double v, double phi1, double phi2, std::complex<double> RR, std::complex<double> RRerr, std::complex<double> LL, std::complex<double> LLerr, std::complex<double> RL, std::complex<double> RLerr, std::complex<double> LR, std::complex<double> LRerr, double frequency=230e9, double t=0, std::string Station1="", std::string Station2="", std::string Source="");

  // Visibility amplitude and error
  const std::complex<double> RR, LL, RL, LR;   //!< Visibility value in Jy.
  const std::complex<double> RRerr, LLerr, RLerr, LRerr; //!< Visibility error in Jy.

  // uv position, measured in lambda
  const double u; //!< u position measured in lambda.
  const double v; //!< v position measured in lambda.
  const double phi1; //!< field rotation angle at station1  measured in radians.
  const double phi2; //!< field rotation angle at station1  measured in radians.

  // Frequency
  const double frequency; //!< Frequency in Hz, defaults to 230e9
  const double wavelength; //!< Wavelength in cm
  
  // Timing -- always convert to time since Jan 1, 2000 in seconds
  const double tJ2000; //!< Time since Jan 1, 2000 in s, defaults to 0
  
  // Stations
  const std::string Station1; //!< Station 1 identifier, defaults to ""
  const std::string Station2; //!< Station 2 identifier, defaults to ""

  // Source
  const std::string Source; //!< Source identifier, defaults to ""
};

/*!
  \brief Defines a class containing a collection of crosshand visibilities datum objects with simple I/O.

  \details Collections of crosshand visibilities data are defined in data_crosshand_visibilities, which includes simple I/O tools and provides access to a list of appropriately constructed datum_visibility objects.

  \warning Currently assumes a fixed data file format.
  \todo Once data file formats crystalize, implement more generic or multi-format I/O options.
*/
class data_crosshand_visibilities
{
 public:
  //! Defines a default, empty data_visibility object.
  data_crosshand_visibilities();
  //! Defines a default, data_visibility filled by data in file \<file_name\>.
  //! time_type specifies the format of the time field in the importated data. Currently two formats are implemented:
  //! - HHMM e.g. 1230
  //! - HH   e.g. 12.5
  data_crosshand_visibilities(std::string file_name, const std::string time_type="HHMM", bool read_frequency=false);
  //! Defines a default, data_visibility filled by data in the collection of files in the vector \<file_name\>.
  //! time_type specifies the format of the time field in the importated data. Currently two formats are implemented:
  //! - HHMM e.g. 1230
  //! - HH   e.g. 12.5
  //! If no time_type is specified it assumes all the data files are using "HHMM".
  data_crosshand_visibilities(std::vector<std::string> file_name, const std::vector<std::string> time_type=std::vector<std::string> ());

  //! Adds data from the file \<file_name\>.
  void add_data(std::string file_name, const std::string time_type="HHMM", bool read_frequency=false);

  //! Adds data from a datum object
  void add_data(datum_crosshand_visibilities& d);

  //! Returns the number of data points.
  inline size_t size() const { return _visibilities.size(); };

  //! Provides access to the atomic datum element.
  inline datum_crosshand_visibilities& datum(size_t i) const { return (*_visibilities[i]); };

  //! Sets default frequency in read data function
  void set_default_frequency(double frequency) { _frequency=frequency; };

 private:
  std::vector< datum_crosshand_visibilities* > _visibilities;
  double _frequency;
};

};

#endif

/*!
  \file data_closure_phase.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for datum and data objects containing closure phases.
  \details To be added
*/

#ifndef Themis_DATA_CLOSURE_PHASE_H_
#define Themis_DATA_CLOSURE_PHASE_H_

#include <string>
#include <vector>

namespace Themis {

/*! 
  \brief Defines a struct containing individual closure phases and ancillary information.

  \details Defines the atomic element of the closure phase data class.  This contains the closure phase with various additional accoutrements.  This will be passed to all objects that require a closure phase data value (e.g., likelihoods) and therefore can accrete additional elements but cannot rearrange elements internally to ensure backwards compatability.  Because data must never change during analysis, all elements are necessarily consts, requiring some minor gymnastics at initialization.  Note that here we define the closure phase to be \f${\rm arg}(V_{12} V_{23} V_{31})\f$.

  \warning Currently forces u1+u2+u3=0 and v1+v2+v3=0.
  \todo Define a standard for identifying stations and sources.
*/
 struct datum_closure_phase
{
  datum_closure_phase(double u1, double v1, double u2, double v2, double CP, double err, double frequency=230e9, double t=0, std::string Station1="", std::string Station2="", std::string Station3="", std::string Source="");

  // Closure phase and error
  const double CP;  //!< Closure phase in degrees.
  const double err; //!< Closure phase error in degrees.

  // uv position, measured in lambda
  const double u1; //!< u position of station 1, measured in lambda.
  const double v1; //!< v position of station 1, measured in lambda.
  const double u2; //!< u position of station 2, measured in lambda.
  const double v2; //!< v position of station 2, measured in lambda.
  const double u3; //!< u position of station 3, measured in lambda.
  const double v3; //!< v position of station 3, measured in lambda.

  // Frequency
  const double frequency; //!< Frequency in Hz, defaults to 230 GHz.
  const double wavelength; //!< Wavelength in cm.
  
  // Timing -- always convert to time since Jan 1, 2000 in seconds
  const double tJ2000; //!< Time since Jan 1, 2000 in s, defaults to 0.
  
  // Stations
  const std::string Station1; //!< Station 1 identifier, defaults to "".
  const std::string Station2; //!< Station 2 identifier, defaults to "".
  const std::string Station3; //!< Station 3 identifier, defaults to "".

  // Source
  const std::string Source; //!< Source identifier, defaults to "".
};


/*!
  \brief Defines a class containing a collection of closure phases datum objects with simple I/O.

  \details Collections of closure phase data are defined in data_closure_phase, which includes simple I/O tools and provides access to a list of appropriately constructed datum_closure_phase objects.

  \warning Currently assumes a fixed data file format.
  \todo Once data file formats crystalize, implement more generic or multi-format I/O options.
*/
class data_closure_phase
{
 public:
  //! Defines a default, empty data_closure_phase object.
  data_closure_phase();
  //! Defines a default, data_closure_phase filled by data in file \<file_name\>.
  //! time_type specifies the format of the time field in the importated data. Currently two formats are implemented:
  //! - HHMM e.g. 1230
  //! - HH   e.g. 12.5
  //! themis convention defines the sign convention used for the Fourier transforms.
  //! - true uses exp(-2 i pi ...)
  //! - false    uses exp(+2 i pi ...)
  data_closure_phase(std::string file_name, const std::string time_type="HH", bool themis_convention=true);
  //! Defines a default, data_closure_phase filled by data in the collection of files in the vector \<file_name\>.
  //! time_type specifies the format of the time field in the importated data. Currently two formats are implemented:
  //! - HHMM e.g. 1230
  //! - HH   e.g. 12.5
  //! If no time_type is specified it assumes all the data files are using "HH".
  //! themis_convention defines the sign convention used for the Fourier tranforms.
  //! - True (Themis convention) uses exp(-2 i pi ...)
  //! - False (EHT convention)    uses exp(+2 i pi ...)
  data_closure_phase(std::vector<std::string> file_name, const std::vector<std::string> time_type=std::vector<std::string> (), std::vector<bool> themis_convention=std::vector<bool> ());

  //! Adds data from the file \<file_name\>.
  void add_data(std::string file_name, const std::string time_type="HH", bool themis_convention=true);

  //! Adds data from a datum object
  void add_data(datum_closure_phase& d);
  

  //! Returns the number of data points.
  inline size_t size() const { return _closure_phases.size(); };

  //! Provides access to the atomic datum element.
  inline datum_closure_phase& datum(size_t i) const { return (*_closure_phases[i]); };

 private:
  std::vector< datum_closure_phase* > _closure_phases;
};

};

#endif

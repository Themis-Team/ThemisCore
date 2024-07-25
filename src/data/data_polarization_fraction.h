/*!
  \file data_polarization_fraction.h
  \author Roman Gold
  \date Apr 2017
  \brief Header file for a fractional polarization data class.  
  \details Collections of mbreve data
 \f$\breve{m}:=(|\tilde{Q}+i\tilde{U}|)/|\tilde{I}|\f$.  are defined
 in data_polarization_fraction, which includes simple I/O tools and
 provides access to a list of appropriately constructed datum_mbreve
 objects.
*/


#ifndef Themis_DATA_POLARIZATION_FRACTION_H_
#define Themis_DATA_POLARIZATION_FRACTION_H_

#include <string>
#include <vector>

namespace Themis {

/*!  \brief Defines a struct containing a collection of linear
  polarization fractions in the visbility domain,
  \f$\breve{m}:=(|\tilde{Q}+i\tilde{U}|)/|\tilde{I}|\f$.
  
  \details This will be passed to all objects that
  require an mbreve data value (e.g., likelihoods) and therefore can
  accrete additional elements but cannot rearrange elements internally
  to ensure backwards compatability.  Because data must never change
  during analysis, all elements are necessary consts, requiring some
  minor gymnastics at initialization.
*/
struct datum_polarization_fraction
{
  datum_polarization_fraction(double u, double v, double mbreve_amp, double err, double frequency=230e9, double t=0, std::string Station1="", std::string Station2="", std::string Source="");

  // Visibility amplitude and error
  const double mbreve_amp;   //!< fractional linear polarization amplitude in the visibility domain [dim less]
  const double err; //!< error in mbreve_amp [dim less]
  // const double mbreve_amp_err; //!< error in mbreve_amp [dim less]

  // uv position, measured in lambda
  const double u; //!< u position measured in [lambda].
  const double v; //!< v position measured in [lambda].

  // Frequency
  const double frequency; //!< Observing frequency in [Hz], defaults to 230e9
  const double wavelength; //!< Wavelength in [cm]
  
  // Timing -- always convert to time since Jan 1, 2000 in seconds
  const double tJ2000; //!< Time since Jan 1, 2000 in s, defaults to 0
  
  // Stations
  const std::string Station1, Station2; //!< Source identifier, defaults to ""

  // Source
  const std::string Source; //!< Source identifier, defaults to ""
};


/*!
  \brief Defines a class containing a collection of linear polarization fraction amplitude datum objects with simple I/O.

  \details Collections of linear polarizationo fraction amplitude data are defined in data_polarization_fraction, which includes simple I/O tools and provides access to a list of appropriately constructed datum_polarization_fraction objects.

  \warning Currently assumes a fixed data file format.
  \bug Station name conventions are inconsistent with visibility amplitude data!
  \todo Once data file formats crystalize, implement more generic or multi-format I/O options.
  
*/
class data_polarization_fraction
{
  public:
    data_polarization_fraction();
    data_polarization_fraction(std::string file_name);
    data_polarization_fraction(std::vector<std::string> file_name);

    void add_data(std::string file_name);


    //! Adds data from a datum object
    void add_data(datum_polarization_fraction& d);

    
    inline size_t size() const { return _mbreve.size(); };
    inline datum_polarization_fraction& datum(size_t i) const { return (*_mbreve[i]); };

  private:
    std::vector< datum_polarization_fraction* > _mbreve;
};

};

#endif

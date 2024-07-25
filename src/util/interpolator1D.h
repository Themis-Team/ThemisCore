/*!
  \file interpolator1D.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for a multipurpose linear interpolation function.
  \details Implements interpolatoion functions for 1d data. Currently we have linear and monotone cubic interpolation methods.
*/

#ifndef Themis_INTERPOLATOR1D_H
#define Themis_INTERPOLATOR1D_H

#include <string>
#include <vector>
#include <iostream>

namespace Themis {

/*! 
  \brief Defines a general purpose 1D linear interpolator.

  \details Takes a monotonic, but not necessarily regular, pair of independent and dependent variables, x and f, and encapsulates linear interpolation.  Can also read these in from a file given the column and number of headers (assumes the independent variable is listed in the first column).

*/
class Interpolator1D
{
 public:
  //! Default constructor, x and f must be set before operator() can be called.
  Interpolator1D(std::string method="linear");
  //! Construct Interpolator1D object with x and f defined.
  Interpolator1D(std::vector<double> x, std::vector<double> f, std::string method="linear");
  //! Construct Interpolator1D object with x and f as read in from \<ifile\> in columns 1 (x) and \<column\> (f) after ignoring the first \<headers\> lines.
  Interpolator1D(std::string ifile, size_t column, size_t headers=0, std::string method="linear");

  //! Sets/resets the tabulated independent (x) and dependent (f) variable values.
  void set_tables(std::vector<double> x, std::vector<double> f);

  //! Linear interpolation function
  double operator()(double x) const;

 private:
  std::vector<double> _x, _y, _s, _dx, _dy;
  std::vector<double> _c1,_c2,_c3; //Interpolation coeff.
  std::string _method;

  //!< Linear interpolation or linear
  inline double linear(double x) const;
  //!< Monotone cubic, or mcubic interpolation using the https://epubs.siam.org/doi/10.1137/0717021 Fritch&Carlson method.
  inline double mcubic(double x) const;

  void set_mcubic();
};

};
#endif

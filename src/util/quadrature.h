/*!
 \file quadrature.h
 \author Avery Broderick
 \date April 24, 2019
 \brief Header file for a general purpose Gaussian Quadrature class that implements a variety of integration schemes
 \details Uses a variety of integration schemes to integrate a 1d function. The user must first declare the function in
          the appropriate integrand class for this to work.
*/





//////////////////////////////////////////
//Does integration for some function using a variety of methods
#ifndef Themis_QUADRATURE_H
#define Themis_QUADRATURE_H

#include <algorithm>
#include <string>
#include <cmath>
#include <iostream>
namespace Themis {

/*! 
  \brief Defines a general purpose 1D Gaussian quadrature class that implements a variety of weighting schemes.

  \details Implemented methods include: Legendre and Hermite weights.

  \warning Assumes that the integrand belongs to the Intergrand class.
*/


class Integrand
{
 /*!
  \brief Defines a general purpose integrand class that must be passed to any integration schemes that are defined below
*/
 public:
  virtual ~Integrand() {};
  //! Defines the integrand function
  virtual double operator()(double) const = 0;
};

class GaussianQuadrature
{
 public:
  //! Constructor that sets the number of fixed points to be used by the Gaussian quadrature.
  //! Defaults to 2048 and Legendre quadrature weights.
  GaussianQuadrature(int N=2048);
  ~GaussianQuadrature();

  //! Returns the integral of the integrand from a to b.
  double integrate(Integrand& i, double a, double b);

  //! Returns the number of abissca points used in the Gaussian quadrature
  int npoints() { return _N; };

  // Set the weights to be some "Classical function"
  //   Types include:
  //                  "GaussLegendre support [-1,1] (This is the default)"
  //                  "GaussHermite support [-inf, inf]"
  //
  void set_weights(std::string type);
  
 private:
  //Number of points in the to be used by the Gaussian quadrature
  int _N;
  //Holds the weights
  double *_x, *_v;

  //Constructs the weights for the Gaussian quadrature using the methods described in Numerical Recipes
  void gauss_legendre(double x1, double x2, double x[], double w[], int n) const;
  void gauss_hermite(double x[], double w[], int n) const;

  
};

};
#endif

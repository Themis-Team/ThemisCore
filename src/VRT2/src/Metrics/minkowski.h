/****************************************/
/****** Header file for metric.cpp ******/
/****** Contains Minkowski Base Class ******/
/* NOTES: 
   Flat Space Spherical by default
*/
/****************************************/

// Only include once
#ifndef VRT2_MINKOWSKI_H
#define VRT2_MINKOWSKI_H

// Standard Library Headers
#include <cmath>

// Local Headers
#include "metric.h"

namespace VRT2 {

class Minkowski : public Metric
{
 public:
  Minkowski();
  virtual void initialize(); // sets the Ng/NDg stuff (one time initialization)
  virtual ~Minkowski() { };

  // Set position
  virtual void get_fcns(); // every time initialization

  // Elements
  virtual double g(int);     // g_ij (non-zero elements only!)
  virtual double ginv(int);  // g^ij (non-zero elements only!)
  virtual double detg();     // Determinant
  virtual double Dg(int);    // g_ij,k (non-zero elements only!)
  virtual double Dginv(int); // g^ij,_k (non-zero elements only!)

  // Christoffel Symbols
  virtual double Gamma(int); // Gamma^i_jk (non-zero elements only!)

  // Horizon
  virtual double horizon() const;

  // Parameters
  virtual double mass() const { return 0; };
  virtual double ang_mom() const { return 0; };
  
  // Check for defined quantities (|2=g,|3=ginv,|5=detg,|7=Dg,|11=Dginv,|13=Gamma)

 private:
  // Useful Values
  double _r, _sn;
};

};

#endif

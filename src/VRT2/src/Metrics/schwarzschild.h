/***********************************************/
/****** Header file for schwarzschild.cpp ******/
/* NOTES: 
   Returns schwarzschild metric in standard coords.
*/
/***********************************************/

// Only include once
#ifndef VRT2_SCHWARZSCHILD_H
#define VRT2_SCHWARZSCHILD_H

// Standard Library Headers
#include <cmath>

// Local Headers
#include "metric.h"

namespace VRT2 {
class Schwarzschild : public Metric {

 public:
  Schwarzschild(double Mass);
  virtual void initialize(); // sets the Ng/NDg stuff (one time initialization)
  virtual ~Schwarzschild() { };

  // Set position
  virtual void get_fcns(); // every time initialization

  // Elements
  virtual double g(int); // g_ij (non-zero elements only!)
  virtual double ginv(int); // g^ij (non-zero elements only!)
  virtual double detg(); // Determinant
  virtual double Dg(int); // g_ij,k (non-zero elements only!)
  virtual double Dginv(int); // g^ij,_k (non-zero elements only!)

  // Christoffel Symbols
  virtual double Gamma(int);

  // Horizon
  virtual double horizon() const;

  // Parameters
  virtual double mass() const;
  virtual double ang_mom() const { return 0; };

  // Check for defined quantities (|2=g,|3=ginv,|5=detg,|7=Dg,|11=Dginv)
  
  // ISCO (specialized for this and Kerr only so far!)
  double rISCO() const { return 6.0*_M;  };

  
 private:
  // Mass
  const double _M;

  // Useful Values not defined in Metric
  double _r,_sn,_sigma;
};
};
#endif

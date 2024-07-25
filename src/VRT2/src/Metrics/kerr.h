/**************************************/
/****** Header file for kerr.cpp ******/
/* NOTES: 
   Returns Kerr metric in Boyer-Lindquist
     coords.
*/
/**************************************/

// Only include once
#ifndef VRT2_KERR_H
#define VRT2_KERR_H

// Standard Library Headers
#include <cmath>

// Local Headers
#include "metric.h"

namespace VRT2 {

class Kerr : public Metric
{
 public:
  Kerr(double Mass, double Spin);
  virtual void initialize(); // sets the Ng/NDg stuff (one time initialization)
  virtual ~Kerr() {};

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
  virtual double ang_mom() const;

  // Check for defined quantities (|2=g,|3=ginv,|5=detg,|7=Dg,|11=Dginv)
  
  // ISCO (specialized for this and Schwarzchild only so far!)
  double rISCO() const;


 private:
  const double _M, _a;
  // Useful Values not defined in Metric
  double _r,_sn,_cs,_r2,_sn2,_cs2,_sncs,_a2,_Delta,_Sigma,_ra2;  
};

};

#endif

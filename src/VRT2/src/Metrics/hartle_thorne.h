/**************************************/
/****** Header file for hartle_thorne.cpp ******/
/* NOTES: 
   Returns Hartle Thorne metric in Boyer-Lindquist
     coords.
*/
/**************************************/

// Only include once
#ifndef VRT2_HARTLE_THORNE_H
#define VRT2_HARTLE_THORNE_H

// Standard Library Headers
#include <cmath>

// Local Headers
#include "metric.h"
#include "interpolator2D.h"

namespace VRT2 {

class HartleThorne : public Metric
{
 public:
  HartleThorne(double Mass, double Spin, double epsilon);
  virtual void initialize(); // sets the Ng/NDg stuff (one time initialization)
  virtual ~HartleThorne() { };

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

  double rISCO() const;

  // Parameters
  virtual double mass() const;
  virtual double ang_mom() const;
  virtual double quad_mom() const;

  // Check for defined quantities (|2=g,|3=ginv,|5=detg,|7=Dg,|11=Dginv)
  
 private:
  const double _M, _a, _ep;
  // Useful Values
  double _r,_sn,_cs,_r2,_sn2,_cs2,_sncs,_a2,_Delta,_Sigma,_ra2,_M2,_M3,_r3,_sw,_sw2,_lg,_2M,_c2s;  


  double _rh, _risco;



  // Utility functions for zeroing the equation defining the ISCO
  double get_ISCO();
  double ISCOFunc(double r);
  double rtflsp(double x1, double x2, double xacc);
  double zriddr(double x1, double x2, double xacc);

};

};
#endif

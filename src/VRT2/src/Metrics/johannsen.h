/**********************************************************/
/*****Header file for Johannsen metric PRD 88 04402 (2013)*/
/****  NOTE: Assumes Boyer-Lindquist style coordinates.
 *
 */


#ifndef VRT2_JOHANNSEN_H
#define VRT2_JOHANNSEN_H

#include <cmath>


#include "metric.h"


namespace VRT2 {

class Johannsen : public Metric
{
 public: 
  Johannsen(double Mass, double spin, double alpha13, double alpha22, double alpha52, double epsilon, double beta);
  virtual void initialize(); //creates hash table and stuff for metric
  virtual ~Johannsen() { };

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

  double rISCO();

  // Parameters
  virtual double mass() const;
  virtual double ang_mom() const;

  double alpha13() const;
  double alpha22() const;
  double alpha52() const;
  double epsilon() const;
  double beta() const;

 private:
  const double _M, _a, _alpha13, _alpha22, _alpha52, _epsilon, _beta;
  double _M2, _a2, _M3, _2M;

  //Uesfule stored values for efficency
  double _r, _sn, _cs, _r2, _sn2, _cs2, _sncs, _ra2, _a2sn2;
  
  //Stored funstion values
  double _f, _A1, _A2, _A5, _A22, _A1A2, _tDelta, _tSigma, _d, _d2, _d3, _X, _pg_tt, _pg_tp, _pg_pp;

  //stored horizon and isco radius
  double _rh, _risco;

  //Utiility functions for the isco and stuff
    //Energy of a equatorial orbit in Keplerian motion
  double energy(double r);
  double ISCOFunc(double r);
  double get_ISCO();
  //double rtflsp(double x1, double x2, double xacc);
  //double zriddr(double x1, double x2, double xacc);
  double grad_descent(double x0, double xacc);

};


};
#endif

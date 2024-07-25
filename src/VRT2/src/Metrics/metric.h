/****************************************/
/****** Header file for metric.cpp ******/
/****** Contains Metric Base Class ******/
/* NOTES: 
   Flat Space Spherical by default
*/
/****************************************/

// Only include once
#ifndef VRT2_METRIC_H
#define VRT2_METRIC_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <bitset>

namespace VRT2 {
class Metric {

 public:
  Metric();
  Metric(int Ng, int NDg, int NG);
  virtual ~Metric() {};

  virtual void initialize(); // sets the Ng/NDg stuff (one time initialization)

  // Set position
  void reset( const double[] );
  void reset( double,double,double,double );
  void reset( const std::valarray<double>& );
  void reset( const std::vector<double>& );

  virtual void get_fcns(); // every time initialization

  // Get current position
  std::valarray<double> local_position() const { return _x; };

  // Elements
  virtual double g(int) = 0;     // g_ij (non-zero elements only!)
  virtual double ginv(int) = 0;  // g^ij (non-zero elements only!)
  virtual double detg() = 0;     // Determinant
  virtual double Dg(int) = 0;    // g_ij,k (non-zero elements only!)
  virtual double Dginv(int) = 0; // g^ij,_k (non-zero elements only!)

  // Christoffel Symbols
  virtual double Gamma(int) = 0; // Gamma^i_jk (non-zero elements only!)

  // Elements by ijk
  void mk_hash_table();
  double g(int i, int j);            // g_ij
  double ginv(int i, int j);         // g^ij
  double Dg(int i, int j, int k);    // g_ij,k
  double Dginv(int i, int j, int k); // g^ij,_k
  double Gamma(int i, int j, int k); // Gamma^i_jk

  // Horizon
  virtual double horizon() const;

  // Parameters
  virtual double mass() const { return 0; };
  virtual double ang_mom() const { return 0; };
  virtual double quad_mom() const { return 0; };

  // integer arrays with non-zero values
  const int Ng, NDg, NG;
  std::valarray<int> gi, gj;
  std::valarray<int> Dgi, Dgj, Dgk; 
  std::valarray<int> Gi, Gj, Gk;
  
  // Check for defined quantities 
  // (0=g,1=ginv,2=detg,3=Dg,4=Dginv,5=Gamma)
  std::bitset<6> defined_list;

  // Derivatives Check
  void derivatives_check(double x[], double h);
  
  // Output (for debugging)
  friend std::ostream& operator<<(std::ostream&, Metric&);


  // Last function run in initialization to check metric geometry against
  // that expected from fourvector.h
  void check_compiler_optimization();

  //protected:
  // Local x
  std::valarray<double> _x;
  // Quantities inside Metric which are accessible via the Elements fcns
  double _detg;
  std::valarray<double> _g, _ginv, _Dg, _Dginv, _Gamma;
  // Counters/Indicies
  int _gi[4][4], _Dgi[4][4][4], _Gi[4][4][4];

 private:
  // Useful Values
  double _r, _sn;

};
};
#endif

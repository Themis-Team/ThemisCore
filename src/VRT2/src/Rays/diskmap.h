/**********************************************/
/****** Header file for polarizationmap.cpp ******/
/* Notes:
   Generates a polarization map by tracing rays
*/
/**********************************************/

// Only include once
#ifndef VRT2_DISKMAP_H
#define VRT2_DISKMAP_H

// Standard Library Headers
#include <valarray>
#include <vector>
#include <cmath>
#include <math.h>
using namespace std;
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

// Local Headers
#include "metric.h"
#include "fourvector.h"
#include "vrt2_constants.h"
#include "ray.h"
#include "progressindicator.h"
#include "sc_disk_map.h"

// MPI Header
#ifdef MPI_MAP
#include <mpi.h>
#endif

namespace VRT2 {
class DiskMap{
 public:
  // Constructor and Destructor
  DiskMap(Metric &g, Ray &ray, SC_DiskMap &stop, int verbosity=1);
  ~DiskMap() {};
  
  // Set initial conditions surface
  void set_R_THETA(double,double);

  // Set progress stream
  void set_progress_stream(std::ostream &);
  void set_progress_stream(std::string);

  // Generate Polarization DiskMap
  void generate(double,double,int,double,double,int);
  void generate(double,double,double,int,double,double,int);

  // Generate Multiple Rays
  void eta_section(double,double,double,int);
  void eta_section(double,double,double,double,int);
  void xi_section(double,double,double,int);
  void xi_section(double,double,double,double,int);

  // Polarization Information functions
  double I(int,int); // returns I at int,int
  double Q(int,int); // returns Q at int,int
  double U(int,int); // returns U at int,int
  double V(int,int); // returns V at int,int
  double tau(int,int); // returns tau_int at int,int
  double D(int,int); // return D_int at int,int

  // Initial Position functions
  int init_conds(double,double,double,FourVector<double>&,FourVector<double>&);

  // File Input/Output functions
  void output_map(std::string);
  void output_map(std::ostream&);
  
 protected:
  Metric &_g; // local metric
  Ray &_ray; // Ray generator.  Contains all we need to know!
  SC_DiskMap &_stop; // To reset before running rays.
  int _verbosity; // how much stuff to print out (0=nothing!)


  double _R, _THETA;
  double _frequency,_frequency0; // Frequency of DiskMap
  double _xi_lo, _xi_hi; // horizontal range
  double _eta_lo, _eta_hi; // vertical range
  int _N_xi, _N_eta; // number of steps
  std::vector<double> _I, _Q, _U, _V; // Matrix with I,Q,U,V
  std::vector<double> _tau, _D; // Diagnostics

  std::ostream *_pstream; // Output stream for progress counters
  std::ofstream _lpstream; // Local output stream for setting stream by file name

  // MPI stuff
  int _rank, _size;
#ifdef MPI_MAP
  void collect(int rank=0); // Collect into single array at end
#endif
};
};
#endif

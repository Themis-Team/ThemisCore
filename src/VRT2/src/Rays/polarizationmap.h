/**********************************************/
/****** Header file for polarizationmap.cpp ******/
/* Notes:
   Generates a polarization map by tracing rays
*/
/**********************************************/

// Only include once
#ifndef VRT2_POLARIZATIONMAP_H
#define VRT2_POLARIZATIONMAP_H

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
#include "interpolator2D.h"
#include "vrt2_globs.h"

// MPI Header
#ifdef VRT2_USE_MPI_MAP
#include <mpi.h>
#endif

namespace VRT2 {
class PolarizationMap{
 public:
  // Constructor and Destructor
#ifdef VRT2_USE_MPI_MAP
  PolarizationMap(Metric &g, Ray &ray, double M, double D, MPI_Comm& comm, int verbosity=1, bool rescale_intensity=true);
#endif
  PolarizationMap(Metric &g, Ray &ray, double M, double D, int verbosity=1, bool rescale_intensity=true);
  ~PolarizationMap();

  // Set frequency0 (scale/default frequency in Hz)
  void set_f0(double);

  // Set initial conditions surface
  void set_R_THETA(double,double);

  // Set progress stream
  void set_progress_stream(std::ostream &);
  void set_progress_stream(std::string);

  // Determine coef to convert N to I
  double get_N_to_I();

  // Generate Polarization PolarizationMap
  void generate(double,double,int,double,double,int);
  void generate(double,double,double,int,double,double,int);

  // Access map
  void get_map(std::valarray<double>& xi, std::valarray<double>& eta, std::valarray<double>& I, std::valarray<double>& Q, std::valarray<double>& U, std::valarray<double>& V);

  // Refine map with selective mesh refinement
  void refine(double refine_factor=5e-3);
  void old_refine();
  //  void background_subtracted_refine(std::vector<double>& xi, std::vector<double>& eta, std::vector<double>& I, std::vector<double>& Q, std::vector<double>& U, std::vector<double>& V);
  void set_background_intensity();
  void unset_background_intensity(); // default

  // Integrate IQUV over region
  std::valarray<double> integrate();
  double I_int();
  double Q_int();
  double U_int();
  double V_int();

  // Generate Multiple Rays
  void eta_section(double,double,double,int);
  void eta_section(double,double,double,double,int);
  void xi_section(double,double,double,int);
  void xi_section(double,double,double,double,int);
  void eta_point_section(double[],int);
  void eta_point_section(double,double[],int);

  // Polarization Information functions
  double I(int,int); // returns I at int,int
  double Q(int,int); // returns Q at int,int
  double U(int,int); // returns U at int,int
  double V(int,int); // returns V at int,int
  double tau(int,int); // returns tau_int at int,int
  double D(int,int); // return D_int at int,int

  // Initial Position functions
  int init_conds(double,double,double,FourVector<double>&,FourVector<double>&);
  int point_init_conds(double,double,double[],FourVector<double>&,FourVector<double>&);

  // File Input/Output functions
  void output_map(std::string);
  void output_map(std::ostream&);

  // Size of pmap
  inline int xi_size(){return _N_xi;};
  inline int eta_size(){return _N_eta;};
  
  // Set
  
 protected:
  Metric &_g; // local metric
  Ray &_ray; // Ray generator.  Contains all we need to know!
  int _verbosity; // how much stuff to print out (0=nothing!)


  double _R, _THETA;
  double _frequency,_frequency0; // Frequency of PolarizationMap
  double _xi_lo, _xi_hi; // horizontal range
  double _eta_lo, _eta_hi; // vertical range
  int _N_xi, _N_eta; // number of steps
  std::valarray<double> _I, _Q, _U, _V; // Matrix with I,Q,U,V
  std::valarray<double> _tau, _D; // Diagnostics
  std::valarray<double> _IQUV_int; // Integrated IQUV over

  std::valarray<double> _xibg, _etabg, _Ibg; // Background xi,eta,I to subtract off during refine calculation

  std::ostream *_pstream; // Output stream for progress counters
  std::ofstream _lpstream; // Local output stream for setting stream by file name

  double _BHM, _BHD;

  bool _rI; // Whether or not to rescale the intensity in accordance with distance.

#ifdef VRT2_USE_MPI_MAP
  MPI_Comm _pmap_communicator;
  bool _created_pmap_communicator;
#endif



  // MPI stuff
  int _rank, _size;
#ifdef VRT2_USE_MPI_MAP
  void collect(int rank=0); // Collect into single array at end
#endif
};
};
#endif

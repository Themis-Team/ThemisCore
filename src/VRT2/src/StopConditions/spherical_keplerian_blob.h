/**********************************************/
/****** Header file for spherical_keplerian_blob.cpp ******/
/* Notes:
   Gives conditions to stop evolution at a spherical blob moving
in a Keplerian orbit with polarization parallel to the aximuthal
axis.
   Requires Metric metric to be defined as a global and initialized.
*/
/**********************************************/

// Only include once
#ifndef VRT2_SPHERICAL_KEPLERIAN_BLOB_H
#define VRT2_SPHERICAL_KEPLERIAN_BLOB_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>

// Local Headers
#include "vrt2_constants.h"
#include "stopcondition.h"
#include "fourvector.h"
#include "vrt2_globs.h"

namespace VRT2 {
class SphericalKeplerianBlob : public StopCondition {
 public:
  SphericalKeplerianBlob(Metric &g, double router, double rinner, double Rorb, double Rblob, double phi0=0.0);
  virtual ~SphericalKeplerianBlob() {};
  
  void set_phi0(double phi0) { _phi0 = phi0; };

  // The condition (1 stop, 0 don't stop)
  virtual int stop_condition(double[],double[]);

  // Limiting intensities (from optically thick blob)
  virtual double I(double[],int);
  virtual std::valarray<double> IQUV(double[]);
  
 private:
  double get_theta(double);
  
  const double _Rorb;  // Blob orbital radius
  const double _Rblob; // Size of blob (in blob proper frame)
  
  double _phi0;
  FourVector<double> _xblob, _ublob;
  double _E, _Omega;

  // blob velocity
  void blob_position(double);
  void blob_surface_velocity(FourVector<double>&, FourVector<double>&);

  // Whether or not we are intersecting the blob
  bool intersecting_blob(double []);
  
};
};
#endif

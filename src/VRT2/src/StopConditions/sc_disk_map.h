/**********************************************************/
/*** Returns the positions of Equatorial Disk crossings ***/
/*                                                        */
/* Returns the first above-to-below crossing in first two */
/* elements of IQUV array, and first below-to-above in    */
/* second two.                                            */
/*                                                        */
/**********************************************************/

// Only include once
#ifndef VRT2_SC_DISKMAP_H
#define VRT2_SC_DISKMAP_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>

// Local Headers
#include "stopcondition.h"
#include "metric.h"
#include "fourvector.h"

namespace VRT2 {
class SC_DiskMap : public StopCondition
{
 public:
  SC_DiskMap(Metric& g, double router, double rinner);
  virtual ~SC_DiskMap() {};

  // These take y[] and dydx[] as arguments

  // The condition (1 stop, 0 don't stop)
  virtual int stop_condition(double[],double[]);

  void reset() {_nswitches=-1;}; // Reset command

  // The Stokes' Parameters when stopped
  virtual std::valarray<double> IQUV(double[]);

 protected:
  double _xold[3], _xnew[3];
  int _nswitches; // 0 -> looking, 1 -> found one, 2 -> found second or new ray

  double _sa[2], _sb[2]; // r,phi for from above and below equatorial plane crossings

  void check_switch(double y[]);
};

};
#endif



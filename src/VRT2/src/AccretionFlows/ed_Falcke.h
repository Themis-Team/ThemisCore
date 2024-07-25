/*********************************************************/
/*** Generates jet model density distribution          ***/
/*  (Falcke et al.)                                      */
/*                                                       */
/*********************************************************/

#ifndef VRT2_ED_Falcke_H
#define VRT2_ED_Falcke_H

#include <algorithm>
#include <cmath>
#include <math.h>
using namespace std;

#include "electron_density.h"
#include "Falcke_jet_model.h"

namespace VRT2 {
class ED_Falcke : public ElectronDensity
{
 public:
  ED_Falcke(double denstiy_scale, FalckeJetModel& jet);
  virtual ~ED_Falcke() {};

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  double _density_scale; // Overall density scale

  FalckeJetModel& _jet;
};

inline double ED_Falcke::get_density(double,double r,double theta,double)
{
  double z = std::fabs( r*std::cos(theta) );
  double rho = std::fabs( r*std::sin(theta) ) + 1.0e-10; // Tiny deals with z-axis

  double dtmp;
  if (rho<_jet.rperp(z)) // If inside jet
  {
    r = std::max(r,_jet.nozzle_height());
    z = std::max(std::fabs(z),_jet.nozzle_height());
    dtmp = _density_scale * std::pow(_jet.nozzle_height()/r,2.0) / _jet.mach(z);
  }
  else
    dtmp = 0.0;

  //std::cerr << "density = " << dtmp << "\n\n";

  return dtmp;
}
};
#endif

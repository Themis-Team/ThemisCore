/*************************************************************/
/*** Generates temperature for Falcke jet model            ***/
/*                                                           */
/*************************************************************/

#ifndef VRT2_T_FALCKE_H
#define VRT2_T_FALCKE_H

#include "temperature.h"
#include "Falcke_jet_model.h"

namespace VRT2 {
class T_Falcke : public Temperature
{
 public:
  T_Falcke(double temperature_scale, FalckeJetModel& jet);
  virtual ~T_Falcke() {};

  virtual double get_temperature(double t, double r, double theta, double phi);

 private:
  double _temperature_scale;
  FalckeJetModel& _jet;
  //double _h;
};

inline double T_Falcke::get_temperature(double,double r,double theta,double)
{
  double z = std::fabs( r*std::cos(theta) );
  double rho = std::fabs( r*std::sin(theta) );

  double Ttmp;
  if (rho<_jet.rperp(z)) // If inside jet
    Ttmp =  _temperature_scale * std::pow(_jet.mach(z),-1.0/3.0);
  else
    Ttmp = 0.0;

  /*
  std::cout << "T:"
	    << std::setw(15) << r
	    << std::setw(15) << theta
	    << std::setw(15) << Ttmp
	    << std::setw(15) << _jet.mach(z)
	    << '\n' << std::endl;
  */
  /*
  std::cout << "r = " << r
	    << "  theta = " << theta;
  std::cout << "  T = " << Ttmp << "\n\n";
  */

  return Ttmp;
}
};
#endif

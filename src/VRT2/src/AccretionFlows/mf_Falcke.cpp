#include "mf_Falcke.h"

namespace VRT2 {

MF_Falcke::MF_Falcke(Metric& g, double B0, FalckeJetModel& jet, AccretionFlowVelocity& afv)
  : MagneticField(g), _B0(B0), _jet(jet), _afv(afv)
{
}

FourVector<double>& MF_Falcke::get_field_fourvector(double t, double r, double theta, double phi)
{
  // Set sign for split monopole
  double z = r*std::cos(theta);
  double sgn = (z<0 ? -1 : 1);
  z = std::fabs(z);

  // Make Br=0 for now.  However, in future make ~ 1/r^2
  double Br = 0;
  double Bth = 0;

  // Set Bphi=B0/r M^(-1/6) / (r*sin(theta))  where the last term makes it B^phi instead of B^\hat{\phi} !!!
  double Bph = sgn * _B0 * (_jet.nozzle_height()/r) * std::pow( _jet.mach(z), -1.0/6.0 ) / (r*std::sin(theta));

  // Get Bt so that B.u = 0
  FourVector<double> utmp = _afv(t,r,theta,phi);

  double Bt = - (Br*utmp.cov(1) + Bth*utmp.cov(2) + Bph*utmp.cov(3))/utmp.cov(0);

  // Set _b
  _b.mkcon(Bt,Br,Bth,Bph);

  /*
  std::cout << "B:" << std::setw(15) << r
	    << std::setw(15) << theta
	    << std::setw(15) << _jet.mach(z)
	    << std::setw(15) << Bph
	    << std::setw(15) << Bt
	    << std::setw(15) << utmp.cov(0)
	    << std::setw(15) << utmp.cov(1)
	    << std::setw(15) << (_b*_b) << "\n\n";
  */
  //std::cout << "B.u = " << (utmp*_b) << "\n\n";

  return _b;
}

};

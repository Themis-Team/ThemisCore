#include "mf_toroidal_beta.h"

namespace VRT2 {
MF_ToroidalBeta::MF_ToroidalBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double beta)
  : MagneticField(g), _u(u), _ne(ne), _beta(beta)
{
}

FourVector<double>& MF_ToroidalBeta::get_field_fourvector(double t, double r, double theta, double phi)
{
  // Get proper magnitude (=b^2)
  double P = _ne(t,r,theta,phi) * (_g.mass()/r) * 1.5e-3 / 6.0; // m_p c^2 = 1.5e-3 erg, 6 from virial thm
  double Bmag = std::sqrt( 8.0*VRT2_Constants::pi * P / _beta );
  double tmpval;

  Bmag = std::max(Bmag,1e-10); // in G, 10^-20 is really small

  //Bmag = std::sqrt( (_ne(t,r,theta,phi) * _g.mass()/r) / _beta );

  // Get direction orthogonal to flow velocity along phi direction
  FourVector<double> v(_g);
  FourVector<double> utmp = _u(t,r,theta,phi);
  _b.mkcon(0.0,0.0,0.0,1.0);
  double tmp = utmp*_b;
  _b = _b + (utmp*_b)*utmp;
  v = _b;
  tmpval = _b*_b;
  _b *= Bmag/std::sqrt(tmpval);

  if (vrt2_isnan((tmpval)))
    std::cout << "Nanned in MF_ToroidalBeta:"
	      << std::setw(15) << t
	      << std::setw(15) << r
	      << std::setw(15) << theta
	      << std::setw(15) << phi
	      << std::setw(15) << (_b*_b)
	      << std::setw(15) << Bmag
	      << std::setw(15) << (utmp*utmp)
	      << std::setw(15) << _g.g(3,3)
	      << std::setw(15) << (v*v)
	      << std::setw(15) << tmp
	      << std::setw(15) << utmp.cov(0)
	      << std::setw(15) << utmp.cov(1)
	      << std::setw(15) << utmp.cov(2)
	      << std::setw(15) << utmp.cov(3)
	      << std::setw(15) << _g.detg()
	      << std::endl;

  return _b;
}
};

// Include Statements
#include "rt_pwpa.h"

namespace VRT2 {
RT_PW_PA::RT_PW_PA(Metric& g, double Theta)
  : RadiativeTransfer(g), _Theta(Theta)
{
}

RT_PW_PA::RT_PW_PA(const double y[], Metric& g, double Theta)
  : RadiativeTransfer(g), _Theta(Theta)
{
  reinitialize(y);
}

RT_PW_PA::RT_PW_PA(FourVector<double>& x, FourVector<double>& k, Metric& g, double Theta)
  : RadiativeTransfer(g), _Theta(Theta)
{
  reinitialize(x,k);
}

// Do Geometric rotation
void RT_PW_PA::IQUV_rotate(double iquv[], double lambdai, const double yi[], const double dydxi[], double lambdaf, const double yf[], const double dydxf[])
{
  double qtmp = iquv[1];
  double utmp = iquv[2];

  double dpa = delta_tPA(yi,yf);
  double cdp = std::cos(dpa);
  double sdp = std::sin(dpa);

  /*
  std::cout << std::setw(15) << yi[1]
	    << std::setw(15) << yf[1]
      	    << std::setw(15) << dpa
	    << std::setw(15) << PW_polarization_angle(yi)
	    << std::setw(15) << PW_polarization_angle(yf)
    	    << std::endl;
  */

  iquv[1] = cdp*qtmp + sdp*utmp;
  iquv[2] = -sdp*qtmp + cdp*utmp;

  //iquv[1] = iquv[1];
  //iquv[2] += dpa*0.180/M_PI * iquv[0];

  //iquv[2] = std::fmod( iquv[2] + 0.360 * iquv[0], 0.360 * iquv[0] );

  _g.reset(yf);
  reinitialize(yf);
}

// Get polarization angle at this position for the vertical direction via Penrose-Walker constant
// RESETS METRIC AND RT -> MUST RESET BACK WHEN DONE
double RT_PW_PA::PW_polarization_angle(const double y[])
{
  _g.reset(y);
  reinitialize(y);

  // Useful common functions
  double st = std::sin(_x.con(2));
  double ct = std::cos(_x.con(2));
  double aM = _g.ang_mom()/_g.mass();
  double rM = _x.con(1)/_g.mass();

  // Choose a fidicial direction (z-axis)
  FourVector<double> t(_g), f(_g);
  t.mkcov(1.0,0.0,0.0,0.0); // Choose ZAMO frame as the local time frame
  f.mkcon(0.0,ct,-st/_x.con(1),0.0);
  f = cross_product(t,f,_k);

  // Get Penrose-Walker constant associated with this fiducial direction
  std::complex<double> i(0.0,1.0);
  std::complex<double> Kpw = ( (_k.con(0)*f.con(1)-_k.con(1)*f.con(0))
			       + aM*st*st*(_k.con(1)*f.con(3)-_k.con(3)*f.con(1))
			       -
			       i*st*( (rM*rM+aM*aM)*(_k.con(3)*f.con(2)-_k.con(2)*f.con(3))
					- aM*(_k.con(0)*f.con(2)-_k.con(2)*f.con(0)))
			       ) * (rM - i*aM*ct);

  double E = _k.cov(0);
  double L = _k.cov(3);
  double Q = _k.cov(2)*_k.cov(2) + ct*ct*( -aM*aM*E*E + L*L/(st*st) );
    
  L /= E;
  Q /= E*E;
  
  
  double sT = std::sin(_Theta), cT = std::cos(_Theta);
  double S = L/sT - aM*sT;
  double T = std::max(0.0, Q - L*L*cT*cT/(sT*sT) + aM*aM*cT*cT ); // DEALS WITH CATASTROPHIC SUBSTRACTION
  // First chooses root, second accounts for k_theta and wrap around (i.e., k_theta is in the wrong direction if
  //  theta > pi!
  //T = (_k.con(2)>0 ? 1.0 : -1.0) * ( std::fmod(std::fmod(_x.con(2),2.0*M_PI)+2.0*M_PI,2.0*M_PI) > M_PI ? -1 : 1) * std::sqrt( T );
  //T = (_k.con(2)>0 ? 1.0 : -1.0) * std::sqrt( T );
  T = std::sqrt( T );

  return atan2( -S*Kpw.real()+T*Kpw.imag(), -S*Kpw.imag()-T*Kpw.real() );
}

double RT_PW_PA::delta_tPA(const double yi[], const double yf[])
{
  return 2.0*( PW_polarization_angle(yf) - PW_polarization_angle(yi) ); // factor of 2 comes from Stoke's vs. polarization angle.
}

// Characteristic local length to affine parameter difference
double RT_PW_PA::dlambda(const double y[], const double dydx[])
{
  return 1.0;
}
};

#include "pwpa_tester.h"
#include <complex>

namespace VRT2 {
PWPATester::PWPATester(Metric &g, double router, double rinner, double Rorb, double Rblob, double phi0)
  : StopCondition(g,router,rinner),  _Rorb(Rorb), _Rblob(Rblob), _phi0(phi0), _xblob(g,0.0), _ublob(g,0.0)
{
  // Set xblob initially
  _xblob.mkcon(0.0,_Rorb,0.5*M_PI,_phi0);

  // Get ublob and _E
  g.reset(_xblob.con());
  double gppr = _g.Dginv(3,3,1);
  double gtpr = _g.Dginv(0,3,1);
  double L = ( std::sqrt( gtpr*gtpr - _g.Dginv(0,0,1)*gppr ) - gtpr )/gppr;
  double gtp = _g.ginv(0,3);
  double gtt = _g.ginv(0,0);

  _E = -( gtt + 2.0*gtp*L + _g.ginv(3,3)*L*L );
  _E = 1.0/std::sqrt(_E);

  if (vrt2_isnan(_E))
    std::cerr << "Perhaps you meant prograde?\n";

  _Omega = ( _g.ginv(3,3)*L + gtp )/( gtt + gtp*L );

  _Omega = 0.0;

  double ut = -_E*( gtt + gtp*L );  // remember E=-u_t, L=u_phi/u_t
  double uphi = ut*_Omega;

  _ublob.mkcon(ut,0.0,0.0,uphi);
}

/*** Condition at which propagation stops regardless of adiabaticity ***/
int PWPATester::stop_condition(double y[], double dydx[])
{
  if ( (y[1] > _rout) && (dydx[1] > 0) )
    return 1; // Stop
  else if (y[1] < _rin*_g.horizon())
    return 1;
  else if (intersecting_blob(y))
    return 1; // Hitting blob, so stop
  else
    return 0; // Don't Stop
}

bool PWPATester::intersecting_blob(double y[])
{
  // Set metric to blob center position
  blob_position(y[0]);
  _g.reset(_xblob.con());
  
  // Determine differential radius from blob center
  FourVector<double> x(_g);
  //   Put theta and phi into correct range.
  double theta = std::fmod(y[2]+M_PI,2.0*M_PI)-M_PI;
  double phi = y[3] + (theta<0 ? M_PI : 0.0);
  theta = std::fabs(theta);
  x.mkcon(y[0],y[1],theta,phi);
  x -= _xblob;
  std::valarray<double> xtmp = x.con();
  //   Put everything in right range (i.e. 0 and 2pi are close)
  //    (Note that theta can take on negative values, and thus must fold this correctly into phi)
  xtmp[2] = std::fabs(std::fmod(xtmp[2]+M_PI,2.0*M_PI)-M_PI);
  xtmp[3] = std::fmod(std::fmod(xtmp[3],2.0*M_PI)+3.0*M_PI,2.0*M_PI)-M_PI;
  x.mkcon(xtmp);
  double r2 = x*x;
  double rdu = x*_ublob;
  double r = std::sqrt( r2 + rdu*rdu );

  // Reset metric to ray position
  _g.reset(y);

  // Check for blob intersection
  if (r<_Rblob)
    return true;
  return false;
}

/*** Intensity at stopped place (for optically thick stuff) ***/
double PWPATester::I(double y[],int)
{
  if (intersecting_blob(y)) {

    return 1.0;

    // (1) Get the temperature of the blob (virial temperature at blob CENTER)
    blob_position(y[0]);

    _g.reset(_xblob.con());
    double kT = (_E-1.0/std::sqrt(-_g.ginv(0,0)));
    kT *= VRT2_Constants::me * VRT2_Constants::c*VRT2_Constants::c; // Put kT in units of ergs


    // (2) Get the red-shifted frequency at this point (keep angular velocity const.)    
    _g.reset(y);

    FourVector<double> u_surf(_g);
    blob_surface_velocity(_xblob,u_surf);


    FourVector<double> k(_g);
    k.mkcov(y+4);
    double omega = std::fabs(k*u_surf);
    // Put omega in Hz units
    //extern double wavelength;
    //omega *= 2.0*VRT2_Constants::pi*VRT2_Constants::c/(wavelength*VRT2_Constants::M_SgrA);
    omega *= 2.0*VRT2_Constants::pi*VRT2_Constants::c/(VRT2_Constants::M_SgrA_cm);


    //  (3) Generate blackbody intensity and return (remember that this is N_I \proto I/omega^3 !)
    double kT_o_hw = (kT/(VRT2_Constants::hbar*omega)); // Get kt/hw where omega is in units of 1 keV/hbar
    double hbarc = VRT2_Constants::hbar*VRT2_Constants::c; // Get black body intensity

    return 2.0*hbarc * kT_o_hw / (4.0*VRT2_Constants::pi*VRT2_Constants::pi*std::pow(VRT2_Constants::c,3.0));
  }

  return 0.0;
}

/*** Stokes' Parameters at stopped place ***/
std::valarray<double> PWPATester::IQUV(double y[])
{
  std::valarray<double> iquv(0.0,5);

  // Total intensity
  iquv[0] = I(y,0);

  if (intersecting_blob(y)) {
    


    iquv[1] = 0.5*iquv[0];
    iquv[2] = iquv[3] = 0.0;


    _g.reset(y);
    FourVector<double> _x(_g), _k(_g);
    _x.mkcon(y);
    _k.mkcov(y+4);
    
    double _Theta = 0.5*M_PI;

    // Useful common functions
    double ct = std::cos(_x.con(2));
    double st = std::sin(_x.con(2));
    double a = _g.ang_mom();
    
    // Choose a fidicial direction (z-axis)
    FourVector<double> t(_g), f(_g);
    t.mkcov(1.0,0.0,0.0,0.0);
    f.mkcon(0.0,ct,-st/_x.con(1),0.0);
    //f.mkcon(0.0,1.0,0.0,0.0);
    f = cross_product(t,f,_k);
    f = cross_product(t,f,_k);
    

    // Get Penrose-Walker constant associated with this fiducial direction
    std::complex<double> i(0.0,1.0);
    std::complex<double> Kpw = ( (_k.con(0)*f.con(1)-_k.con(1)*f.con(0))
				 + a*st*st*(_k.con(1)*f.con(3)-_k.con(3)*f.con(1))
				 
				 - i*st*( (_x.con(1)*_x.con(1)+a*a)*(_k.con(3)*f.con(2)-_k.con(2)*f.con(3))
					  - a*(_k.con(0)*f.con(2)-_k.con(2)*f.con(0)))
				 ) * (_x.con(1) - i*a*ct);
    
    double E = _k.cov(0);
    double L = _k.cov(3);
    double Q = _k.cov(2)*_k.cov(2) + ct*ct*( -a*a*E*E + L*L/(st*st) );
    
    L /= E;
    Q /= E*E;
    

    double sT = std::sin(_Theta), cT = std::cos(_Theta);
    double S = L/sT - a*sT;
    double T = Q - L*L*cT*cT/(sT*sT) + a*a*cT*cT;
    // First chooses root, second accounts for k_theta and rap around (i.e., k_theta is in the wrong direction if
    //  theta > pi!
    //T = (_k.con(2)>0 ? 1.0 : -1.0) * ( std::fmod(std::fmod(_x.con(2),2.0*M_PI)+2.0*M_PI,2.0*M_PI) > M_PI ? -1 : 1) * std::sqrt( T );
    T = std::sqrt( T );

    iquv[3] = std::fmod(atan2( -S*Kpw.real()+T*Kpw.imag(), -S*Kpw.imag()-T*Kpw.real() ) + 2.0*M_PI, 2.0*M_PI) * (0.180/M_PI) * iquv[0];
  }
  else
    iquv[1] = iquv[2] = iquv[3] = 0.0;

  return iquv;
}

// Assumes that the metric has been reset
// Note on conventions:
//  (1) t goes backwards, as expected, so u should be forwards (i.e., u^t>0 -> u_t<0)
//  (2) E = -u_t, period.
//  (3) L = u_phi/u_t, period.
//  (4) Omega = u^phi/u^t, period.
void PWPATester::blob_position(double t)
{
  _xblob.mkcon(t,_Rorb,0.5*M_PI,_phi0 + t*_Omega);
}
void PWPATester::blob_surface_velocity(FourVector<double>& x, FourVector<double>& u)
{
  u.mkcon(1.0,0.0,0.0,_Omega);
  u *= 1.0/sqrt(-(u*u));
}
};

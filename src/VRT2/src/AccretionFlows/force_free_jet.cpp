#include "force_free_jet.h"

namespace VRT2 {

ForceFreeJet::ForceFreeJet(Metric& g, double p, double r_inner_edge, double r_foot_print, double r_load, double B0, double n0, double gamma_max)
: _g(g), _p(p), _ri(r_inner_edge), _rj(r_foot_print), _rl(r_load), _B0(B0), _n0(n0), _gamma_max(gamma_max), _bF(g), _uF(g), _b(g), _u(g), _afv(g,*this), _ed(*this), _mf(g,*this)
{
  _psii = psi(_ri,0.5*M_PI);
  _psij = psi(_rj,0.5*M_PI);
  
  _dtheta = _rj;



#ifdef CONSERVATIVE_MASS_TRANSPORT
  //get_F(1000);
  get_F2(_rl,_rj);
#endif
}


MagneticField& ForceFreeJet::mf()
{
  return _mf;
}
AccretionFlowVelocity& ForceFreeJet::afv()
{
  return _afv;
}
ElectronDensity& ForceFreeJet::ed()
{
  return _ed;
}



inline double ForceFreeJet::psi(double r, double theta) const
{
  return ( std::pow(r,_p)*(1.0-std::cos(theta)) );
}
inline double ForceFreeJet::dpsi_dr(double r, double theta) const
{
  return ( _p*std::pow(r,_p-1)*(1.0-std::cos(theta)) );
}
inline double ForceFreeJet::dpsi_dth(double r, double theta) const
{
  return ( std::pow(r,_p)*std::sin(theta) );
}

inline double ForceFreeJet::OmegaZAMO(double r, double theta) const
{
  double a = _g.ang_mom()/_g.mass();
  double a2 = a*a;
  double r2a2 = r*r + a2;
  double delta = r*r - 2.0*_g.mass()*r + a2;
  double sn = std::sin(theta);

  return ( 2.0*_g.mass()*r*a/( r2a2*r2a2 - a2*delta*sn*sn ) );
  //return ();
}

inline double ForceFreeJet::Omega(double psi) const
{
  if (psi<_psii)
    psi = _psii;

  // NEED TO THINK ABOUT THIS, SINCE WE MIGHT WANT TO CHOOSE THE STUFF INSIDE
  // OF THE ISCO TO ORBIT ON FIXED-L ORBITS INSTEAD.
  // Omega = 1/[r^(3/2) + a] until r=r_i
  return (  1.0/( std::pow(psi,1.5/_p) + _g.ang_mom()/_g.mass() )  );
  //return (0.0);
}


inline double ForceFreeJet::F(double psi) const
{
#ifdef CONSERVATIVE_MASS_TRANSPORT

  if (_psv.size())
  {
  
    double lps = std::log(psi);
    int ips = int( (lps-_psv[0])/(_psv[_psv.size()-1]-_psv[0]) * _psv.size() );
    
    if (ips<0)
      return _Fv[0];
    else if (ips>int(_psv.size())-2)
      return 0.0;
    else
      return (  ( _Fv[ips+1]*(lps-_psv[ips]) + _Fv[ips]*(_psv[ips+1]-lps) )/(_psv[ips+1]-_psv[ips])  );
  }
  else
    return 0;

#else

  return (  std::exp( -0.5*std::pow(psi/(_psij),2.0/_p) ) );

#endif
}



void ForceFreeJet::get_F(double r)
{
  int Nth = 1000;
  double theta, ps, F;
  std::vector<double> psv, Fv;
  for (int ith=0; ith<Nth; ++ith)
  {
    theta = 1e-6 + (ith)*0.5*M_PI/double(Nth-1);
    ps = psi(r,theta);

    _g.reset(0,r,theta,0);
    compute_all(r,theta);


    F = std::exp(-theta*theta/(2.0*_dtheta*_dtheta)) * _gamma*( _g.ginv(0,0) + _g.ginv(0,3)/Omega(ps) ) / ( _bF2 * _uF.con(0));


    psv.push_back(ps);
    Fv.push_back(F);
  }


  for (int i=0; i<Nth; ++i)
  {
    
    ps = std::exp( std::log(psv[0]) + i*std::log(psv[psv.size()-1]/psv[0])/double(Nth) );
    _psv.push_back( std::log(ps) );
    

    int j;
    for (j=0; j<int(psv.size()) && ps>psv[j]; ++j)
      {}

    _Fv.push_back(  (Fv[j]*(ps-psv[j-1]) + Fv[j-1]*(psv[j]-ps))/(psv[j]-psv[j-1]) );
  }
}

void ForceFreeJet::get_F2(double rh, double rj)
{
  int Nth = 3000;
  double Rmax = 100;
  double R,z;
  double r, theta;
  double ps, F;
  std::vector<double> psv, Fv;
  std::vector<double> Rv, zv;
  for (int ith=0; ith<Nth; ++ith)
  {
    z = rh;
    R = 1e-6 + ith*Rmax/double(Nth-1);
    theta = atan2(R,z);
    r = std::sqrt(z*z+R*R);

    /*
    if (ith<(Nth/2))
    {
      theta = 1e-10 + (ith)*0.5*M_PI/double(Nth/2);
      z = rh*std::cos(theta);
      R = rj*std::sin(theta);
      theta = atan2(R,z);
      r = std::sqrt(z*z+R*R);
    }
    else
    {
      theta = 0.5*M_PI;
      R = r = rj + (ith-Nth/2)*Rmax/double(Nth/2);
    }
    */

    ps = psi(r,theta);

    _g.reset(0,r,theta,0);
    compute_all(r,theta);


    //F = std::exp(-theta*theta/(2.0*_dtheta*_dtheta)) * _gamma*( _g.ginv(0,0) + _g.ginv(0,3)/Omega(ps) ) / ( _bF2 * _uF.con(0));
    F = std::exp(-R*R/(2.0*rj*rj)) * _gamma*( _g.ginv(0,0) + _g.ginv(0,3)/Omega(ps) ) / ( _bF2 * _uF.con(0));
    

    psv.push_back(ps);
    Fv.push_back(F);

    zv.push_back(z);
    Rv.push_back(R);
  }



  for (int i=0; i<Nth; ++i)
  {
    
    ps = std::exp( std::log(psv[0]) + i*std::log(psv[psv.size()-1]/psv[0])/double(Nth) );
    _psv.push_back( std::log(ps) );
    

    if (ps<psv[0])
      _Fv.push_back(Fv[0]);
    else
    {
      int j;
      for (j=1; j<int(psv.size()) && ps>psv[j]; ++j)
	{}
      _Fv.push_back(  (Fv[j]*(ps-psv[j-1]) + Fv[j-1]*(psv[j]-ps))/(psv[j]-psv[j-1]) );
    }
  }

  /*
  for (size_t i=0; i<Nth; ++i)
    std::cout << std::setw(15) << Rv[i]
	      << std::setw(15) << zv[i]
	      << std::setw(15) << psv[i]
	      << std::setw(15) << Fv[i]
	      << std::setw(15) << _psv[i]
	      << std::setw(15) << _Fv[i]
	      << '\n';
  */


}





void ForceFreeJet::compute_ubn(double r, double theta)
{
  double rz = std::cos(theta);
  double abth = std::acos( std::fabs(rz) );


  std::valarray<double> x(4);
  bool floored_theta = false;
  if (std::fabs(abth) < 1e-6 ) // Theta is getting too small and bad things can happen along the axis
  {
    floored_theta = true;
    x = _g.local_position();
    abth = (abth>0 ? 1 : -1)*1e-6;
    _g.reset(x[0],r,abth,x[3]);
  }

  double ps = psi(r,abth);
  double Om = Omega(ps);
  double OmZAMO = OmegaZAMO(r,abth);

  Om = std::max(Om,OmZAMO);

  // Get the Field velocity
  _uF.mkcon(1.0,0.0,0.0,Om);
  _uF2 = (_uF*_uF);
  _uF *= 1.0/std::sqrt(std::fabs(_uF2));
  _uF2 = ( _uF2>0 ? 1 : -1);

  //std::cerr << "uF fine" << std::endl;

  // Get the Field-velocity magnetic field
  //  ASSUMES BOYER-LINDQUIST STRUCTURE
  double br = -_uF2*_g.g(1,1) * dpsi_dth(r,abth) / (_uF.con(0)*_g.detg());

  //std::cerr << "bFr fine" << std::endl;

  double bth = (rz>0 ? 1.0 : -1.0)*_uF2*_g.g(2,2) * dpsi_dr(r,abth) / (_uF.con(0)*_g.detg());

  //std::cerr << "bFth fine" << std::endl;

  double bph = - 2.0*Om*ps * _uF.con(0);

  bph = std::min(std::fabs(bph),std::fabs(1000.0*_B0/r))*(bph >= 0 ? 1 : -1);

  //std::cerr << "bFph fine" << std::endl;


  if (_gamma_max>1) {
    double beta_max = std::sqrt(1-1.0/(_gamma_max*_gamma_max));
    bph /= beta_max;
  }
  


  double bt = -bph*Om;
  _bF.mkcov(bt,br,bth,bph);
  _bF *= -(rz>0 ? 1.0 : -1.0);

  // Get the jet-plasma velocity
  _bF2 = _bF*_bF;
  double beta = _bF.con(0)*_uF2/(_uF.con(0)*_bF2);
  _beta = beta;
  _gamma = _uF2/std::sqrt( std::max(1e-10,-(_uF2 + beta*beta*_bF2)));
  //_gamma = _uF2/std::sqrt( std::fabs(_uF2 + beta*beta*_bF2));

  //std::cerr << "beta and gamma fine" << std::endl;


  _u = -_gamma*(_uF + beta*_bF);
  _b = _gamma*_B0*(_bF - _uF2*beta*_bF2*_uF);

  //std::cerr << "u and b fine" << std::endl;
  //std::cerr << "uF" << _uF << std::endl;
  //std::cerr << "bF" << _bF << std::endl;
  //std::cerr << "u" << _u << std::endl;
  //std::cerr << "b" << _b << std::endl;


#ifdef CONSERVATIVE_MASS_TRANSPORT
  _n = _n0 *_uF.con(0)/(_gamma*( _g.ginv(0,0) + _g.ginv(0,3)/Om )) * _bF2 * F(ps) * (1 - std::exp(-0.5*r*r/(_rl*_rl)) );
#else
  //_n = _n0 * F(ps) * std::pow(r+_rl,-2.0) * (1.0 - std::exp(-0.5*r*r/(_rl*_rl)) );
  _n = _n0 * F(ps) *  std::pow(r+_rl,-2.0) * (1.0 - std::exp(-0.5*r*r/(_rl*_rl)) );
#endif

  //std::cerr << "n fine" << std::endl;

  //std::cerr << "u" << _u << std::endl;


  //if ( std::isnan((_u*_u)) || ( r<1.11439 && std::fabs(theta-1.56761)<1e-3 ) ) {
  if ( std::isnan((_u*_u)) ) {
    std::cout << "Four Velocity is NaN'd:"
	      << "  (r,theta):"
	      << std::setw(15) << r
	      << std::setw(15) << theta
	      << std::setw(15) << (_u*_u)
	      << " | "
	      << std::setw(15) << _g.ginv(0,0)
	      << std::setw(15) << _g.ginv(0,3)
	      << std::setw(15) << _g.ginv(1,1)
	      << std::setw(15) << _g.ginv(2,2)
	      << std::setw(15) << _g.ginv(3,3)
	      << " | "
	      << std::setw(15) << _g.g(0,0)
	      << std::setw(15) << _g.g(0,3)
	      << std::setw(15) << _g.g(1,1)
	      << std::setw(15) << _g.g(2,2)
	      << std::setw(15) << _g.g(3,3)
	      << "  u^a:"
	      << std::setw(15) << _u.con(0)
	      << std::setw(15) << _u.con(1)
	      << std::setw(15) << _u.con(2)
	      << std::setw(15) << _u.con(3)  
	      << "  u_a:"
	      << std::setw(15) << _u.cov(0)
	      << std::setw(15) << _u.cov(1)
	      << std::setw(15) << _u.cov(2)
	      << std::setw(15) << _u.cov(3) 
	      << "  uF^a:"
	      << std::setw(15) << _uF.con(0)
	      << std::setw(15) << _uF.con(1)
	      << std::setw(15) << _uF.con(2)
	      << std::setw(15) << _uF.con(3) 
	      << "  bF^a:"
	      << std::setw(15) << _bF.con(0)
	      << std::setw(15) << _bF.con(1)
	      << std::setw(15) << _bF.con(2)
	      << std::setw(15) << _bF.con(3) 
	      << "  Omega " 
	      << std::setw(15) << Om
	      << std::endl;
    //std::exit(2);
  }

  //std::cerr << "u" << _u << std::endl;
  

  if (floored_theta ) // Theta is getting too small and bad things can happen along the axis
    _g.reset(x[0],x[1],x[2],x[3]);
  
  //std::cerr << "u" << _u << std::endl;

}

      //std::cerr << "uh" << uh << std::endl;
      //std::cerr << "u" << _u << std::endl;

bool ForceFreeJet::past_light_cylinder_proximity_limit()
{
  return ( bool( std::fabs(_gamma)>1e4 || std::isnan(_gamma) || (1+(_u*_u)) > 1e-10 ) );
}


#define DR (1e-1)
//#define DTHETA (1e-2)
void ForceFreeJet::compute_all(double r, double theta)
{
  compute_ubn(r,theta);

  //if (std::fabs(_gamma)>1e4 || std::isnan(_gamma) || (1+(_u*_u)) > 1e-10) // We are too close to the light cylinder
  if ( past_light_cylinder_proximity_limit() )
  {
    std::valarray<double> x = _g.local_position();
    double thtmp = x[2];
    double rl, rh, r0;
    rl = rh = r0 =x[1];
    /*
    std::cerr << "Proximity Alert! Light Cylinder Ahead" 
	      << std::setw(15) << r*std::sin(theta)
	      << std::setw(15) << r*std::cos(theta)
	      << std::setw(15) << _gamma
	      << std::setw(15) << 1+_u*_u
	      << std::endl;
    */
    do {


      // Get lower limit
      rl -= DR;
      _g.reset(x[0],rl,thtmp,x[3]);
      compute_ubn(rl,thtmp);
      //thl -= DTHETA;
      //_g.reset(x[0],rtmp,thl,x[3]);
      //compute_ubn(rtmp,thl);
      
      //std::cerr << "u" << _u << std::endl;

      FourVector<double> ul = _u;
      FourVector<double> bl = _b;
      double nl = _n;
      double gammal = _gamma;
      double bF2l = _bF2;

      rh += DR;
      _g.reset(x[0],rh,thtmp,x[3]);
      compute_ubn(rh,thtmp);
      //thh += DTHETA;
      //_g.reset(x[0],rtmp,thh,x[3]);
      //compute_ubn(rtmp,thh);

      //std::cerr << "u" << _u << std::endl;

      FourVector<double> uh = _u;

      //std::cerr << "uh" << uh << std::endl;
      //std::cerr << "u" << _u << std::endl;

      FourVector<double> bh = _b;

      double nh = _n;
      double gammah = _gamma;
      double bF2h = _bF2;

      //std::cerr << "uh" << uh << std::endl;
      /*std::cerr << "thl" << std::setw(15) << thl << std::endl;
      std::cerr << "thh" << std::setw(15) << thh << std::endl;
      std::cerr << "th0" << std::setw(15) << th0 << std::endl;
      std::cerr << "x[2]" <<std::setw(15) << x[2] << std::endl;
      std::cout << "thl" << std::setw(15) << thl
		<< "thh" << std::setw(15) << thh
		<< "th0" << std::setw(15) << th0 
		<< "x[2]" <<std::setw(15) << x[2]
		<< "\n ul" << ul 
		<< "\n uh" << uh << std::endl;*/

      _g.reset(x[0],x[1],x[2],x[3]);

      //std::cerr << "after reset x2" << std::setw(15) << x[2] << std::endl;
      //std::cerr << "uh" << uh << std::endl;

      double wh = (rh-r0)/(rh-rl);
      double wl = (r0-rl)/(rh-rl);

      //std::cerr << "wh" << std::setw(15) << wh << std::endl;
      //std::cerr << "wl" << std::setw(15) << wl << std::endl;
      //std::cerr << "uh+ul" << std::setw(15) << uh+ul << std::endl;

      _u = wh*uh + wl*ul;
      _b = wh*bh + wl*bl;
      _n = wh*nh + wl*nl;
      _gamma = wh*gammah + wl*gammal;
      _bF2 = wh*bF2h + wl*bF2l;

      /*std::cout << "thl" << std::setw(15) << thl
		<< "thh" << std::setw(15) << thh
		<< "th0" << std::setw(15) << th0 
		<< "x[2]" <<std::setw(15) << x[2]
		<< "\n ul" << ul 
		<< "\n uh" << uh 
		<< "ul^2" << std::setw(15) << (ul*ul) 
		<< "uh^2" << std::setw(15) << (uh*uh) << std::endl;

      std::cout << "_u" << _u 
      << "_u*_u" << std::setw(15)<< (_u*_u) << std::endl;*/

    } while ( (_u*_u) > 0 );

    _u = (1.0/std::sqrt(-(_u*_u))) * _u;


#if 0
    // Get lower limit
    do {
      thl -= DTHETA;
      _g.reset(x[0],rtmp,thl,x[3]);
      compute_ubn(rtmp,thl);
      /// } while (!past_light_cylinder_proximity_limit());
    } while (false);

    FourVector<double> ul = _u;
    FourVector<double> bl = _b;
    double nl = _n;
    double gammal = _gamma;
    double bF2l = _bF2;

    double ul2 = (ul*ul);

      //std::cerr << "uh" << uh << std::endl;
      //std::cerr << "u" << _u << std::endl;

    // Get upper limit
    do {
      thh += DTHETA;
      _g.reset(x[0],rtmp,thh,x[3]);
      compute_ubn(rtmp,thh);
      //} while (!past_light_cylinder_proximity_limit());
    } while (false);

    FourVector<double> uh = _u;
    FourVector<double> bh = _b;
    double nh = _n;
    double gammah = _gamma;
    double bF2h = _bF2;


    double uh2 = (uh*uh);

    /*    
    FourVector<double> utmp(_g), btmp(_g);
    double ntmp, gammatmp, bF2tmp;
    std::valarray<double> x = _g.local_position();
    double rtmp = x[1];
    double thtmp = x[2]-DTHETA;
    _g.reset(x[0],rtmp,thtmp,x[3]);
    compute_ubn(rtmp,thtmp);
    
    ul = _u;

    utmp = _u;
    btmp = _b;
    ntmp = _n;
    gammatmp = _gamma;
    bF2tmp = _bF2;

    thtmp = x[2]+DTHETA;
    _g.reset(x[0],rtmp,thtmp,x[3]);
    compute_ubn(rtmp,thtmp);


    _u = 0.5*(_u+utmp);
    _b = 0.5*(_b+btmp);
    _n = 0.5*(_n+ntmp);
    _gamma = 0.5*(_gamma+gammatmp);
    _bF2 = 0.5*(_bF2+bF2tmp);
    */

    _g.reset(x[0],x[1],x[2],x[3]);

    double wh = (thh-th0)/(thh-thl);
    double wl = (th0-thl)/(thh-thl);

    _u = wh*uh + wl*ul;
    _b = wh*bh + wl*bl;
    _n = wh*nh + wl*nl;
    _gamma = wh*gammah + wl*gammal;
    _bF2 = wh*bF2h + wl*bF2l;



    FourVector<double> ufoo=_u;

    _u = (1.0/std::sqrt(-(_u*_u))) * _u;

#endif

    /*
    std::cout << std::setw(15) << _gamma
	      << std::setw(15) << _u.con(0)
	      << std::setw(15) << (_u*_u)
	      << '\n';
    */

    /*
    if ( std::isnan((_u*_u)) ) 
    {
      std::cout << "Four Velocity is NaN'd in all:"
		<< "  (r,theta):"
		<< std::setw(15) << r
		<< std::setw(15) << theta
		<< std::setw(15) << (_u*_u)
		<< std::setw(15) << ul2
		<< std::setw(15) << (u0*u0)
		<< std::setw(15) << uh2
		<< std::setw(15) << (ufoo*ufoo)
		<< "  thetas:"
		<< std::setw(15) << thl
		<< std::setw(15) << th0
		<< std::setw(15) << thh
		<< "  weights:"
		<< std::setw(15) << wh
		<< std::setw(15) << wl
		<< "  gammas:"
		<< std::setw(15) << _gamma
		<< std::setw(15) << gammah
		<< std::setw(15) << gammal
		<< "  u^a:"
		<< std::setw(15) << _u.con(0)
		<< std::setw(15) << _u.con(1)
		<< std::setw(15) << _u.con(2)
		<< std::setw(15) << _u.con(3)  
		<< "  ufoo^a:"
		<< std::setw(15) << ufoo.con(0)
		<< std::setw(15) << ufoo.con(1)
		<< std::setw(15) << ufoo.con(2)
		<< std::setw(15) << ufoo.con(3)  
		<< "  uh^a:"
		<< std::setw(15) << uh.con(0)
		<< std::setw(15) << uh.con(1)
		<< std::setw(15) << uh.con(2)
		<< std::setw(15) << uh.con(3)  
		<< "  u0^a:"
		<< std::setw(15) << u0.con(0)
		<< std::setw(15) << u0.con(1)
		<< std::setw(15) << u0.con(2)
		<< std::setw(15) << u0.con(3)  
		<< "  ul^a:"
		<< std::setw(15) << ul.con(0)
		<< std::setw(15) << ul.con(1)
		<< std::setw(15) << ul.con(2)
		<< std::setw(15) << ul.con(3)  
		<< std::endl;
      //std::exit(2);
    }
    */

  }


}
#undef DR


FourVector<double>& ForceFreeJet::magnetic_field(double r, double theta)
{
  compute_all(r,theta);
  return _b;
}
FourVector<double>& ForceFreeJet::velocity(double r, double theta)
{
  compute_all(r,theta);

  if ( std::isnan((_u*_u)) ) {
    std::cerr << "Four Velocity is NaN'd in velocity:"
	      << "  (r,theta):"
	      << std::setw(15) << r
	      << std::setw(15) << theta
	      << "  u^a:"
	      << std::setw(15) << _u.con(0)
	      << std::setw(15) << _u.con(1)
	      << std::setw(15) << _u.con(2)
	      << std::setw(15) << _u.con(3)  
	      << "  u_a:"
	      << std::setw(15) << _u.cov(0)
	      << std::setw(15) << _u.cov(1)
	      << std::setw(15) << _u.cov(2)
	      << std::setw(15) << _u.cov(3) 
	      << "  uF^a:"
	      << std::setw(15) << _uF.con(0)
	      << std::setw(15) << _uF.con(1)
	      << std::setw(15) << _uF.con(2)
	      << std::setw(15) << _uF.con(3) 
	      << "  bF^a:"
	      << std::setw(15) << _bF.con(0)
	      << std::setw(15) << _bF.con(1)
	      << std::setw(15) << _bF.con(2)
	      << std::setw(15) << _bF.con(3) 
	      << std::endl;
    std::exit(2);
  }


  return _u;
}
double ForceFreeJet::density(double r, double theta)
{
  compute_all(r,theta);
  return std::max(_n,1e-20);
}


FourVector<double>& ForceFreeJet::MF_FFJ::get_field_fourvector(double, double r, double theta, double)
{
  return _ffj.magnetic_field(r,theta);
}
FourVector<double>& ForceFreeJet::AFV_FFJ::get_velocity(double, double r, double theta, double)
{
  return _ffj.velocity(r,theta);
}
double ForceFreeJet::ED_FFJ::get_density(double, double r, double theta, double)
{
  return _ffj.density(r,theta);
}

};

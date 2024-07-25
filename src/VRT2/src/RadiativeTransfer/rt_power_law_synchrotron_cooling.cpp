#include "rt_power_law_synchrotron_cooling.h"
#include "fast_math.h"

namespace VRT2 {
RT_PowerLawSynchrotronCooling::RT_PowerLawSynchrotronCooling(Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u, 
                 MagneticField& B,
					       double injection_time, double spectral_index, double gamma_min)
  : RadiativeTransfer(g), _ne(ne), _u(u), _B(B),
    _injection_time(injection_time), _spectral_index(spectral_index), 
    _gamma_min(gamma_min)
{
  set_constants();
}
RT_PowerLawSynchrotronCooling::RT_PowerLawSynchrotronCooling(const double y[], 
                 Metric& g, ElectronDensity& ne, 
                 AccretionFlowVelocity& u, MagneticField& B,
					       double injection_time, double spectral_index, double gamma_min)
  : RadiativeTransfer(y,g), _ne(ne), _u(u), _B(B),
    _injection_time(injection_time), _spectral_index(spectral_index), 
    _gamma_min(gamma_min)
{
  set_constants();
}
RT_PowerLawSynchrotronCooling::RT_PowerLawSynchrotronCooling(
                 FourVector<double>& x, FourVector<double>& k, 
                 Metric& g, ElectronDensity& ne, 
                 AccretionFlowVelocity& u, MagneticField& B,
					       double injection_time, double spectral_index, double gamma_min)
  : RadiativeTransfer(x,k,g), _ne(ne), _u(u), _B(B),
    _injection_time(injection_time), _spectral_index(spectral_index), _gamma_min(gamma_min)
{
  set_constants();
}

void RT_PowerLawSynchrotronCooling::set_frequency_scale(double omega0)
{
  RadiativeTransfer::set_frequency_scale(omega0);
  set_constants();
}

void RT_PowerLawSynchrotronCooling::set_length_scale(double L)
{
  RadiativeTransfer::set_length_scale(L);
  set_constants();
}

void RT_PowerLawSynchrotronCooling::reinitialize(const double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
  set_common_funcs();
}

void RT_PowerLawSynchrotronCooling::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  _x = x;
  _k = k;
  set_common_funcs();
}

void RT_PowerLawSynchrotronCooling::set_constants()
{
  // Emission constant (in cgs units)
  _Cjnu = VRT2_Constants::me * VRT2_Constants::c * VRT2_Constants::re
    * std::pow( 3.0, _spectral_index + 0.5 ) * std::exp(gammln(0.5*_spectral_index + 11.0/6.0) + gammln(0.5*_spectral_index + 1.0/6.0))
    / (8.0*VRT2_Constants::pi*(_spectral_index + 1.0));

  // Absorption constant (in cgs units)
  _Calphanu = 2.0 * VRT2_Constants::pi * VRT2_Constants::re * VRT2_Constants::c
    * std::pow( 3.0, _spectral_index + 1.0 ) * std::exp(gammln(0.5*_spectral_index + 25.0/12.0) + gammln(0.5*_spectral_index + 5.0/12.0))
    / 4.0;

  // Stokes Q ratios to Stokes I quantities
  _epsilon_Q = (_spectral_index+1.0)/(_spectral_index+5.0/3.0);
  _zeta_Q = (_spectral_index+1.5)/(_spectral_index+13.0/6.0);

  // Cyclotron frequency coefficients omega_B = _ComegaB * B * sin(theta)
  _ComegaB = VRT2_Constants::e / (VRT2_Constants::me * VRT2_Constants::c);
  // Put length into length scale as this is what dl is in
  _Cjnu *= _length_scale;
  _Calphanu *= _length_scale;


}

void RT_PowerLawSynchrotronCooling::set_common_funcs()
{
  FourVector<double> b = _B(_x);
  double bmag = std::sqrt( (b*b) );
  _sn_alpha = (bmag>0 ? (b*_k)/(bmag * _omega) : 1.0); // using k^2=0
  _sn_alpha = std::sqrt( std::max(0.0,1.0 - _sn_alpha*_sn_alpha) ); // deals with catastrophic subtraction
  
  //Grab time evolution parameters for spectral evolution
  _delta_t = (_x.con(0)-_injection_time)*_length_scale/VRT2::VRT2_Constants::c;
  double A = 2*VRT2::VRT2_Constants::re*VRT2::VRT2_Constants::re
              *bmag*bmag*_sn_alpha*_sn_alpha
              /(VRT2::VRT2_Constants::me*VRT2::VRT2_Constants::c);
  
  _gamma_min_time = _gamma_min/(1.0 + A*_gamma_min*_delta_t);
  if ( _delta_t > 0)
  {
    _gamma_max_time = 1.0/(A*_delta_t);

    // Density constant n_gamma = n0 gamma^-s, n0 = _Cn n_tot
    _Cn = 2.0*_spectral_index/(FastMath::pow(_gamma_min_time,-2*_spectral_index) - FastMath::pow(_gamma_max_time,-2*_spectral_index));
  
    _n0 = _Cn * _ne(_x);
  }
  else
  {
    _gamma_max_time = std::numeric_limits<double>::infinity();
    _Cn = 2.0*_spectral_index*std::pow(_gamma_min_time,2*_spectral_index);
  }

  FourVector<double> u = _u(_x);
  _omega = - (_k*u); // Get scaled omega

  if (_omega<=0 || vrt2_isnan(_omega)) {
    std::cout << "omega sick:"
	      << std::setw(15) << _omega
	      << std::setw(15) << (u*u)
	      << std::setw(15) << _x.con(0)
	      << std::setw(15) << _x.con(1)
	      << std::setw(15) << _x.con(2)
	      << std::setw(15) << _x.con(3)
	      << std::setw(15) << u.con(0)
	      << std::setw(15) << u.con(1)
	      << std::setw(15) << u.con(2)
	      << std::setw(15) << u.con(3)
	      << std::endl;

  }


  // HACK TO CHECK ANGULAR DEPENDENCE AT LOW INCLINATIONS (FACE-ON)
  //_sn_alpha = std::sqrt(2.0/3.0);

  _omegaB = _ComegaB * bmag;

  _omega *= _omega_scale; // Get real omega instead of scaled omega

  // Get rotation coeffs to align Stokes bases
  get_Stokes_alignment_angle(u,b,_cs,_sn);
}

void RT_PowerLawSynchrotronCooling::get_Stokes_alignment_angle(FourVector<double>& u, FourVector<double>& b, double& cs, double& sn)
{
  FourVector<double> uZAMO(_g), z(_g);
  uZAMO.mkcov(1.0,0.0,0.0,0.0);
  z.mkcon(0.0,std::cos(_x.con(2)),-std::sin(_x.con(2))/_x.con(1),0.0);

  double norm;  // To deal with vanishing ebperp in an okay way?

  FourVector<double> eperp = cross_product(uZAMO,_k,z);
  norm = eperp*eperp;
  eperp *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
  FourVector<double> ebperp = cross_product(u,_k,b);
  norm = ebperp*ebperp;
  ebperp *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
  FourVector<double> ebpara = cross_product(u,_k,ebperp);
  norm = ebpara*ebpara;
  ebpara *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);

  double cstmp = (eperp*ebperp);
  double sntmp = (eperp*ebpara); // Check in comparision to choice of cross product

  // Use recursion relations to get cos(2*phi), sin(2*phi)
  cs = cstmp*cstmp - sntmp*sntmp;
  sn = 2.0*cstmp*sntmp;

  if ( (vrt2_isnan(cs) || vrt2_isnan(sn)) && (_omegaB>0.0))
  {
    std::cout << "Nanned in get_Stokes_alignment_angle:"
	      << std::setw(15) << _x.con(0)
	      << std::setw(15) << _x.con(1)
	      << std::setw(15) << _x.con(2)
	      << std::setw(15) << _x.con(3)
	      << std::setw(15) << (uZAMO*uZAMO)
	      << std::setw(15) << (eperp*eperp)
	      << std::setw(15) << (ebperp*ebperp)
	      << std::setw(15) << (u*u)
	      << std::setw(15) << (b*b)
	      << std::setw(15) << cs
	      << std::setw(15) << sn
	      << std::setw(15) << cstmp
	      << std::setw(15) << sntmp
	      << std::endl;
    /*
    std::cout << "eperp : \n" << cross_product(uZAMO,_k,z) << std::endl;
    std::cout << "ebperp :\n" << cross_product(u,_k,b) << std::endl;
    std::cout << "ebpara :\n" << cross_product(u,_k,ebperp) << std::endl;
    std::cout << "_k:\n" << _k << std::endl;
    std::cout << "uZAMO:\n" << uZAMO << std::endl;
    std::cout << "u:\n" << u << std::endl;
    std::cout << "z:\n" << z << std::endl;
    std::cout << "b:\n" << b << std::endl;
    */
  }
}

double RT_PowerLawSynchrotronCooling::isotropic_absorptivity(const double dydx[])
{
  if (_omegaB > 0.0)
    //return _Calphanu * (_n0/_omega) * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1) * dl_dlambda(dydx);
    return _Calphanu * (_n0/_omega) * FastMath::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1) * dl_dlambda(dydx);
  else
    return 0.0;
}

std::valarray<double>& RT_PowerLawSynchrotronCooling::IQUV_abs(const double iquv[], const double dydx[])
{

  if (_omegaB > 0.0 && _delta_t >=0 ) {
    double consts = _Calphanu * _n0 * dl_dlambda(dydx);
    double abs;
    double omega_min = _gamma_min_time*_gamma_min_time*_omegaB;
    double omega_max = 1.5*_gamma_max_time*_gamma_max_time*_omegaB;
    if ( _delta_t == 0.0)
      omega_max = std::numeric_limits<double>::infinity();

    if (_omega>omega_min && _omega < omega_max)
      abs = consts/_omega * FastMath::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1);
    else if (_omega <= omega_min)
      //abs = consts/omega_min * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index+2.5-1) * std::pow(_omega/omega_min,1.0/3.0);
      abs = consts/omega_min * FastMath::pow(_omegaB*_sn_alpha/omega_min,_spectral_index+2.5-1) * FastMath::pow(_omega/omega_min,1.0/3.0);
    else
      abs = consts/(omega_max) * FastMath::pow(_omegaB*_sn_alpha/omega_max,_spectral_index+2.5-1)* FastMath::pow(_omega/omega_max,2.5-1)*FastMath::exp(-_omega/omega_max + 1);

    double qabs = _cs * _zeta_Q * abs;
    double uabs = _sn * _zeta_Q * abs;

    // HACK LIGHT CURVE TEST HACK
    //_iquv_abs[0] = abs*iquv[0];
    _iquv_abs[0] = abs*iquv[0] + qabs*iquv[1] + uabs*iquv[2];
    _iquv_abs[1] = qabs*iquv[0] + abs*iquv[1];
    _iquv_abs[2] = uabs*iquv[0] + abs*iquv[2];
    _iquv_abs[3] = abs*iquv[3];



    if (vrt2_isnan(_iquv_abs[0]) ||
	vrt2_isnan(_iquv_abs[1]) ||
	vrt2_isnan(_iquv_abs[2]) ||
	vrt2_isnan(_iquv_abs[3]) )
      std::cout << "Error_in_Power_Law_Synch_IQUV_ABS:"
		<< std::setw(15) << _x.con(0)
		<< std::setw(15) << _x.con(1)
		<< std::setw(15) << _x.con(2)
		<< std::setw(15) << _x.con(3)
		<< std::setw(15) << _ne(_x)
		<< std::setw(15) << _iquv_abs[0]
		<< std::setw(15) << _iquv_abs[1]
		<< std::setw(15) << _iquv_abs[2]
		<< std::setw(15) << _iquv_abs[3]
		<< std::setw(15) << _omegaB
		<< std::setw(15) << _sn_alpha
		<< std::setw(15) << _omega
		<< std::setw(15) << _zeta_Q
		<< std::setw(15) << dl_dlambda(dydx)
		<< std::setw(15) << _cs
		<< std::setw(15) << _sn
		<< std::setw(15) << abs
		<< std::setw(15) << consts
    << std::setw(15) << _gamma_min_time
    << std::setw(15) << _gamma_max_time
		<< std::endl;

  }
  else
    _iquv_abs = 0.0;

  return _iquv_abs;
}

std::valarray<double>& RT_PowerLawSynchrotronCooling::IQUV_ems(const double dydx[])
{
  if (_omegaB > 0.0 && _delta_t >= 0) {
    double consts = (_Cjnu * _omegaB * _n0/(_omega*_omega*_omega)) * dl_dlambda(dydx);
    double ems;
    double omega_min = _gamma_min_time*_gamma_min_time*_omegaB;
    double omega_max = _gamma_max_time*_gamma_max_time*_omegaB;
    if ( _delta_t == 0)
      omega_max = std::numeric_limits<double>::infinity();
    if (_omega > omega_min && _omega < omega_max)
      //ems = consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index);
      ems = consts * _sn_alpha * FastMath::pow(_omegaB*_sn_alpha/_omega,_spectral_index);
    else if ( _omega <= omega_min)
      //ems = consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index) * std::pow(_omega/omega_min,1.0/3.0);
      ems = consts * _sn_alpha * FastMath::pow(_omegaB*_sn_alpha/omega_min,_spectral_index) * FastMath::pow(_omega/omega_min,1.0/3.0);
    else
      ems = consts* _sn_alpha * FastMath::pow(_omegaB*_sn_alpha/omega_max,_spectral_index)*std::sqrt(omega_max/_omega)*FastMath::exp(-_omega/omega_max+ 1.0);

    _iquv_ems[0] = ems;
    _iquv_ems[1] = _cs * _epsilon_Q * ems;
    _iquv_ems[2] = _sn * _epsilon_Q * ems;
  }
  else
    _iquv_ems = 0.0;

  if (vrt2_isnan(_iquv_ems[0]) ||
      vrt2_isnan(_iquv_ems[1]) ||
      vrt2_isnan(_iquv_ems[2]) )//||
    //std::fabs(_x.con(0)-(-3078.78))<0.1 )
    std::cout << "Error in Power Law Synch, IQUV_EMS:"
	      << std::setw(15) << _iquv_ems[0]
	      << std::setw(15) << _iquv_ems[1]
	      << std::setw(15) << _iquv_ems[2]
	      << std::setw(15) << _omegaB
	      << std::setw(15) << _sn_alpha
	      << std::setw(15) << _omega
	      << std::setw(15) << _epsilon_Q
	      << std::setw(15) << _cs
	      << std::setw(15) << _sn
	      << std::setw(15) << _n0
        << std::setw(15) << _gamma_min_time
        << std::setw(15) << _gamma_max_time
	      << std::endl;

  return _iquv_ems;
}

double RT_PowerLawSynchrotronCooling::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

double RT_PowerLawSynchrotronCooling::gammln(double xx)
{
  double x,y,tmp,ser;
  static double cof[6]={76.18009172947146,-86.50532032941677,
			24.01409824083091,-1.231739572450155,
			0.1208650973866179e-2,-0.5395239384953e-5};
  int j;

  y=x=xx;
  tmp=x+5.5;
  tmp -= (x+0.5)*log(tmp);
  ser=1.000000000190015;
  for (j=0;j<=5;j++)
    ser += cof[j]/++y;
  return -tmp+log(2.5066282746310005*ser/x);
}


void RT_PowerLawSynchrotronCooling::dump(std::ostream& dout, double dydx[])
{
  double ems, abs;
  if (_omegaB > 0.0) {
    double ems_consts = _Cjnu * _omegaB * _n0 * std::pow(_omega,-3.0) * dl_dlambda(dydx);
    double omega_min = _gamma_min*_gamma_min*_omegaB;
    if (_omega > omega_min)
      ems = ems_consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index);
    else
      ems = ems_consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index) * std::pow(_omega/omega_min,1.0/3.0);


    double abs_consts = _Calphanu * _n0 * dl_dlambda(dydx);
    if (_omega>omega_min)
      abs = abs_consts/_omega * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1);
    else
      abs = abs_consts/omega_min * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index+2.5-1) * std::pow(_omega/omega_min,1.0/3.0);


  }
  else
  {
    ems = 0;
    abs = 0;
  }



  FourVector<double> u = _u(_x);



  dout << std::setw(15) << ems
       << std::setw(15) << abs
       << std::setw(15) << _ne(_x)
       << std::setw(15) << _omegaB
       << std::setw(15) << _omega
       << std::setw(15) << _sn_alpha
       << std::setw(15) << u.cov(0)
       << std::setw(15) << _cs
       << std::setw(15) << _sn
       << std::setw(15) << _gamma_min_time
       << std::setw(15) << _gamma_max_time
       << std::setw(15) << dl_dlambda(dydx);

}
};

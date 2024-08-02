#include "rt_axion.h"
#include "fast_math.h"

namespace VRT2 {
RT_Axion::RT_Axion(Metric& g,
		   AccretionFlowVelocity& u,
		   double dn, double ma, double ga, int n,int l, int m)
  : RadiativeTransfer(g), _u(u), _dn(dn), _ma(ma), _ga(ga), _n(n), _l(l), _m(m)
{
  set_constants();
}
RT_Axion::RT_Axion(const double y[], Metric& g,
		   AccretionFlowVelocity& u,
		   double dn, double ma, double ga, int n,int l, int m)
  : RadiativeTransfer(g), _u(u), _dn(dn), _ma(ma), _ga(ga), _n(n), _l(l), _m(m)
{
  set_constants();
}
RT_Axion::RT_Axion(FourVector<double>& x, FourVector<double>& k, Metric& g,
		   AccretionFlowVelocity& u,
		   double dn, double ma, double ga, int n,int l, int m)
  : RadiativeTransfer(g), _u(u), _dn(dn), _ma(ma), _ga(ga), _n(n), _l(l), _m(m)
{
  set_constants();
}

void RT_Axion::set_frequency_scale(double omega0)
{
  RadiativeTransfer::set_frequency_scale(omega0);
  set_constants();
}

void RT_Axion::set_length_scale(double L)
{
  RadiativeTransfer::set_length_scale(L);
  set_constants();
}

void RT_Axion::reinitialize(const double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
  set_common_funcs();
}

void RT_Axion::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  _x = x;
  _k = k;
  set_common_funcs();
}

void RT_Axion::set_constants() // Start-up functions/quantities, things that can be defined only at very beginning.
{
  // NEEDED CONSTANTS:  (ZHIREN)
  // 1. beta 
  // 2. omega_axion
  // 3. wave function normalization
  // 4. alpha  [ sqrt(9/4-alpha) ]




  // CONSTANTS FROM SYNCHROTRON, NOT NECESSARY BUT PROVIDES GUIDANCE.
  // // Emission constant (in cgs units)
  // _Cjnu = VRT2_Constants::me * VRT2_Constants::c * VRT2_Constants::re
  //   * std::pow( 3.0, _spectral_index + 0.5 ) * std::exp(gammln(0.5*_spectral_index + 11.0/6.0) + gammln(0.5*_spectral_index + 1.0/6.0))
  //   / (8.0*VRT2_Constants::pi*(_spectral_index + 1.0));

  // // Absorption constant (in cgs units)
  // _Calphanu = 2.0 * VRT2_Constants::pi * VRT2_Constants::re * VRT2_Constants::c
  //   * std::pow( 3.0, _spectral_index + 1.0 ) * std::exp(gammln(0.5*_spectral_index + 25.0/12.0) + gammln(0.5*_spectral_index + 5.0/12.0))
  //   / 4.0;

  // // Stokes Q ratios to Stokes I quantities
  // _epsilon_Q = (_spectral_index+1.0)/(_spectral_index+5.0/3.0);
  // _zeta_Q = (_spectral_index+1.5)/(_spectral_index+13.0/6.0);

  // // Cyclotron frequency coefficients omega_B = _ComegaB * B * sin(theta)
  // _ComegaB = VRT2_Constants::e / (VRT2_Constants::me * VRT2_Constants::c);
  // // Density constant n_gamma = n0 gamma^-s, n0 = _Cn n_tot
  // //_Cn = 2.0*_spectral_index*std::pow(_gamma_min,2*_spectral_index+1);
  // _Cn = 2.0*_spectral_index*std::pow(_gamma_min,2*_spectral_index);

  // // Put length into length scale as this is what dl is in
  // _Cjnu *= _length_scale;
  // _Calphanu *= _length_scale;
}

void RT_Axion::set_common_funcs() // Every point functions that might be shared among radiative coefficients (ems, abs)
{
  // NEEDED FUNCTIONS/EVALUATIONS:  (ZHIREN)
  // 0. da/dt
  // 1. da/dr
  // 2. da/dtheta
  // 3. da/dphi




  // CONSTANTS FROM SYNCHROTRON, NOT NECESSARY BUT PROVIDES GUIDANCE.
  // _n0 = _Cn * _ne(_x);

  // FourVector<double> u = _u(_x);
  // _omega = - (_k*u); // Get scaled omega

  // if (_omega<=0 || vrt2_isnan(_omega)) {
  //   std::cout << "omega sick:"
  // 	      << std::setw(15) << _omega
  // 	      << std::setw(15) << (u*u)
  // 	      << std::setw(15) << _x.con(0)
  // 	      << std::setw(15) << _x.con(1)
  // 	      << std::setw(15) << _x.con(2)
  // 	      << std::setw(15) << _x.con(3)
  // 	      << std::setw(15) << u.con(0)
  // 	      << std::setw(15) << u.con(1)
  // 	      << std::setw(15) << u.con(2)
  // 	      << std::setw(15) << u.con(3)
  // 	      << std::endl;

  // }


  // FourVector<double> b = _B(_x);
  // double bmag = std::sqrt( (b*b) );
  // _sn_alpha = (bmag>0 ? (b*_k)/(bmag * _omega) : 1.0); // using k^2=0
  // _sn_alpha = std::sqrt( std::max(0.0,1.0 - _sn_alpha*_sn_alpha) ); // deals with catastrophic subtraction

  // // HACK TO CHECK ANGULAR DEPENDENCE AT LOW INCLINATIONS (FACE-ON)
  // //_sn_alpha = std::sqrt(2.0/3.0);

  // _omegaB = _ComegaB * bmag;

  // _omega *= _omega_scale; // Get real omega instead of scaled omega

  // // Get rotation coeffs to align Stokes bases
  // get_Stokes_alignment_angle(u,b,_cs,_sn);
}

// void RT_Axion::get_Stokes_alignment_angle(FourVector<double>& u, FourVector<double>& b, double& cs, double& sn)
// {
//   FourVector<double> uZAMO(_g), z(_g);
//   uZAMO.mkcov(1.0,0.0,0.0,0.0);
//   z.mkcon(0.0,std::cos(_x.con(2)),-std::sin(_x.con(2))/_x.con(1),0.0);

//   double norm;  // To deal with vanishing ebperp in an okay way?

//   FourVector<double> eperp = cross_product(uZAMO,_k,z);
//   norm = eperp*eperp;
//   eperp *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
//   FourVector<double> ebperp = cross_product(u,_k,b);
//   norm = ebperp*ebperp;
//   ebperp *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
//   FourVector<double> ebpara = cross_product(u,_k,ebperp);
//   norm = ebpara*ebpara;
//   ebpara *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);

//   double cstmp = (eperp*ebperp);
//   double sntmp = (eperp*ebpara); // Check in comparision to choice of cross product

//   // Use recursion relations to get cos(2*phi), sin(2*phi)
//   cs = cstmp*cstmp - sntmp*sntmp;
//   sn = 2.0*cstmp*sntmp;

//   if ( (vrt2_isnan(cs) || vrt2_isnan(sn)) && (_omegaB>0.0))
//   {
//     std::cout << "Nanned in get_Stokes_alignment_angle:"
// 	      << std::setw(15) << _x.con(0)
// 	      << std::setw(15) << _x.con(1)
// 	      << std::setw(15) << _x.con(2)
// 	      << std::setw(15) << _x.con(3)
// 	      << std::setw(15) << (uZAMO*uZAMO)
// 	      << std::setw(15) << (eperp*eperp)
// 	      << std::setw(15) << (ebperp*ebperp)
// 	      << std::setw(15) << (u*u)
// 	      << std::setw(15) << (b*b)
// 	      << std::setw(15) << cs
// 	      << std::setw(15) << sn
// 	      << std::setw(15) << cstmp
// 	      << std::setw(15) << sntmp
// 	      << std::endl;
//     /*
//     std::cout << "eperp : \n" << cross_product(uZAMO,_k,z) << std::endl;
//     std::cout << "ebperp :\n" << cross_product(u,_k,b) << std::endl;
//     std::cout << "ebpara :\n" << cross_product(u,_k,ebperp) << std::endl;
//     std::cout << "_k:\n" << _k << std::endl;
//     std::cout << "uZAMO:\n" << uZAMO << std::endl;
//     std::cout << "u:\n" << u << std::endl;
//     std::cout << "z:\n" << z << std::endl;
//     std::cout << "b:\n" << b << std::endl;
//     */
//   }
// }

double RT_Axion::isotropic_absorptivity(const double dydx[])
{
  return 0.0;
}

std::valarray<double>& RT_Axion::IQUV_abs(const double iquv[], const double dydx[])
{
  // Define the "rotativity", K, here, in terms of:
  //   * the time _x.con(0)
  //   * the radial position _x.con(1)
  //   * the theta position _x.con(2)
  //   * the phi position _x.con(3)
  // in Boyer-Lindquist coords.
  
  // I think you need to determine the Axion cloud density, given, dn, ma, n, l, m. 
  // And you need various constants to get to K with ga.
  // 

  double K = 0;

  // I, Q, U, V
  _iquv_abs[0] = 0.0;
  _iquv_abs[1] = K*iquv[2];  // dQ/dz =  K U
  _iquv_abs[2] = -K*iquv[1]; // dU/dz = -K Q
  _iquv_abs[3] = 0.0;

  return _iquv_abs;
}

std::valarray<double>& RT_Axion::IQUV_ems(const double dydx[])
{
  _iquv_ems = 0.0;
  return _iquv_ems;
}

double RT_Axion::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

void RT_Axion::dump(std::ostream& dout, double dydx[])
{
//   double ems, abs;
//   if (_omegaB > 0.0) {
//     double ems_consts = _Cjnu * _omegaB * _n0 * std::pow(_omega,-3.0) * dl_dlambda(dydx);
//     double omega_min = _gamma_min*_gamma_min*_omegaB;
//     if (_omega > omega_min)
//       ems = ems_consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index);
//     else
//       ems = ems_consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index) * std::pow(_omega/omega_min,1.0/3.0);


//     double abs_consts = _Calphanu * _n0 * dl_dlambda(dydx);
//     if (_omega>omega_min)
//       abs = abs_consts/_omega * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1);
//     else
//       abs = abs_consts/omega_min * std::pow(_omegaB*_sn_alpha/omega_min,_spectral_index+2.5-1) * std::pow(_omega/omega_min,1.0/3.0);


//   }
//   else
//   {
//     ems = 0;
//     abs = 0;
//   }



//   FourVector<double> u = _u(_x);



//   dout << std::setw(15) << ems
//        << std::setw(15) << abs
//        << std::setw(15) << _ne(_x)
//        << std::setw(15) << _omegaB
//        << std::setw(15) << _omega
//        << std::setw(15) << _sn_alpha
//        << std::setw(15) << u.cov(0)
//        << std::setw(15) << _cs
//        << std::setw(15) << _sn
//        << std::setw(15) << dl_dlambda(dydx);

}
};

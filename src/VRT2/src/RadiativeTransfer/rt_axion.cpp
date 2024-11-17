#include "rt_axion.h"
#include "fast_math.h"

//ma should be in M^-1
namespace VRT2 {
RT_Axion::RT_Axion(Metric& g,
		   AccretionFlowVelocity& u,
		   double M, double ma, double ga)
  : RadiativeTransfer(g), _u(u), _M(M), _ma(ma), _ga(ga)
{
  set_constants();
}
RT_Axion::RT_Axion(const double y[], Metric& g,
		   AccretionFlowVelocity& u,
		   double M, double ma, double ga) //, int n,int l, int m)
  : RadiativeTransfer(g), _u(u), _ma(ma), _ga(ga) //, _n(n), _l(l), _m(m)
{
  set_constants();
}
RT_Axion::RT_Axion(FourVector<double>& x, FourVector<double>& k, Metric& g,
		   AccretionFlowVelocity& u,
		   double M, double ma, double ga) //, int n,int l, int m)
  : RadiativeTransfer(g), _u(u), _ma(ma), _ga(ga) //, _n(n), _l(l), _m(m)
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
// std::tuple<double, double, double> RT_Axion::set_constants(double alpha)
{
  double Mg = VRT2::VRT2_Constants::M_sun * _M; // Black hole mass in g (from Msun)
  // std::cout << "Mg (Black hole mass in grams): " << Mg << std::endl;
  double mg = _ma * 1.78266192e-33; // Axion mass in g (from eV)
  // std::cout << "mg (Axion mass in grams): " << mg << std::endl;
  double spin =  _g.ang_mom() / _g.mass();
  // std::cout << "spin: " << spin << std::endl;

  _alpha = VRT2::VRT2_Constants::G * Mg * mg / (VRT2::VRT2_Constants::hbar * VRT2::VRT2_Constants::c); // calculate coupling constant alpha
  // std::cout << "_alpha (Coupling constant alpha): " << _alpha << std::endl;
  _mu = _alpha / 1.0 ; // ALP mass in 1/M_sgra
  // std::cout << "_mu (ALP mass in  1/M_sgra): " << _mu << std::endl;

  // _M should be dimensionless, in terms of M_sun, _mu the same
  _rp = r_p(spin); // THIS DOESN'T MAKE SENSE
  _rm = r_m(spin); // THIS DOESN'T MAKE SENSE
  _omega_crit = omega_crit(spin);
  _omega_21 = omega_21(_mu, spin);
  _sigma = sigma(_mu, spin);
  _q = q_term(_mu, spin);
  _chi = x_term(_mu, spin);

  // std::cout << "_rp (Event horizon radius r+): " << _rp << std::endl;
  // std::cout << "_rm (Event horizon radius r-): " << _rm << std::endl;
  // std::cout << "_omega_crit (Critical angular frequency): " << _omega_crit << std::endl;
  // std::cout << "_omega_21 (Transition frequency omega_21): " << _omega_21 << std::endl;
  // std::cout << "_sigma: " << _sigma << std::endl;
  // std::cout << "_q (q-term): " << _q << std::endl;
  // std::cout << "_chi (x-term): " << _chi << std::endl;

  // Find the maximum value of Re(a) to normalize it first
  // double _t_find_max = 0;
  // double _theta_find_max = M_PI / 2;
  // double _phi_find_max = 0;
  double r_min = 2.0 ;   // start from 2M_sgra
  double r_max = 302.0 ; // end at 302M_sgra, which is enough to cover the maximum for low alpha
  double step = 0.1 ;    // step size
  // _Re_a_max = find_max_Re_a_not_normed(_t_find_max, _theta_find_max, _phi_find_max, r_min, r_max, step);
  _Re_a_max = find_max_Re_a_not_normed(r_min, r_max, step);
  // std::cout << "_Re_a_max (Maximum Re(a) not normalized): " << _Re_a_max << std::endl;

  // _R_norm_factor = 1; // my a0
  // _norm_factor = -0.5 * std::sqrt(3 / (2 * M_PI)) * _R_norm_factor;
  _norm_factor = std::pow(10, 24) / _Re_a_max; // merge R and Y normed factors plus the scaling factor to make amax ~ fa. Pick fa = 10^15 GeV = 10^24 eV
  // std::cout << "_norm_factor (Normalization factor): " << _norm_factor << std::endl;

  // plot_Re_a_not_normed(r_min, r_max, step);

  // std::cout << "Constants:"
  //     << std::setw(15) << "spin: " << spin
  //     << std::setw(15) << "_alpha:" << _alpha
  //     << std::setw(15) << "_mu:" << _mu
  //     << std::setw(15) << "_rp:" << _rp
  //     << std::setw(15) << "_rm:" << _rm
  //     << std::setw(15) << "_omega_crit:" << _omega_crit
  //     << std::setw(15) << "_omega_21:" << _omega_21
  //     << std::setw(15) << "_sigma:" << _sigma
  //     << std::setw(15) << "_q:" << _q
  //     << std::setw(15) << "_chi:" << _chi
  //     << std::setw(15) << "_Re_a_max:" << _Re_a_max
  //     << std::setw(15) << "_norm_factor:" << _norm_factor
  //     << std::endl;

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

// These helper functions are all in natural units G=hbar=c=1 and in M_sgrA
double RT_Axion::r_p(double spin)
{
  return 1 + std::sqrt(1 - spin * spin);
}

double RT_Axion::r_m(double spin)
{
  return 1 - std::sqrt(1 - spin * spin);
}

double RT_Axion::omega_crit(double spin)
{
  return spin * 1 / (2 * r_p(spin)); // m=1
}

double RT_Axion::omega_21(double mu, double spin)
{
  return mu * (1 - _alpha * _alpha / 8.0 - std::pow(_alpha, 4) / 128 - std::pow(_alpha, 4) / 8.0 + (spin * std::pow(_alpha, 5)) / 12); // n=2, l=m=1
}

double RT_Axion::sigma(double mu, double spin)
{
  return (2 * r_p(spin) * (omega_21(mu, spin) - omega_crit(spin))) / (r_p(spin) - r_m(spin));
}

double RT_Axion::q_term(double mu, double spin)
{
  return -std::sqrt(mu * mu - omega_21(mu, spin) * omega_21(mu, spin));
}

double RT_Axion::x_term(double mu, double spin)
{
  return (mu * mu - 2 * omega_21(mu, spin) * omega_21(mu, spin)) / q_term(mu, spin); 
}


double RT_Axion::Re_a_not_normed(double r)
{
  // note that norm factor is 1, suppose we're at t=phi=0, theta=pi/2
  // return _R1 * std::cos(_arg) * std::sin(theta);
  return std::pow((r - _rm), _chi - 1) * std::exp(_q * r) * std::cos(_sigma * std::log((r - _rm) / (r - _rp)));
}

double RT_Axion::find_max_Re_a_not_normed(double r_min, double r_max, double step)
{
  double max_value = -1e20; // Initialize to a very small value
  //double max_r = r_min;     // Store the r corresponding to the max value

  for (double r = r_min; r <= r_max; r += step)
  {
    // Calculate Re_a_not_normed at (t=0, theta=pi/2, phi=0)
    double value = Re_a_not_normed(r);

    if (value > max_value)
    {
      max_value = value;
      //max_r = r; // Store the r value where max occurs
    }
  }
  return max_value;
}

// #include <fstream> // Include this for file operations

// void RT_Axion::plot_Re_a_not_normed(double r_min, double r_max, double step)
// {
//     std::ofstream outfile("Re_a_vs_r.txt"); // Open a file to write data
//     if (!outfile)
//     {
//         std::cerr << "Error opening file for writing." << std::endl;
//         return;
//     }

//     for (double r = r_min; r <= r_max; r += step)
//     {
//         double value = Re_a_not_normed(r);
//         outfile << r << " " << value << std::endl; // Write the data pair to the file
//     }
//     outfile.close(); // Close the file after writing
// }

void RT_Axion::set_common_funcs() // Every point functions that might be shared among radiative coefficients (ems, abs)
{

  double t = _x.con(0); // This is t
  double r = _x.con(1); // This is r
  double phi = _x.con(3); // This is phi

  // _R1_left_low = r - _rm;
  // _R1_left = std::pow(_R1_left_low, _chi - 1);
  // _R1_right = std::exp(_q * r);
  // _R1 = _R1_left * _R1_right;
  _R1 = std::pow((r - _rm), _chi - 1) * std::exp(_q * r);
  _arg = phi - _omega_21 * t + _sigma * std::log((r - _rm) / (r - _rp));

  // std::cout << "RT_Axion time = "
  // 	    << std::setw(15) << t/_M
  // 	    << " _omega_21*t = "
  // 	    << std::setw(15) << _omega_21*(t/_M)
  // 	    << std::endl;

  // std::cout << "Common functions:"
  //     << std::setw(15) << "_R1_left_low:" << _R1_left_low
  //     << std::setw(15) << "_R1_left:" << _R1_left
  //     << std::setw(15) << "_R1_right:" << _R1_right
  //     << std::setw(15) << "_R1:" << _R1
  //     << std::setw(15) << "_arg:" << _arg
  //     << std::endl;
  // double _R1_tbc = std::pow((7 * _M - _rm), _chi - 1) * std::exp(_q * 7 * _M);
  // double _arg_tbc = 0 - _omega_21 * (-1.05197e+08) + _sigma * std::log((7 * _M - _rm) / (7 * _M - _rp));

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
double RT_Axion::Real_a(double t, double r, double theta, double phi)
{
  // The full form of Real axion field
  return _norm_factor * _R1 * std::cos(_arg) * std::sin(theta);
}

    double RT_Axion::dadr(double t, double r, double theta, double phi)
{
  // no change of sign

  double first_part = _R1 / ((r - _rm) * (r - _rp));
  double second_part = (r - _rp) * (-1 + _q * (r - _rm) + _chi) * std::cos(_arg);
  double third_part = _sigma * (_rp - _rm) * std::sin(_arg);

  // std::cout << "Parts of dadr:"
  //     << std::setw(15) << "first_part:" << first_part
  //     << std::setw(15) << "second_part:" << second_part
  //     << std::setw(15) << "third_part:" << third_part
  //     << std::endl;

  return _norm_factor * first_part * (second_part + third_part) * std::sin(theta);
}

double RT_Axion::dadt(double t, double r, double theta, double phi)
{
  // no change of sign

  return _norm_factor * _R1 * _omega_21 * std::sin(_arg) * std::sin(theta);
}

double RT_Axion::dadtheta(double t, double r, double theta, double phi)
{
  // no change of sign

  return _norm_factor * _R1 *  std::cos(_arg) * std::cos(theta);
}

double RT_Axion::dadphi(double t, double r, double theta, double phi)
{
  // change of sign!

  return -_norm_factor * _R1  * std::sin(_arg) * std::sin(theta);
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

  FourVector<double> da_dx(_g);
  da_dx.mkcov(dadt(_x.con(0), _x.con(1), _x.con(2), _x.con(3)),
              dadr(_x.con(0), _x.con(1), _x.con(2), _x.con(3)),
              dadtheta(_x.con(0), _x.con(1), _x.con(2), _x.con(3)),
              dadphi(_x.con(0), _x.con(1), _x.con(2), _x.con(3)));

  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  double K = -2 * _ga * (da_dx * dx_dlam);

  double _r_test = 7.0;
  double _theta_test = M_PI / 2;
  double _phi_test = 0;

  // Check finite difference
  double _dadr_4rg = dadr(_x.con(0), 4.0, _theta_test, _phi_test);
  double _delta_ar_4rg = (Real_a(_x.con(0), 4.00001, _theta_test, _phi_test) - Real_a(_x.con(0), 4.00001, _theta_test, _phi_test)) / 0.00002;
  double _dadr_7rg = dadr(_x.con(0), _r_test, _theta_test, _phi_test);
  double _delta_ar_7rg = (Real_a(_x.con(0), _r_test+0.00001, _theta_test, _phi_test) - Real_a(_x.con(0), _r_test-0.00001, _theta_test, _phi_test))/0.00002;

  FourVector<double>
      da_dx_test(_g);
  da_dx_test.mkcov(dadt(_x.con(0), _r_test, _theta_test, _phi_test),
                   dadr(_x.con(0), _r_test, _theta_test, _phi_test),
                   dadtheta(_x.con(0), _r_test, _theta_test, _phi_test),
                   dadphi(_x.con(0), _r_test, _theta_test, _phi_test));

  std::cout << "K:"
            << std::setw(15) << K
            // << std::setw(15) << _ga
            // << " | "
            << std::setw(15) << _x.con(0)
            // << std::setw(15) << _x.con(1)
            // << std::setw(15) << _x.con(2)
            // << std::setw(15) << _x.con(3)
            // << " | "
            // << std::setw(15) << dx_dlam.con(0)
            // << std::setw(15) << dx_dlam.con(1)
            // << std::setw(15) << dx_dlam.con(2)
            // << std::setw(15) << dx_dlam.con(3)
            << " | "
            << std::setw(15) << da_dx_test.cov(0)
            << std::setw(15) << da_dx_test.cov(1)
            << std::setw(15) << da_dx_test.cov(2)
            << std::setw(15) << da_dx_test.cov(3)
            << " | "
            << std::setw(15) << _dadr_4rg
            << std::setw(15) << _delta_ar_4rg
            << std::setw(15) << _dadr_7rg
            << std::setw(15) << _delta_ar_7rg
            // << std::setw(15) << (dx_dlam*dx_dlam)
            // << std::setw(15) << (da_dx*da_dx)
            // << std::setw(15) << (_k*_k)
            // << " | "
            // << std::setw(15) << K
            << std::endl;

  // I, Q, U, V
  _iquv_abs[0] = 0.0;
  _iquv_abs[1] = K * iquv[2];  // dQ/dz =  K U
  _iquv_abs[2] = -K * iquv[1]; // dU/dz = -K Q
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

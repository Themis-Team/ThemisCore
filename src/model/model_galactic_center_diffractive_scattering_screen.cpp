/*!
    \file model_galactic_center_diffractive_scattering_screen.cpp  
    \author Paul Tiede
    \date October 2018
    \brief implementation file for galactic center diffractive scattering interface.
    \details 
*/

#include "model_galactic_center_diffractive_scattering_screen.h"
#include <cmath>
#include "constants.h"

namespace Themis {


model_galactic_center_diffractive_scattering_screen::model_galactic_center_diffractive_scattering_screen(
                    model_visibility_amplitude& model, 
                    double frequency,
                    std::string scattering_model,
                    double observer_screen_distance, double source_screen_distance,
                    double theta_maj_mas_cm, double theta_min_ma_cm, double POS_ANG, 
                    double scatt_alpha, double r_in, double r_out)
  :_model(model), 
   //_frequency(frequency),
   _wavelength(constants::c/frequency), 
   _scattering_model(scattering_model), 
   //_observer_screen_distance(observer_screen_distance),
   //_source_screen_distance(source_screen_distance), 
   _theta_maj_mas_cm(theta_maj_mas_cm), _theta_min_mas_cm(theta_min_ma_cm), 
   //_POS_ANG(-POS_ANG*M_PI/180.0),
   _phi0((90-POS_ANG)*M_PI/180.0), 
   _M(observer_screen_distance/source_screen_distance), 
   //_rF(std::sqrt(_observer_screen_distance*_wavelength/(2*M_PI)/(1+_M))), 
   _scatt_alpha(scatt_alpha),
   _rin(r_in),
   //_rout(r_out), 
   _P_phi_func("none", scattering_model, scatt_alpha, _phi0)
{
    
  double lambdabar_0 = 1.0/(2.0*M_PI);
  double gammaA2 = std::tgamma(1.0 - _scatt_alpha/2.0);
  double rinL2 = _rin*_rin/(lambdabar_0*lambdabar_0);
  _Qbar = M_PI*M_PI*(rinL2*(1.0+_M))*(rinL2*(1.0+_M))/(gammaA2*std::log(2))*
          (_theta_maj_mas_cm*_theta_maj_mas_cm + _theta_min_mas_cm*_theta_min_mas_cm);
  //convert Qbar to radians^2 from mas^2
  _Qbar *= 1.0/1000.0/3600.0*M_PI/180.0;
  _Qbar *= 1.0/1000.0/3600.0*M_PI/180.0;

  //asymmetry needed for kzeta
  double A = _theta_maj_mas_cm/_theta_min_mas_cm;


  //Find the scattering coefficent at reference wavelength of 1cm
  _C_scatt_0 = _Qbar*gammaA2/(8.0*M_PI*M_PI*rinL2);
  double B_prefac = std::pow(2.0,3.0 - _scatt_alpha)*std::sqrt(M_PI)/(_scatt_alpha*std::tgamma((_scatt_alpha+1.0)/2.0));
  //Scattering asymmetry which is needed for find kZeta for the different models


  //Create Gaussian quadrature object
  GaussianQuadrature Quad;

  //Measure of asymmetry needed to find kZeta
  _zeta0 = (A*A-1)/(A*A+1);
  //Fills anisotropy and screen parameters for the angular distributions
  if ( scattering_model == "vonMises" ){

    //Interpolator for kZeta given some empirical zeta0 = (A^2-1)/(A^2+1)
    Interpolator1D kZetainterp(utils::global_path("src/util/vonMises_kzeta_table.dat"), 2, 1);
    _kZeta = kZetainterp(_zeta0);

  }
  else if ( scattering_model == "dipole" ){

    //First read in the file since the 2d interpolator doesn't have this built in
    
    std::string fname = utils::global_path("src/util/dipole_kzeta_table.dat"); 
    std::ifstream readin(fname);
    if ( !readin.is_open() ){
      std::cerr << fname << " not found.\n";
      std::exit(1);
    }
    std::string stmp;
    size_t Nx,Ny;
    readin >> stmp >>  stmp >> Nx;
    readin.ignore(4096,'\n');
    readin >> stmp >> stmp >> Ny;
    readin.ignore(4096,'\n');
    readin.ignore(4096,'\n');


    std::vector<std::vector<double> > zeta0T(Nx,std::vector<double> (Ny,0.0)),
                                      alphaT(Nx,std::vector<double> (Ny,0.0)),
                                      kZetaT(Nx,std::vector<double> (Ny,0.0));

    for ( size_t i = 0; i < Nx; ++i )
      for ( size_t j = 0; j < Ny; ++j )
      {
        double x, y, z;
        readin >> x >> y >> z;
        zeta0T[i][j] = x;
        alphaT[i][j] = y;
        kZetaT[i][j] = z;

        //std::cerr << std::setw(15) << x
        //          << std::setw(15) << y
        //          << std::setw(15) << z << std::endl;
      }
    readin.close();
    //Now assign the interpolator targets
    std::valarray<double> i2_zeta0(Nx), i2_scatt_alpha(Ny), i2_kZeta(Nx*Ny);
    for ( size_t i = 0; i < Nx; ++i)
      i2_zeta0[i] = zeta0T[i][0];
    for ( size_t i = 0; i < Nx; ++i)
      i2_scatt_alpha[i] = alphaT[0][i];
    for ( size_t j = 0; j < Ny; ++j )
      for ( size_t i = 0; i < Nx; ++i )
        i2_kZeta[j+Nx*i] = kZetaT[i][j];



    
    Interpolator2D kZetainterp(i2_zeta0, i2_scatt_alpha, i2_kZeta);
    kZetainterp.bicubic(_zeta0, _scatt_alpha, _kZeta);

  }
  
  else if ( scattering_model == "boxcar" ){

    Interpolator1D kZetainterp(utils::global_path("src/util/boxcar_kzeta_table.dat"), 2, 1);
    _kZeta = kZetainterp(_zeta0);
  }
  else{
    std::cerr << "Scattering model angular distribution not recongnized! Use 'vonMises', 'dipole', or 'boxcar'\n";
    std::abort();
  }

  
  //Find the correct normalization of the distribution
  _P_phi_func.set_kZeta(_kZeta);
  _P_phi_norm = Quad.integrate(_P_phi_func, 0, 2*M_PI);

  //Find the Bmaj integral
  _P_phi P_phi_maj("major", scattering_model, _scatt_alpha, _phi0);
  P_phi_maj.set_kZeta(_kZeta);
  double int_maj = Quad.integrate(P_phi_maj, 0, 2*M_PI)/_P_phi_norm;

  //Find the Bmin integral
  _P_phi P_phi_min("minor", scattering_model, _scatt_alpha, _phi0);
  P_phi_min.set_kZeta(_kZeta);
  double int_min = Quad.integrate(P_phi_min, 0, 2*M_PI)/_P_phi_norm;

  
  _Bmaj = B_prefac*int_maj/(1+_zeta0);
  _Bmin = B_prefac*int_min/(1-_zeta0);

  /*
  std::cout << "P_phi_prefac : " << 1.0/_P_phi_norm << std::endl;
  std::cout << "kZeta : " << _kZeta << std::endl;
  std::cout << "Qbar : " << _Qbar << std::endl;
  std::cout << "C_scatt_0 : "  << _C_scatt_0 << std::endl;
  std::cout << "int_maj : "  << int_maj << std::endl;
  std::cout << "int_min : "  << int_min << std::endl;
  std::cout << "Bmaj_0 : " << _Bmaj*(1+_zeta0)/2.0*_C_scatt_0 << std::endl;
  std::cout << "Bmin_0 : " << _Bmin*(1-_zeta0)/2.0*_C_scatt_0 << std::endl;
  std::cout << "2/aBmaj : " << 2.0/(_scatt_alpha*_Bmaj) << std::endl;
  std::cout << "2/aBmin : " << 2.0/(_scatt_alpha*_Bmin) << std::endl;
  */

}

double model_galactic_center_diffractive_scattering_screen::Dphi(double u, double v) const
{
  double r = std::sqrt(u*u + v*v)*_wavelength;
  //std::cerr << r/_wavelength << std::endl;
  double rrin2 = r*r/(_rin*_rin);
  double phi = std::atan2(v,u);

  double pMaj = 2.0/(_scatt_alpha*_Bmaj);
  double pMin = 2.0/(_scatt_alpha*_Bmin);


  double Dmaj = _wavelength*_wavelength*_C_scatt_0*(1+_zeta0)/2.0*_Bmaj*std::pow(pMaj,-_scatt_alpha/(2.0-_scatt_alpha))*
                ( std::pow( 1 + std::pow( pMaj, 2.0/(2.0-_scatt_alpha) )*rrin2 ,_scatt_alpha/2.0) -1 );
                
  double Dmin = _wavelength*_wavelength*_C_scatt_0*(1-_zeta0)/2.0*_Bmin*std::pow(pMin,-_scatt_alpha/(2.0-_scatt_alpha))*
                ( std::pow( 1 + std::pow( pMin, 2.0/(2.0-_scatt_alpha) )*rrin2 ,_scatt_alpha/2.0) -1 );
  
  
  return (Dmaj+Dmin)/2.0 + (Dmaj-Dmin)/2.0*std::cos(2.0*(phi-_phi0));
}

double model_galactic_center_diffractive_scattering_screen::Q(double x, double y) const
{
  //2pi is because Q in Psaltis et. al. use wavenumber.
  double q = 2*M_PI*std::sqrt(x*x+y*y) + 1e-12/_rin;
  double phiq = std::atan2(y,x);
  double qrin = q*_rin;

  return _Qbar*std::pow(qrin, -_scatt_alpha - 2.0)*std::exp(-qrin*qrin)*_P_phi_func(phiq)/_P_phi_norm;
}


model_galactic_center_diffractive_scattering_screen::_P_phi::_P_phi(std::string axis, std::string model, double alpha, double phi0)
  :_axis(axis),_model(model),_alpha(alpha),_phi0(phi0)
{
  if (_axis == "major"){
    _phaseshift = 0;
    _angPow = 1.0;
  }
  else if (_axis == "minor"){
    _phaseshift = -M_PI/2.0;
    _angPow = 1.0;
  }
  else if (_axis == "none"){
    _phaseshift = 0;
    _angPow = 0.0;
  }
  else{
    std::cerr << "axis must be 'major' 'minor' or 'none'\n";
    std::exit(1);
  }
}


double model_galactic_center_diffractive_scattering_screen::_P_phi::operator()(double phi) const
{
  
  double angPart = std::pow(std::abs(std::cos(_phi0 - phi+_phaseshift)), _alpha*_angPow);
  
  
  if (_model == "vonMises"){
    return angPart*std::cosh(_kZeta*std::cos(phi - _phi0));
  }
  else if (_model == "dipole"){
    return angPart*std::pow( 1.0 + _kZeta*std::sin(phi-_phi0)*std::sin(phi-_phi0), -_alpha/2.0-1.0);
  }
  else if (_model == "boxcar"){
    return angPart*(1.0 - ((M_PI/(2.0*(1.0 + _kZeta)) < (std::fmod(phi,M_PI))) & (std::fmod(phi,M_PI) < (M_PI - M_PI/(2.0*(1.0 + _kZeta))))));
  }
  else {
    std::cerr << "Incorrect wandering magnetic field model in model_image_refractive scatter _P_phi::operator()" << std::endl;
    std::exit(1);
  }

}

void model_galactic_center_diffractive_scattering_screen::generate_model(std::vector<double> parameters)
{
  _model.generate_model(parameters);
}

void model_galactic_center_diffractive_scattering_screen::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}

double model_galactic_center_diffractive_scattering_screen::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
  double scatt_kernel = std::exp(-0.5*Dphi(d.u/(1+_M),d.v/(1+_M)));
  return scatt_kernel*_model.visibility_amplitude(d,acc);
  
}


};

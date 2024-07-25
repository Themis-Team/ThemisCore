/*!
    \file model_image_diffractive_scattering_screen.cpp  
    \author Avery Broderick
    \date June 2020
    \brief implementation file for a variable diffractive scattering screen interface, based on model_vsibility_galactic_center_diffractive_Scattering_screen.
    \details 
*/

#include "model_image_diffractive_scattering_screen.h"
#include <cmath>
#include "constants.h"


#define TABULATE_MAJMIN


namespace Themis {

model_image_diffractive_scattering_screen::model_image_diffractive_scattering_screen(model_image& model, 
										     double frequency,
										     std::string scattering_model)
  : _model(model), 
    _wavelength(constants::c/frequency), 
    _scattering_model(scattering_model),
    _screen_parameters(6,0.0),
    _kZetainterp_vonMises(utils::global_path("src/util/vonMises_kzeta_table.dat"), 2, 1),
    _kZetainterp_boxcar(utils::global_path("src/util/boxcar_kzeta_table.dat"), 2, 1),
    _P_phi_func("none", scattering_model, 1.38, 0.1413716694) // Initialize with GC defaults
{
  // Adding 6 scattering kernel parameters
  _size = _model.size()+6;

  // Inform
  std::cerr << "Scattering your visibility model with variable model\n";


  //Fills anisotropy and screen parameters for the angular distributions
  if ( scattering_model == "dipole" ){
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
    
    _kZetainterp_dipole.set_f(i2_zeta0,i2_scatt_alpha,i2_kZeta);
  }
  else if ( scattering_model!="vonMises" && scattering_model!="boxcar" ){
    std::cerr << "Scattering model angular distribution not recongnized! Use 'vonMises', 'dipole', or 'boxcar'\n";
    std::abort();
  }

#ifdef TABULATE_MAJMIN
  size_t Nnx = 32;
  std::vector<double> q1v(Nnx), Ppn(Nnx);
  std::valarray<double> q1(Nnx), q2(Nnx), int_min(Nnx*Nnx), int_maj(Nnx*Nnx);
  double imn, imj, pn;
  if (scattering_model=="vonMises" || scattering_model=="dipole")
  {
    for (size_t i=0; i<Nnx; ++i) {
      q1v[i] = q1[i] = -2 + 5.5*double(i)/double(Nnx-1); // log10(kzeta)
    }
    for (size_t i=0; i<Nnx; ++i)
      q2[i] = 2.0*(i+0.5)/double(Nnx); // alpha
    for (size_t j=0; j<Nnx; ++j)
      for (size_t i=0; i<Nnx; ++i)
      {
	generate_minmajnorm(scattering_model,std::pow(10.0,q1[i]),q2[j],0.0,imn,imj,pn);
	int_min[j+Nnx*i] = imn;
	int_maj[j+Nnx*i] = imj;
	Ppn[i] = pn;
	/*
	std::cout << std::setw(5) << i
		  << std::setw(5) << j
		  << std::setw(15) << q1[i]
		  << std::setw(15) << q2[j]
		  << std::setw(15) << imn
		  << std::setw(15) << imj
		  << std::setw(15) << pn
		  << std::endl;
	*/
      }
  }
  else if (scattering_model=="boxcar")
  {
    for (size_t i=0; i<Nnx; ++i) {
      q1v[i] = q1[i] = -2 + 5.5*double(i)/double(Nnx-1); // log10(kzeta)
    }
    for (size_t i=0; i<Nnx; ++i)
      q2[i] = 2.0*M_PI*double(i)/double(Nnx-1); // phi0
    for (size_t j=0; j<Nnx; ++j)
      for (size_t i=0; i<Nnx; ++i)
      {
	generate_minmajnorm(scattering_model,std::pow(10.0,q1[i]),0.0,q2[j],imn,imj,pn);
	int_min[j+Nnx*i] = imn;
	int_maj[j+Nnx*i] = imj;
	Ppn[i] = pn;
      }
  }
  else
  {
    std::cerr << "Scattering model angular distribution not recongnized! Use 'vonMises', 'dipole', or 'boxcar'\n";
    std::abort();
  }
  _int_min_table.set_f(q1,q2,int_min);
  _int_maj_table.set_f(q1,q2,int_maj);
  _Ppn_table.set_tables(q1v,Ppn);

  std::cerr << "Scattering minor/major axis tables set.\n";
#endif


  
  std::vector<double> scdef;
  scdef.push_back(1.38); // screen theta_maj
  scdef.push_back(0.703); // screen theta_min
  scdef.push_back(81.9*M_PI/180.0); // screen PA
  scdef.push_back(1.38); // screen alpha
  scdef.push_back(std::log10(800e5)); // Inner radius in km
  scdef.push_back((2.82*3.086e2)/(5.53*3.086e21)); // M
  generate_scattering_screen(scdef);    
}

double model_image_diffractive_scattering_screen::Dphi(double u, double v) const
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

double model_image_diffractive_scattering_screen::Q(double x, double y) const
{
  //2pi is because Q in Psaltis et. al. use wavenumber.
  double q = 2*M_PI*std::sqrt(x*x+y*y) + 1e-12/_rin;
  double phiq = std::atan2(y,x);
  double qrin = q*_rin;

  return _Qbar*std::pow(qrin, -_scatt_alpha - 2.0)*std::exp(-qrin*qrin)*_P_phi_func(phiq)/_P_phi_norm;
}


model_image_diffractive_scattering_screen::_P_phi::_P_phi(std::string axis, std::string model, double alpha, double phi0)
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


double model_image_diffractive_scattering_screen::_P_phi::operator()(double phi) const
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

void model_image_diffractive_scattering_screen::generate_minmajnorm(std::string scattering_model, double kZeta, double alpha, double phi0, double& int_min, double& int_maj, double& P_phi_norm)
{
  // Set particulars of integrand function
  _P_phi_func.set_alpha(alpha);
  _P_phi_func.set_phi0(phi0);

  //Create Gaussian quadrature object
  GaussianQuadrature Quad;
  
  //Find the correct normalization of the distribution
  _P_phi_func.set_kZeta(kZeta);
  P_phi_norm = Quad.integrate(_P_phi_func, 0, 2*M_PI);

  //Find the Bmaj integral
  _P_phi P_phi_maj("major", scattering_model, alpha, phi0);
  P_phi_maj.set_kZeta(kZeta);
  int_maj = Quad.integrate(P_phi_maj, 0, 2*M_PI)/P_phi_norm;

  //Find the Bmin integral
  _P_phi P_phi_min("minor", scattering_model, alpha, phi0);
  P_phi_min.set_kZeta(kZeta);
  int_min = Quad.integrate(P_phi_min, 0, 2*M_PI)/P_phi_norm;
}
  
void model_image_diffractive_scattering_screen::generate_scattering_screen(std::vector<double> screen_parameters)
{
  // Save the parameter list
  _screen_parameters = screen_parameters;

  // Assume useful names
  _theta_maj_mas_cm = screen_parameters[0];
  _theta_min_mas_cm = screen_parameters[1];
  _phi0 = 0.5*M_PI - screen_parameters[2];
  _scatt_alpha = screen_parameters[3];
  _rin = std::pow(10.0,screen_parameters[4]);
  _M = screen_parameters[5];

  double lambdabar_0 = 1.0/(2.0*M_PI);
  double gammaA2 = std::tgamma(1.0 - _scatt_alpha/2.0);
  double rinL2 = _rin*_rin/(lambdabar_0*lambdabar_0);
  _Qbar = M_PI*M_PI*(rinL2*(1.0+_M))*(rinL2*(1.0+_M))/(gammaA2*std::log(2))*
          (_theta_maj_mas_cm*_theta_maj_mas_cm + _theta_min_mas_cm*_theta_min_mas_cm);
  //convert Qbar to radians^2 from mas^2
  _Qbar *= 1.0/1000.0/3600.0*M_PI/180.0;
  _Qbar *= 1.0/1000.0/3600.0*M_PI/180.0;

  //asymmetry needed for kZeta
  double A = _theta_maj_mas_cm/_theta_min_mas_cm;

  //Find the scattering coefficent at reference wavelength of 1cm
  _C_scatt_0 = _Qbar*gammaA2/(8.0*M_PI*M_PI*rinL2);
  double B_prefac = std::pow(2.0,3.0 - _scatt_alpha)*std::sqrt(M_PI)/(_scatt_alpha*std::tgamma((_scatt_alpha+1.0)/2.0));
  //Scattering asymmetry which is needed for find kZeta for the different models

  //Measure of asymmetry needed to find kZeta
  _zeta0 = (A*A-1)/(A*A+1);

  //Fills anisotropy and screen parameters for the angular distributions
  if ( _scattering_model == "vonMises" )
    _kZeta = _kZetainterp_vonMises(_zeta0);
  else if ( _scattering_model == "dipole" )
    _kZetainterp_dipole.bicubic(_zeta0, _scatt_alpha, _kZeta);
  else if ( _scattering_model == "boxcar" )
    _kZeta = _kZetainterp_boxcar(_zeta0);


#ifdef TABULATE_MAJMIN
  double int_min=0.0, int_maj=0.0;
  double log10kZeta = std::log10(std::max(_kZeta,1.0e-2));
  if (_scattering_model=="vonMises" || _scattering_model=="dipole")
  {
    _P_phi_norm = _Ppn_table(log10kZeta);
    _int_min_table.bicubic(log10kZeta,_scatt_alpha,int_min);
    _int_maj_table.bicubic(log10kZeta,_scatt_alpha,int_maj);
  }
  else if (_scattering_model=="boxcar")
  {
    _P_phi_norm = _Ppn_table(log10kZeta);
    _int_min_table.bicubic(log10kZeta,_phi0,int_min);
    _int_maj_table.bicubic(log10kZeta,_phi0,int_maj);
  }
#else
  // Set particulars of integrand function
  _P_phi_func.set_alpha(_scatt_alpha);
  _P_phi_func.set_phi0(_phi0);

  //Create Gaussian quadrature object
  GaussianQuadrature Quad;

  //Find the correct normalization of the distribution
  _P_phi_func.set_kZeta(_kZeta);
  _P_phi_norm = Quad.integrate(_P_phi_func, 0, 2*M_PI);

  //Find the Bmaj integral
  _P_phi P_phi_maj("major", _scattering_model, _scatt_alpha, _phi0);
  P_phi_maj.set_kZeta(_kZeta);
  double int_maj = Quad.integrate(P_phi_maj, 0, 2*M_PI)/_P_phi_norm;

  //Find the Bmin integral
  _P_phi P_phi_min("minor", _scattering_model, _scatt_alpha, _phi0);
  P_phi_min.set_kZeta(_kZeta);
  double int_min = Quad.integrate(P_phi_min, 0, 2*M_PI)/_P_phi_norm;
#endif
  
  _Bmaj = B_prefac*int_maj/(1+_zeta0);
  _Bmin = B_prefac*int_min/(1-_zeta0);


  /*
  std::cout << "-----------------------------------------------------------" << std::endl;
  std::cout << "theta_maj_mas_cm : " << _theta_maj_mas_cm << std::endl;
  std::cout << "theta_min_mas_cm : " << _theta_min_mas_cm << std::endl;
  std::cout << "phi0 : " << _phi0 << std::endl;
  std::cout << "scatt_alpha : " << _scatt_alpha << std::endl;
  std::cout << "rin : " << _rin << std::endl;
  std::cout << "M : " << _M << std::endl;
    
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
  std::cout << "-----------------------------------------------------------" << std::endl;
  */
}
  
void model_image_diffractive_scattering_screen::generate_model(std::vector<double> parameters)
{
  // Save and strip off screen parameters
  std::vector<double> scparams(6,0.0);
  for (int j=5; j>=0; --j)
  {
    scparams[j] = parameters[parameters.size()-1];
    parameters.pop_back();
  }
  // If different than before, make the screen
  if (scparams!=_screen_parameters)
    generate_scattering_screen(scparams);

  // Generate model
  _model.generate_model(parameters);
}

void model_image_diffractive_scattering_screen::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}

std::complex<double> model_image_diffractive_scattering_screen::visibility(datum_visibility& d, double accuracy)
{
  std::complex<double> scatt_kernel(std::exp(-0.5*Dphi(d.u/(1+_M),d.v/(1+_M))),0.0);
  return scatt_kernel*_model.visibility(d,accuracy);
}

double model_image_diffractive_scattering_screen::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
{
  double scatt_kernel(std::exp(-0.5*Dphi(d.u/(1+_M),d.v/(1+_M))));
  return scatt_kernel*_model.visibility_amplitude(d,accuracy);
}

double model_image_diffractive_scattering_screen::closure_phase(datum_closure_phase& d, double accuracy)
{
  return _model.closure_phase(d,accuracy);
}

double model_image_diffractive_scattering_screen::closure_amplitude(datum_closure_amplitude& d, double accuracy)
{
  return _model.closure_amplitude(d,accuracy);
}



void model_image_diffractive_scattering_screen::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  std::cerr << "Generate image is not defined for model_image_diffractive_scattering_screen.\n";
  std::exit(1);
}

  
};

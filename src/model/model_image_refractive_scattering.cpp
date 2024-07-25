/*!
    \file model_image_refractive_scattering.cpp  
    \author Paul Tiede
    \date October 2018
    \brief implementation file for refractive scattering interface.
    \details 
*/

#include <cmath>
#include <sstream>

#include "model_image_refractive_scattering.h"
#include "constants.h"

namespace Themis {
model_image_refractive_scattering::model_image_refractive_scattering(model_image& model, size_t nModes,
                                  double tobs, double frequency, 
                                  std::string scattering_model,
                                  double observer_screen_distance, double source_screen_distance,
                                  double theta_maj_mas_cm, double theta_min_mas_cm, double POS_ANG, 
                                  double scatt_alpha, double rin, double rout,
                                  double vx_ss_kms, double vy_ss_kms)
:_model(&model), _nModes(nModes),_tobs(tobs),_frequency(frequency), _wavelength(constants::c/frequency), 
 _scattering_model(scattering_model), _observer_screen_distance(observer_screen_distance), 
 _source_screen_distance(source_screen_distance),   
 _theta_maj_mas_cm(theta_maj_mas_cm), _theta_min_mas_cm(theta_min_mas_cm),
 //_POS_ANG(-POS_ANG*M_PI/180.0),
 _phi0((90-POS_ANG)*M_PI/180.0),
  _M(_observer_screen_distance/_source_screen_distance),
 _rF(std::sqrt(_observer_screen_distance*_wavelength/(2*M_PI)/(1+_M))),
 _scatt_alpha(scatt_alpha), _rin(rin),
 //_rout(rout),
 _vx_ss_kms(vx_ss_kms), _vy_ss_kms(vy_ss_kms),
 _epsilon(_nModes, std::vector<std::complex<double> >(_nModes,std::complex<double>(0.0,0.0)) ),
 _nray(128), _fov(100*1e-6/3600*M_PI/180.0),
 _P_phi_func("none", scattering_model, scatt_alpha, _phi0)
{

  //set mpi communicator

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "Creating model_image_refractive_scattering in rank " << world_rank << std::endl;

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
  
  //std::cout << "P_phi_prefac : " << 1.0/_P_phi_norm << std::endl;
  //std::cout << "kZeta : " << _kZeta << std::endl;
  //std::cout << "Qbar : " << _Qbar << std::endl;
  //std::cout << "C_scatt_0 : "  << _C_scatt_0 << std::endl;
  //std::cout << "int_maj : "  << int_maj << std::endl;
  //std::cout << "int_min : "  << int_min << std::endl;
  //std::cout << "Bmaj_0 : " << _Bmaj*(1+_zeta0)/2.0*_C_scatt_0 << std::endl;
  //std::cout << "Bmin_0 : " << _Bmin*(1-_zeta0)/2.0*_C_scatt_0 << std::endl;
  //std::cout << "2/aBmaj : " << 2.0/(_scatt_alpha*_Bmaj) << std::endl;
  //std::cout << "2/aBmin : " << 2.0/(_scatt_alpha*_Bmin) << std::endl;


 

}

std::string model_image_refractive_scattering::model_tag() const
{
  std::stringstream tag;
  tag << "model_image_refractive_scattering" << "  MISSING DATA ABOUT SCREEN PROPERTIES, MUST BE FILLED IN\n";
  tag << "SUBTAG START\n";
  tag << _model->model_tag() << '\n';  
  tag << "SUBTAG FINISH";
  
  return tag.str();
}
  
void model_image_refractive_scattering::generate_model(std::vector<double> parameters)
{
  size_t n_model_params = parameters.size()-(_nModes*_nModes-1);
  std::vector<double> src_params(parameters.begin(), parameters.begin()+n_model_params); //Parameters for source images
  _position_angle = 0;
  // Check to see if these differ from last set used.
  if (_generated_model && parameters==_current_parameters)
    return;
  //Check to see if intrinsic model image hasn't changed
  else if (_generated_model && src_params==_current_model_params)
  {
    
    std::vector<double> screen_params(parameters.begin() + n_model_params, parameters.end()); //Parameters for scattering screen
   
    if (_nModes*_nModes-1 != screen_params.size()){
      std::cerr << "Number of modes for screen and screen parameter size not matching!\n";
      std::abort();
    }
    //Here we skip blurring the image again since the intrinsic source hasn't changed.    
    // Generate the image using the user-supplied routine
    generate_image(screen_params,_I,_alpha,_beta);
    _generated_visibilities = false;
  }
  else
  {
    _current_parameters = parameters;
    _current_model_params = src_params;
    //Create iterators for splitting the params
    std::vector<double> screen_params(parameters.begin() + n_model_params, parameters.end()); //Parameters for scattering screen
  
    if (_nModes*_nModes-1 != screen_params.size()){
      std::cerr << "Number of modes for screen and screen parameter size not matching!\n";
      std::abort();
    }
    //for ( int i = 0; i < src_params.size(); i++ ){
    //  std::cout << src_params[i] << std::endl;
    //}

    //Now generate src image and get model visibilities
    _model->generate_model(src_params);
    generate_model_visibilities();

    

    //First we blur the image to get the ensemble blurred image
    ensemble_blur_image();

    // Generate the image using the user-supplied routine
    generate_image(screen_params,_I,_alpha,_beta);
    
    // Set some boolean flags for what is and is not defined
    _generated_model = true;
    _generated_visibilities = false;
  }

}

void model_image_refractive_scattering::generate_model_visibilities()
{

  //First generate _alpha and _beta
  double d_alpha = _fov/_nray;
  double d_beta = _fov/_nray;
  
  //Resize array accordingly
  if (_alpha.size() != _nray || _beta.size() != _nray){
    _alpha.resize(_nray);
    _beta.resize(_nray);
    for ( size_t i = 0;  i < _nray; ++i){
      _alpha[i].resize(_nray);
      _beta[i].resize(_nray); 
    }
  }
  //Fill array for image
  for ( size_t i = 0; i < _nray; ++i)
    for ( size_t j = 0; j < _nray; ++j )
    {
      _alpha[i][j] = -_fov/2.0 + d_alpha*i + 0.5*d_alpha/2.0;
      _beta[i][j] = -_fov/2.0 + d_beta*j + 0.5*d_beta/2.0;
    }
  
  unsigned int Npad = 8; // Default padding factor of 8, MUST BE AN EVEN NUMBER

  
  //initialize
  size_t ncol, nrow;					//dimensions of the fourier transform map
  double ul_max, vl_max, d_ul, d_vl;	//vars for calculating u/lambda and v/lambda

  //determine number of columns and rows to use (Npad * origina dims)
  nrow = _nray*Npad;
  ncol = _nray*Npad;		//ensures that both column and row sizes are even numbers
  

  // Construct u/lambda and v/lambda
  ul_max= 1.0/(2.0*d_alpha);		        //this is our MAX frequency (based on 1/image size)
  vl_max= 1.0/(2.0*d_beta);			//note that this is in units of lambda
  d_ul = (ul_max*2.0)/((double)(nrow));	//now we determine the incremental value for the u and v
  d_vl = (vl_max*2.0)/((double)(ncol));	//conversion so we can make a meshgrid for the u/v plane

  //construct meshgrid for ul and vl
  if (_u.size()!=_v.size() || _v.size()!=nrow || _Vsrc.size() != nrow)
  {
    _u.resize(nrow);
    _v.resize(nrow);
    _Vsrc.resize(nrow);
  }
  for (size_t i=0; i<nrow; ++i)			//go through 2d map
  {
    if (_u[i].size()!=_v[i].size() || _v[i].size()!=ncol || _Vsrc[i].size() != ncol)
    {
      _u[i].resize(ncol);
      _v[i].resize(ncol);
      _Vsrc[i].resize(ncol);
    }
    for (size_t j=0; j<ncol; ++j)
    {
      //produce meshgrid recentering now not needed since I changed how the spacing is defined. (Paul Tiede)
      _u[i][j] = -ul_max + d_ul*((double)i);// - 0.5*d_ul;
      _v[i][j] = -vl_max + d_vl*((double)j);// - 0.5*d_vl;
      
      datum_visibility tmp(_u[i][j],_v[i][j],std::complex<double>(0.0,0.0),std::complex<double>(0.0,0.0),_frequency,_tobs,"","","");
      _Vsrc[i][j] = _model->visibility(tmp,0.0);
    }
  }


}

void model_image_refractive_scattering::set_mpi_communicator(MPI_Comm comm)
{
  _model->set_mpi_communicator(comm);
}

void model_image_refractive_scattering::set_screen_size(double fov)
{
  _fov = fov;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_refractive_scattering: Rank " << rank << " using screen size " << _fov*1e6*3600*180.0/M_PI << " uas" << std::endl;
}


void model_image_refractive_scattering::set_image_resolution(size_t nray)
{
  _nray = nray;
  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_refractive_scattering: Rank " << rank << " using image resolution " << _nray << std::endl;
}


void model_image_refractive_scattering::get_unscattered_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  if (!_generated_model){
    std::cerr << "model_image_refractive_scattering : Intrinsic model has not been generated yet arrays will be empty\n";
  }
  _model->get_image(alpha,beta,I);

}
void model_image_refractive_scattering::get_ensemble_average_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{

  if (!_generated_model){
    std::cerr << "model_image_refractive_scattering : Intrinsic model has not been generated yet arrays will be empty\n";
  }
  alpha = _alpha;
  beta = _beta;
  I = _Iea;
}
void model_image_refractive_scattering::get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  if (!_generated_model){
    std::cerr << "model_image_refractive_scattering : Intrinsic model has not been generated yet arrays will be empty\n";
  }
  alpha = _alpha;
  beta = _beta;
  I = _I;
}

void model_image_refractive_scattering::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{


  //Now we need to compute the phase screen
  compute_kphase_screen(parameters);

  //First resize I so it has the proper size
  
  size_t nrow = _nray;
  size_t ncol = _nray;
  if ( I.size() != nrow || alpha.size() != nrow || beta.size()!= nrow )
  {
    I.resize(nrow);
    for (size_t i = 0; i < _nray; ++i) 
      I[i].resize(ncol);
  }

  //Now we can refractive scatter the image using equation 9. of Johnson & Narayan 2016
  //Our image is stored in terms of angular coordinates in radians while the screen in physical distance to screen
  double rxMax = std::fabs(_alpha[0][0]*_observer_screen_distance);
  double ryMax = std::fabs(_beta[0][0]*_observer_screen_distance);
  for (size_t i = 0; i < nrow; ++i)
    for ( size_t j = 0; j < ncol; ++j )
    {
      double rx = _alpha[i][j]*_observer_screen_distance;
      double ry = _beta[i][j]*_observer_screen_distance;
      double gradxPhi, gradyPhi,Ia;
      _i2D_Dxphi.bicubic(rx,ry,gradxPhi);
      _i2D_Dyphi.bicubic(rx,ry,gradyPhi);
      //Now wrap since ehtim does it and the screen should wrap.
      double xx = std::fmod(rx + _rF*_rF*gradxPhi + rxMax, 2.0*rxMax);
      double yy = std::fmod(ry + _rF*_rF*gradyPhi + rxMax, 2.0*ryMax);
      if ( xx < 0 )
        xx += rxMax*2.0;
      if ( yy < 0 )
        yy += ryMax*2.0;
      _i2D_Iea.bicubic(xx-rxMax, yy-ryMax, Ia);

      /* 
      std::cout << std::setw(15) << rx
                << std::setw(15) << ry
                << std::setw(15) << gradxPhi
                << std::setw(15) << gradyPhi
                << std::setw(15) << Ia
                << std::setw(15) << _Iea[i][j] << std::endl;
      */
      I[i][j] = Ia; 
    }
  
}

double model_image_refractive_scattering::Dphi(double u, double v) const
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
  

void model_image_refractive_scattering::ensemble_blur_image()
{

  //First get the source visibilities
  std::vector<std::vector<std::complex<double> >  > Vea(_Vsrc.size(), std::vector<std::complex<double> > 
                                                                        (_Vsrc[0].size(), std::complex<double>(0.0,0.0)));
  //Apply the scattering kernel and unphase center the image so I can actually find it
  //

  for ( size_t i = 0; i <  Vea.size(); i++)
    for ( size_t j = 0; j < Vea[0].size(); j++ ){
      std::complex<double> scatt_kernel(std::exp(-0.5*Dphi(_u[i][j]/(1+_M),_v[i][j]/(1+_M))),0);
      //Now we need to dephase center and unflip the image
      Vea[i][j] = _Vsrc[i][j]*scatt_kernel*
                  std::exp(std::complex<double>(0,2.0*M_PI) * (_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );
      //std::cerr << _Vsrc[i][j] << std::endl;
;
      //std::cerr << scatt_kernel << std::endl;
    }


  
  //Now fft the visibilities to get the image
  std::vector<std::vector<double> > Ibig;
  Ibig = ifft_2d(ifft_shift(Vea));


  size_t nrow = _nray;
  size_t ncol = _nray;
  _Iea.resize(nrow);
  for (size_t i = 0; i < _Iea.size(); ++i)
    _Iea[i].resize(ncol);
  
  double norm = (_alpha[1][1]-_alpha[0][0])*(_beta[1][1]-_beta[0][0]);
  //This image has been padded so we need to first unpad it.
  //Copy the image but first flip it since we had flipped it earlier to match image on the earth
  for ( size_t i = 0; i < _nray; ++i)
    for ( size_t j = 0; j< _nray; ++j)
      _Iea[i][j] = Ibig[_nray-1-i][j]/norm;

  //Set the interpolation objects for when we have to shift it according to refractive scattering
  if (_i2drx.size()!=nrow)
    _i2drx.resize(nrow);
  if (_i2dry.size()!=ncol)
    _i2dry.resize(ncol);
  if (_i2dIea.size()!=nrow*ncol){
    _i2dIea.resize(nrow*ncol);
  }
  
  //Stor the ensemble blurred image in terms of the distance at scattering screen. 
  //This is to make it easier when I refractively scatter the image.
  for (size_t i=0; i<nrow; ++i)
    _i2drx[i] = _alpha[i][0]*_observer_screen_distance;
  for (size_t j=0; j<ncol; ++j)
    _i2dry[j] = _beta[0][j]*_observer_screen_distance;
  for (size_t j=0; j<ncol; ++j)
    for (size_t i=0; i<nrow; ++i)
      _i2dIea[j+ncol*i] = _Iea[i][j];	//magnitude

  _i2D_Iea.set_f(_i2drx,_i2dry,_i2dIea);

  
}


std::vector<std::vector<double> > model_image_refractive_scattering::ifft_2d(const std::vector<std::vector<std::complex<double> > > &V)
{
  fftw_complex *in, *out;
  fftw_plan p;
  unsigned int rows = V.size();		//number of rows
  unsigned int cols = V[0].size(); 	//number of cols
  unsigned int numelements = rows*cols;
  
  std::vector<double> tmp_out;
  std::vector<std::vector<double > > out_c;		//input (complex var)
  
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  
   
  //now we do the same thing but we copy over the values from our complex vector
  for (unsigned int i=0; i<rows; i++)
  {
    for (unsigned int j=0; j<cols; j++)
    {
      in[i*cols + j][0] = V[i][j].real();	//copy it over to our fftw storage
      in[i*cols + j][1] = V[i][j].imag();      //complex conjugate the signal since we are performing an IDFT
    }
  }

  fftw_execute(p); /* repeat as needed */
  
  out_c.clear();
  for (unsigned int i=0; i<rows; i++)
  {
    tmp_out.clear();
    for (unsigned int j=0; j<cols; j++)
    {
      //divide by N because of fftw convention
      tmp_out.push_back(out[i*cols + j][0]/numelements);
    }
    out_c.push_back(tmp_out);
  }

  //memory cleanup
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
  
  return out_c;
}

std::vector<std::vector<std::complex<double> > > model_image_refractive_scattering::ifft_shift(const std::vector<std::vector<std::complex<double> > > &V)
{

  std::vector<std::vector<std::complex<double> > > out_c = V;

  std::rotate(out_c.begin(), out_c.begin() + (out_c.size())/2, out_c.end());
  for ( size_t i = 0; i < out_c.size(); i++ ){
    std::rotate(out_c[i].begin(), out_c[i].begin() + (out_c[i].size())/2, out_c[i].end());
  }

  return out_c;
}


model_image_refractive_scattering::_P_phi::_P_phi(std::string axis, std::string model, double alpha, double phi0)
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


double model_image_refractive_scattering::_P_phi::operator()(double phi) const
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

void model_image_refractive_scattering::make_epsilon_screen(std::vector<double> screen_params)
{   
  //First I need to populate the scattering screen matrix. The screen is NxN-1 complex modes (DC component vanishes)
  //with conjugation symmetry to ensure it's Fourier transform is real. Depending on whether the screen
  //is even or odd this will change how I populate it.
  const int nModes = _nModes;
  if (screen_params.size() != size_t(nModes*nModes)-1){
    std::cerr << "The number independent parameters for a NxN mode screen must be equal to N^2-1\n";
    std::exit(1);
  }

  //epsilon screen. This is a complex random Gaussian with unit variance.
  
  //How the array is filled changes depending on whether it is even or odd due to the real parity of the screen
  //For N even the array has three purely real elements (note we set the dc component to be zero following ehtim convention)
  if (nModes % 2 == 0 ){
    //First grab the last three elements of the array. These form the real parts.
    _epsilon[0][0] = 0;
    _epsilon[nModes/2][nModes/2] = screen_params[screen_params.size()-1];
    _epsilon[nModes/2][0] = screen_params[screen_params.size()-2];
    _epsilon[0][nModes/2] = screen_params[screen_params.size()-3];

    //Alright now we just need to fill the rest of the modes with the proper symmetry
    int nReal = (nModes*nModes-4)/2; //-4 because we took care of the purely real cases already
    int s = 0;
    for ( int i = 1; i < (nModes+1)/2; ++i){
      //This points are purely real and already filled so skip
      if ( i == nModes/2 )
        continue;
      std::complex<double> ze(screen_params[s], screen_params[s+nReal]);
      _epsilon[i][0] = ze;
      _epsilon[nModes-i][0] = std::conj(ze);

      s+=1;
    }

    //Now fill in the rest and break once we hit the mid point
    for ( int j = 1; j < (nModes+2)/2; ++j )
      for ( int i = 0; i < nModes; ++i ){ 
        //Skip the purely real elements
        //std::cerr << std::setw(15) << i
        //          << std::setw(15) << j << std::endl;
        if ( i==0 && j==nModes/2)
          continue;
        if ( i==nModes/2 && j==nModes/2)
          break; //hit mid point so leave the loop
        
        std::complex<double> ze(screen_params[s], screen_params[s+nReal]);
        _epsilon[i][j] = ze;
        
        //Now for conjugation we have to be a bit careful since we can run past the table
        int i2 = nModes-i;
        if (i2 == nModes){
          i2 = 0;
        }
        int j2 = nModes-j;
        if (j2 == nModes){
          j2 = 0;
        }
        _epsilon[i2][j2] = std::conj(ze);
        s++;

      }
      
  }
  else{ //Odd number of modes
    //For this one I will follow the convention in ehtim to ensure easy comparison.
    int s = 0;
    int nReal = (nModes*nModes - 1)/2;
    for ( int i = 1; i < (nModes+1)/2; ++i){
      std::complex<double> ze(screen_params[s], screen_params[s+nReal]);
      _epsilon[i][0] = ze;
      _epsilon[nModes-i][0] = std::conj(ze);
      s++;
      //std::cerr << s << std::endl;
    }
    //Now fill the next N rows
    for ( int j = 1; j < (nModes+1)/2; ++j )
      for ( int i = 0; i < nModes; ++i ){
        std::complex<double> ze(screen_params[s], screen_params[s+nReal]);
        _epsilon[i][j] = ze;

        //Now for conjugation we have to be a bit careful since we can run past the table
        int i2 = nModes-i;
        if (i2 == nModes){
          i2 = 0;
        }
        int j2 = nModes-j;
        if (j2 == nModes){
          j2 = 0;
        }
        _epsilon[i2][j2] = std::conj(ze);
        s++;
      }
  }

}

double model_image_refractive_scattering::Q(double x, double y) const
{
  //2pi is because Q in Psaltis et. al. use wavenumber.
  double q = 2*M_PI*std::sqrt(x*x+y*y) + 1e-12/_rin;
  double phiq = std::atan2(y,x);
  double qrin = q*_rin;

  return _Qbar*std::pow(qrin, -_scatt_alpha - 2.0)*std::exp(-qrin*qrin)*_P_phi_func(phiq)/_P_phi_norm;
}

void model_image_refractive_scattering::compute_kphase_screen(std::vector<double> screen_params)
{
  //First generate the epsilon screen
  make_epsilon_screen(screen_params);

  double lambdabar = _wavelength/(2*M_PI);
  
  //FOV for scattering screen in cm (since alpha is in radians) namely, calculations are done at the scattering screen
  double fovx= _fov*_observer_screen_distance;
  double fovy= _fov*_observer_screen_distance;

  //Important for long term movies where the screen moves past the image.
  double delta_screen_x_px = _vx_ss_kms*1e5*((_tobs))/(fovx/_nModes);
  double delta_screen_y_px = _vy_ss_kms*1e5*((_tobs))/(fovy/_nModes);


  //Arrays for fourier derivatives of the scattering screen
  std::vector<std::vector<std::complex<double> > > kxPhase(_nModes, std::vector<std::complex<double> >(_nModes,std::complex<double>(0.0,0.0)));
  std::vector<std::vector<std::complex<double> > > kyPhase(_nModes, std::vector<std::complex<double> >(_nModes,std::complex<double>(0.0,0.0)));
  std::complex<double> ii(0.0,1.0);
  std::vector<std::vector<std::complex<double> > > u(_nModes,std::vector<std::complex<double> >(_nModes, std::complex<double>(0.0,0.0))), v(_nModes,std::vector<std::complex<double> >(_nModes,std::complex<double>(0.0,0.0)));
  
  std::vector<std::vector<std::complex<double> > > epsilon_shift = _epsilon;
  //FFT shift the screen since it is currently using standard FFTW convention.
  std::rotate(epsilon_shift.begin(), epsilon_shift.begin() + (epsilon_shift.size()+1)/2, epsilon_shift.end());
  for ( size_t i = 0; i < epsilon_shift.size(); i++ ){
    std::rotate(epsilon_shift[i].begin(), epsilon_shift[i].begin() + (epsilon_shift[i].size()+1)/2, epsilon_shift[i].end());
  }
  //std::cerr << "Qbar : " << _Qbar << std::endl; 
  //Now construct the phase screen in Fourier space
  for ( size_t i = 0; i < _nModes; ++i )
    for ( size_t j = 0; j < _nModes; ++j )
    {

      //Construct the screen spacing
      double ru = double(i - std::floor(_nModes/2.0) )/fovx;
      double rv = double(j - std::floor(_nModes/2.0) )/fovy; 
      std::complex<double> phaseK = std::sqrt(Q(ru,rv))*epsilon_shift[i][j]*std::exp(2*M_PI*ii*(i*delta_screen_x_px + j*delta_screen_y_px)/double(_nModes));
      /*
      std::cerr << std::setw(15) << ru*M_PI*2.0
                << std::setw(15) << rv*M_PI*2.0
                << std::setw(15) << Q(ru,rv)
                << std::setw(15) << _P_phi_func(std::atan2(rv,ru))/_P_phi_norm << std::endl;
      */
      kxPhase[i][j] = 2*M_PI*ii*ru*phaseK;
      kyPhase[i][j] = 2*M_PI*ii*rv*phaseK;

    }



  //Now Fourier transform the screen
  std::vector<std::vector<double> > dphidx = ifft_2d(ifft_shift(kxPhase));
  std::vector<std::vector<double> > dphidy = ifft_2d(ifft_shift(kyPhase));


  
  //Now create gradient interpolator object
  double dx = fovx/_nModes;
  double dy = fovy/_nModes;
  /*
  //Screen testing
  std::ofstream outp("phase_test");
    for ( size_t j = 0; j < _nModes; ++j )
      for ( size_t i = 0; i < _nModes; ++i )
      {
        //Construct the screen spacing
        outp << std::setw(15) << (_alpha[0][0]*_observer_screen_distance + dx*i)/_observer_screen_distance
             << std::setw(15) << (_beta[0][0]*_observer_screen_distance + dy*j)/_observer_screen_distance
             << std::setw(15) << dphidx[i][j]*lambdabar/fovx*_nModes*_nModes
             << std::setw(15) << dphidy[i][j]*lambdabar/fovx*_nModes*_nModes  << std::endl;
      }
    outp.close();
  */
  
  if ( _i2drxK.size() != _nModes){
    _i2drxK.resize(_nModes);
    _i2dryK.resize(_nModes);
    _i2dDxPhi.resize(_nModes*_nModes);
    _i2dDyPhi.resize(_nModes*_nModes);
     
  }
  //Note I am recording the image at the scattering screen
  for (size_t i=0; i<_nModes; ++i)
    _i2drxK[i] = -fovx/2.0 + dx*i+0.5*dx;
  for (size_t j=0; j<_nModes; ++j)
    _i2dryK[j] = -fovy/2.0 + dy*j+0.5*dy;
  for (size_t j=0; j<_nModes; ++j)
    for (size_t i=0; i<_nModes; ++i)
    {
      _i2dDxPhi[j+_nModes*i] = dphidx[i][j]*lambdabar/fovx*_nModes*_nModes;
      _i2dDyPhi[j+_nModes*i] = dphidy[i][j]*lambdabar/fovy*_nModes*_nModes;
    }

  _i2D_Dxphi.set_f(_i2drxK,_i2dryK,_i2dDxPhi);
  _i2D_Dyphi.set_f(_i2drxK,_i2dryK,_i2dDyPhi);
}


};

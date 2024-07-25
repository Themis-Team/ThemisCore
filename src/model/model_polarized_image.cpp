/*!
  \file model_polarized_image.cpp
  \author Avery E. Broderick
  \date  March, 2020
  \brief Implements polarized image model class.
  \details To be added
*/

#include "model_polarized_image.h"
#include <cmath>
#include <valarray>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <fstream>

#include <ctime>

namespace Themis {

model_polarized_image::model_polarized_image()
  : _comm(MPI_COMM_WORLD), _generated_model(false), _generated_visibilities(false), _use_spline(false), _position_angle(0.0), _size(1), _modeling_Dterms(false)
{
}

model_polarized_image::~model_polarized_image()
{
}


void model_polarized_image::write_model_tag_file(std::string tagfilename) const
{
  std::ofstream tagout(tagfilename);
  write_model_tag_file(tagout);
}

void model_polarized_image::write_model_tag_file(std::ofstream& tagout) const
{
  tagout << "tagvers-1.0\n" << model_tag() << std::endl;
}
  
void model_polarized_image::model_Dterms(std::vector<std::string> station_codes)
{
  if (_modeling_Dterms)
  {
    std::cerr << "ERROR: model_polarized_image::model_Dterms : D-terms already being modeled!\n";
    std::exit(1);
  }

  _station_codes = station_codes;
  
  // Sort the station codes in ascending order (will be done lexicographically)
  // Because this is a startup cost, dumb bubble sort.
  std::string tmp;
  for (size_t j=0; j<_station_codes.size(); ++j)
    for (size_t k=j; k<_station_codes.size(); ++k)
      if (_station_codes[k]<_station_codes[j])
      {
	tmp = _station_codes[j];
	_station_codes[j] = _station_codes[k];
	_station_codes[k] = tmp;
      }

  // Check for repeats and quit if found
  for (size_t j=1; j<_station_codes.size(); ++j)
    if (_station_codes[j]==_station_codes[j-1])
    {
      std::cerr << "ERROR: model_polarized_image::model_Dterms : Repeated station code found : " << _station_codes[j] << '\n';
      std::exit(1);
    }

  // Generate the hash table to get back to the original ordering.
  for (size_t j=0; j<_station_codes.size(); ++j)
    _station_code_index_hash_table.push_back(get_index_from_station_code(station_codes[j]));
  
  // Increase size
  _size += 4*_station_codes.size();
  
  // Set initial Dterms vector to zero
  _Dterms.resize(2*_station_codes.size(),std::complex<double>(0.0,0.0));

  _modeling_Dterms=true;
}

void model_polarized_image::generate_model(std::vector<double> parameters)
{
  // Read and strip off Dterm parameters
  read_and_strip_Dterm_parameters(parameters);

  // Assumes the last parameter is the position angle, saves it and strips it off
  _position_angle = parameters.back();
  parameters.pop_back();
  
  // Check to see if these differ from last set used.
  if (_generated_model && parameters==_current_parameters)
    return;
  else
  {
    _current_parameters = parameters;
    
    // Generate the image using the user-supplied routine
    generate_polarized_image(parameters,_I,_Q,_U,_V,_alpha,_beta);
    
    // Set some boolean flags for what is and is not defined
    _generated_model = true;
    _generated_visibilities = false;
  }
}

void model_polarized_image::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  std::vector< std::vector<double> > Q=I;
  std::vector< std::vector<double> > U=I;
  std::vector< std::vector<double> > V=I;
  generate_polarized_image(parameters,I,Q,U,V,alpha,beta);
}


void model_polarized_image::generate_complex_visibilities()
{
  if (_generated_model){
    if (!_generated_visibilities){
      compute_raw_visibilities();
      _generated_visibilities=true;
    }
  }
  else{
    std::cerr << "model_polarized_image::generate_complex_visibilities: Must generate model\n"
	      << "  prior to requesting visibilities.\n";
    std::exit(1);
  }
}


std::vector< std::complex<double> > model_polarized_image::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
{
  if (_generated_model){
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else{
    std::cerr << "model_polarized_image::crosshand_visibilities: Must generate model\n"
	      << "  prior to requesting visibilities.\n";
    std::exit(1);
  }
  
  // Obtain counter-rotated (after reflection on the sky -- position angle is E of N) u,v coordinates
  double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
  double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

  // Perform interpolation
  std::complex<double> VI, VQ, VU, VV;
  double Vr, Vi;
  if ( _use_spline)
  {
    _i2D_VIr.bicubic_spline(ru,rv,Vr);
    _i2D_VIi.bicubic_spline(ru,rv,Vi);
    VI = std::complex<double>(Vr,Vi);

    _i2D_VQr.bicubic_spline(ru,rv,Vr);
    _i2D_VQi.bicubic_spline(ru,rv,Vi);
    VQ = std::complex<double>(Vr,Vi);

    _i2D_VUr.bicubic_spline(ru,rv,Vr);
    _i2D_VUi.bicubic_spline(ru,rv,Vi);
    VU = std::complex<double>(Vr,Vi);

    _i2D_VVr.bicubic_spline(ru,rv,Vr);
    _i2D_VVi.bicubic_spline(ru,rv,Vi);
    VV = std::complex<double>(Vr,Vi);
  }
  else
  {
    _i2D_VIr.bicubic(ru,rv,Vr);
    _i2D_VIi.bicubic(ru,rv,Vi);
    VI = std::complex<double>(Vr,Vi);

    _i2D_VQr.bicubic(ru,rv,Vr);
    _i2D_VQi.bicubic(ru,rv,Vi);
    VQ = std::complex<double>(Vr,Vi);

    _i2D_VUr.bicubic(ru,rv,Vr);
    _i2D_VUi.bicubic(ru,rv,Vi);
    VU = std::complex<double>(Vr,Vi);

    _i2D_VVr.bicubic(ru,rv,Vr);
    _i2D_VVi.bicubic(ru,rv,Vi);
    VV = std::complex<double>(Vr,Vi);
  }

  // Convert to RR, LL, RL, LR
  std::vector< std::complex<double> > crosshand_vector(4);
  crosshand_vector[0] = VI+VV; // RR
  crosshand_vector[1] = VI-VV; // LL 
  crosshand_vector[2] = VQ+std::complex<double>(0.0,1.0)*VU; // RL
  crosshand_vector[3] = VQ-std::complex<double>(0.0,1.0)*VU; // LR

  // Apply Dterms
  apply_Dterms(d,crosshand_vector);

  return crosshand_vector;
}

void model_polarized_image::apply_Dterms(const datum_crosshand_visibilities& d, std::vector< std::complex<double> >& crosshand_vector) const
{
  // Apply Dterms if being modeled
  if (_modeling_Dterms)
  {
    // Get the relevant station indexes
    size_t is1 = get_index_from_station_code(d.Station1);
    size_t is2 = get_index_from_station_code(d.Station2);

    // Field rotation angle correction term (note that data file is pre-corrected)
    std::complex<double> ei2p1 = std::exp(std::complex<double>(0.0,2.0)*d.phi1);
    std::complex<double> ei2p2 = std::exp(std::complex<double>(0.0,2.0)*d.phi2);
    
    // Get the relevant Dterms
    std::complex<double> DR1 = _Dterms[2*is1]*ei2p1;
    std::complex<double> DL1 = _Dterms[2*is1+1]*std::conj(ei2p1);
    std::complex<double> DR2c = std::conj(_Dterms[2*is2]*ei2p2);
    std::complex<double> DL2c = std::conj(_Dterms[2*is2+1]*std::conj(ei2p2));

    // Construct the new values
    std::vector< std::complex<double> > cvo = crosshand_vector;
    crosshand_vector[0] = cvo[0] + DR1*DR2c*cvo[1] + DR2c*cvo[2] + DR1*cvo[3];
    crosshand_vector[1] = DL1*DL2c*cvo[0] + cvo[1] + DL1*cvo[2] + DL2c*cvo[3];
    crosshand_vector[2] = DL2c*cvo[0] + DR1*cvo[1] + cvo[2] + DR1*DL2c*cvo[3];
    crosshand_vector[3] = DL1*cvo[0] + DR2c*cvo[1] + DL1*DR2c*cvo[2] + cvo[3];
  }
}

void model_polarized_image::read_and_strip_Dterm_parameters(std::vector<double>& parameters)
{
  // If modeling Dterms:
  if (_modeling_Dterms)
  {
    // Read off the last 4*_Dterms.size() parameters corresponding to the real and
    // imaginary components of the R and L Dterms.
    std::vector< std::complex<double> > Dterms(2*_Dterms.size());
    for (size_t j=0, k=parameters.size()-2*_Dterms.size(); j<_Dterms.size(); j++, k+=2)
      Dterms[j] = std::complex<double>(parameters[k],parameters[k+1]);
    parameters.erase(parameters.end()-2*_Dterms.size(),parameters.end());

    // Now save into ordered Dterm vector
    for (size_t j=0; j<_Dterms.size()/2; ++j)
    {
      _Dterms[2*_station_code_index_hash_table[j]] = Dterms[2*j];
      _Dterms[2*_station_code_index_hash_table[j]+1] = Dterms[2*j+1];
    }
  }
}


double model_polarized_image::polarization_fraction(datum_polarization_fraction& d, double accuracy)
{
  // WITHOUT THE FIELD ROTATION ANGLE CORRECTION
  datum_crosshand_visibilities dtmp(d.u,d.v,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
  std::vector< std::complex<double> > cv = crosshand_visibilities(dtmp,accuracy);
  
  // NOT QUITE THE RIGHT DEFINITION, REALLY [(Q^2+U^2)/(I^2-V^2)]^(1/2).
  // BUT DEFINITELY MORE SENSIBLE AS A CLOSURE QUANTITY.
  return ( std::sqrt(std::abs((cv[1]*cv[2])/(cv[0]*cv[3]))) );
}


std::complex<double> model_polarized_image::visibility(datum_visibility& d, double)
{
  if (_generated_model){
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else{
    std::cerr << "model_polarized_image::visibility: Must generate model\n"
	      << "  prior to requesting visibility.\n";
    std::exit(1);
  }
  
  // Obtain counter-rotated (after reflection on the sky -- position angle is E of N) u,v coordinates
  double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
  double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

  // Perform interpolation
  double Vr,Vi;
  if ( _use_spline)
  {
    _i2D_VIr.bicubic_spline(ru,rv,Vr);
    _i2D_VIi.bicubic_spline(ru,rv,Vi);
  }
  else
  {
    _i2D_VIr.bicubic(ru,rv,Vr);
    _i2D_VIi.bicubic(ru,rv,Vi);  
  }
  std::complex<double> VI = std::complex<double>(Vr,Vi);
  return (VI);
}

// FOR NOW JUST IMPLEMENT FOR STOKES I, SHOULD HAVE THE ABILITY TO DO THIS FOR OTHERS
double model_polarized_image::visibility_amplitude(datum_visibility_amplitude& d, double)
{
  if (_generated_model){
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else{
    std::cerr << "model_polarized_image::visibility_amplitude: Must generate model\n"
	      << "  prior to requesting visibility_amplitudes.\n";
    std::exit(1);
  }
  
  // Obtain counter-rotated (after reflection on the sky -- position angle is E of N) u,v coordinates
  double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
  double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

  // Perform interpolation
  double Vr,Vi;
  if ( _use_spline)
  {
    _i2D_VIr.bicubic_spline(ru,rv,Vr);
    _i2D_VIi.bicubic_spline(ru,rv,Vi);
  }
  else
  {
    _i2D_VIr.bicubic(ru,rv,Vr);
    _i2D_VIi.bicubic(ru,rv,Vi);  
  }
  std::complex<double> VI = std::complex<double>(Vr,Vi);
  return (std::abs(VI));
}

// FOR NOW JUST IMPLEMENT FOR STOKES I, SHOULD HAVE THE ABILITY TO DO THIS FOR OTHERS
double model_polarized_image::closure_phase(datum_closure_phase& d, double)
{
  if (_generated_model)
  {
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else
  {
    std::cerr << "model_polarized_image::closure_phase: Must generate model\n"
	      << "  prior to requesting closure phases.\n";
    std::exit(1);
  }

  // Obtain counter-rotated u,v coordinates
  double u[3]={d.u1,d.u2,d.u3}, v[3]={d.v1,d.v2,d.v3};
  std::complex<double> V[3];
  double ru, rv, Vr, Vi;
  //double c=std::cos(-_position_angle), s=std::sin(-_position_angle);
  double c=std::cos(_position_angle), s=std::sin(_position_angle);
  for (int j=0; j<3; ++j)
  {
    ru = u[j]*c + v[j]*s;
    rv = -u[j]*s + v[j]*c;
    if ( _use_spline)
    {
      _i2D_VIr.bicubic_spline(ru,rv,Vr);
      _i2D_VIi.bicubic_spline(ru,rv,Vi);
    }
    else
    {
      _i2D_VIr.bicubic(ru,rv,Vr);
      _i2D_VIi.bicubic(ru,rv,Vi);  
    }
    V[j] = std::complex<double>(Vr,Vi);
  }
  std::complex<double> V123 = V[0]*V[1]*V[2];
  
  return ( std::imag(std::log(V123))*180.0/M_PI );
}

// FOR NOW JUST IMPLEMENT FOR STOKES I, SHOULD HAVE THE ABILITY TO DO THIS FOR OTHERS
double model_polarized_image::closure_amplitude(datum_closure_amplitude& d, double)
{
  if (_generated_model)
  {
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else
  {
    std::cerr << "model_polarized_image::closure_amplitude: Must generate model\n"
	      << "  prior to requesting visibility_amplitudes.\n";
    std::exit(1);
  }

  // Obtain counter-rotated u,v coordinates
  double u[]={d.u1,d.u2,d.u3,d.u4}, v[]={d.v1,d.v2,d.v3,d.v4};
  double ru, rv, Vr, Vi, VM[4];
  //double c=std::cos(-_position_angle), s=std::sin(-_position_angle);
  double c=std::cos(_position_angle), s=std::sin(_position_angle);
  for (int j=0; j<4; ++j)
  {
    ru = u[j]*c + v[j]*s;
    rv = -u[j]*s + v[j]*c;

    if ( _use_spline)
    {
      _i2D_VIr.bicubic_spline(ru,rv,Vr);
      _i2D_VIi.bicubic_spline(ru,rv,Vi);
    }
    else
    {
      _i2D_VIr.bicubic(ru,rv,Vr);
      _i2D_VIi.bicubic(ru,rv,Vi);  
    }
    VM[j] = std::abs(std::complex<double>(Vr,Vi));
  }

  return ( (VM[0]*VM[2])/ (VM[1]*VM[3]) );
}


void model_polarized_image::output_image(std::string fname, bool rotate)
{
  //First check that image is actually filled
  if (_I.size()==0)
  {
    std::cerr << "model_polarized_image::output_image : Intensity grid is empty are you sure you generated the model/image?\n";
    std::exit(1);
  }

  std::vector<std::vector<double> > Irot(_alpha.size(), std::vector<double> (_alpha[0].size(),0.0));
  std::vector<std::vector<double> > Qrot(_alpha.size(), std::vector<double> (_alpha[0].size(),0.0));
  std::vector<std::vector<double> > Urot(_alpha.size(), std::vector<double> (_alpha[0].size(),0.0));
  std::vector<std::vector<double> > Vrot(_alpha.size(), std::vector<double> (_alpha[0].size(),0.0));

  //Create spline object for rotation
  std::valarray<double> idx(_alpha.size());
  std::valarray<double> idy(_alpha[0].size());
  std::valarray<double> iIrot(_alpha.size()*_alpha[0].size());
  std::valarray<double> iQrot(_alpha.size()*_alpha[0].size());
  std::valarray<double> iUrot(_alpha.size()*_alpha[0].size());
  std::valarray<double> iVrot(_alpha.size()*_alpha[0].size());
  for ( size_t ii = 0; ii < _alpha.size(); ++ii )
    idx[ii] = _alpha[ii][0];
  for ( size_t jj = 0; jj < _alpha[0].size(); ++jj )
    idy[jj] = _beta[0][jj];
  for ( size_t jj = 0; jj < _alpha[0].size(); ++jj )
    for ( size_t ii = 0; ii < _alpha.size(); ++ii )
    {
      iIrot[jj+_alpha.size()*ii] = _I[ii][jj];
      iQrot[jj+_alpha.size()*ii] = _Q[ii][jj];
      iUrot[jj+_alpha.size()*ii] = _U[ii][jj];
      iVrot[jj+_alpha.size()*ii] = _V[ii][jj];
    }
  Interpolator2D I_interp, Q_interp, U_interp, V_interp;
  I_interp.set_f(idx,idy,iIrot);
  Q_interp.set_f(idx,idy,iQrot);
  U_interp.set_f(idx,idy,iUrot);
  V_interp.set_f(idx,idy,iVrot);
  if (rotate)
  {
    for ( size_t ii = 0; ii < _alpha.size(); ++ii)
      for ( size_t jj = 0; jj < _alpha[0].size(); ++jj)
      {
        double x = std::cos(_position_angle)*_alpha[ii][jj] + std::sin(_position_angle)*_beta[ii][jj];
        double y = -std::sin(_position_angle)*_alpha[ii][jj] + std::cos(_position_angle)*_beta[ii][jj];
        I_interp.bicubic(x,y,Irot[ii][jj]);
        Q_interp.bicubic(x,y,Qrot[ii][jj]);
        U_interp.bicubic(x,y,Urot[ii][jj]);
        V_interp.bicubic(x,y,Vrot[ii][jj]);
      }
  }
  else
  {
    for ( size_t ii = 0; ii < _alpha.size(); ++ii)
      for ( size_t jj = 0; jj < _alpha[0].size(); ++jj)
      {
        Irot[ii][jj] = _I[ii][jj];
        Qrot[ii][jj] = _Q[ii][jj];
        Urot[ii][jj] = _U[ii][jj];
        Vrot[ii][jj] = _V[ii][jj];
      }
  }
  
  std::cout << "Outputting image to " << fname << std::endl;
  std::ofstream imout(fname.c_str());
  imout << "Nx:"
        << std::setw(15) << _alpha[0][0]
        << std::setw(15) << _alpha[_alpha.size()-1].back()
        << std::setw(15) << _alpha.size() 
        << std::endl
        << "Ny:"
        << std::setw(15) << _beta[0][0]
        << std::setw(15) << _beta[_beta.size()-1].back()
        << std::setw(15) << _alpha[0].size()
        << std::endl
        << std::setw(5) << "i"
        << std::setw(5) << "j"
        << std::setw(15) << "I (Jy/px)"
        << std::setw(15) << "Q (Jy/px)"
        << std::setw(15) << "U (Jy/px)"
        << std::setw(15) << "V (Jy/px)"
        << std::endl;
  double psize_x = _alpha[1][1] - _alpha[0][0];
  double psize_y = _beta[1][1] - _beta[0][0];
  for ( size_t ix=0; ix<_alpha[0].size(); ix++ )
    for (size_t iy=0; iy<_alpha.size(); iy++)
      imout << std::setw(5) << iy
            << std::setw(5) << ix
            << std::setw(15) << Irot[iy][ix]*psize_x*psize_y
            << std::setw(15) << Qrot[iy][ix]*psize_x*psize_y
            << std::setw(15) << Urot[iy][ix]*psize_x*psize_y
            << std::setw(15) << Vrot[iy][ix]*psize_x*psize_y 
	    << std::endl;
  imout.close();
  
}


  
void model_polarized_image::get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  alpha = _alpha;
  beta = _beta;
  I = _I;
}

void model_polarized_image::get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V) const
{
  alpha = _alpha;
  beta = _beta;
  I = _I;
  I = _Q;
  I = _U;
  I = _V;
}

void model_polarized_image::get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& V) const 
{

  if (!_generated_visibilities){
    std::cerr << "Visibilities not generated yet arrays will be empty\n";
  }
  u = _u;
  v = _v;
  V = _VI;
}
void model_polarized_image::get_visibility_amplitudes(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<double> >& V) const
{
  if (!_generated_visibilities){
    std::cerr << "Visibilities not generated yet arrays will be empty\n";
  }
  u = _u;
  v = _v;
  V.resize(_VI.size());
  for (size_t i=0; i<_VI.size(); ++i)
  {
    V.resize(_VI[i].size());
    for (size_t j=0; j<_VI[i].size(); ++j)
      V[i][j] = std::abs(_VI[i][j]);
  }
}

void model_polarized_image::use_spline_interp( bool use_spline )
{
  _use_spline = use_spline;
}


void model_polarized_image::compute_raw_visibilities()
{
  //time_t start_time;
  //std::time(&start_time);
  //std::cout << "Started in model_polarized_image::compute_raw_visibilities()" << std::endl;

  // Computes the visibilities (magnitude and phase) given the angular
  // distribution on the sky (alpha,beta) of the Intensity.  The units of
  // the visibilities are set by the input units of I.  It is assummed that
  // alpha & beta have the normal meshgrid structure.  u & v are returned
  // as u/lambda and v/lambda.  If Npad is given, the I-map is padded so that
  // it's dimensions are Npad times the original dimensions.  By default, Npad=2.
  // Generally, to access very short baselines, Npad must be 4 or greater.
  //
  // Currently a bit of a memory hog in that the fully padded V's are kept, 
  // roughly requiring Npad^2 as much space as required.  This could be ameliorated
  // at the expense of some simplicity.

  unsigned int Npad = 8; // Default padding factor of 8, MUST BE AN EVEN NUMBER

  
  //initialize
  size_t ncol, nrow; // dimensions of the fourier transform map
  std::vector<std::vector<double> > padded_I, padded_Q, padded_U, padded_V; // padded image
  std::vector<double> tmp_colvec_1, tmp_colvec_2; // placeholders for inserting each column vector in 2d assignments
  double d_alpha, d_beta, ul_max, vl_max, d_ul, d_vl; // vars for calculating u/lambda and v/lambda

  // determine number of columns and rows to use (Npad * origina dims)
  nrow = _I.size()*Npad;
  ncol = _I[0].size()*Npad;		// ensures that both column and row sizes are even numbers
  
  // Check sizes of objects
  if (padded_I.size()!=nrow)
    padded_I.resize(nrow);
  if (padded_Q.size()!=nrow)
    padded_Q.resize(nrow);
  if (padded_U.size()!=nrow)
    padded_U.resize(nrow);
  if (padded_V.size()!=nrow)
    padded_V.resize(nrow);

  for (size_t i=0; i<nrow; ++i)
  {
    for (size_t i=0; i < nrow ; ++i)
    {
      padded_I[i].assign(ncol,0.0);
      padded_Q[i].assign(ncol,0.0);
      padded_U[i].assign(ncol,0.0);
      padded_V[i].assign(ncol,0.0);
    }
  }
  for (size_t i=0; i<_I.size(); ++i)
  {
    for (size_t j=0; j<_I[i].size(); ++j)
    {
      // Flip the image to account for the distinction between the image on the sky (so E on the left, N up) and
      // the baselines are defined on the Earth (so E on the right, N up).  Note that this reverses the sign of the
      // position angle.
      padded_I[i][j] = _I[int(_I.size())-1-int(i)][j];
      padded_Q[i][j] = _Q[int(_Q.size())-1-int(i)][j];
      padded_U[i][j] = _U[int(_U.size())-1-int(i)][j];
      padded_V[i][j] = _V[int(_V.size())-1-int(i)][j];
    }
  }
  
  // Get Visibilities by taking the FFT of I and then shifting it from
  // the standard FFT ordering to the standard analytical ordering.
  _VI = fft_shift(fft_2d(padded_I));	//fft then shift (need to program this, or find libs)
  _VQ = fft_shift(fft_2d(padded_Q));	//fft then shift (need to program this, or find libs)
  _VU = fft_shift(fft_2d(padded_U));	//fft then shift (need to program this, or find libs)
  _VV = fft_shift(fft_2d(padded_V));	//fft then shift (need to program this, or find libs)

  // Get step sizes, assumes uniform spacing
  d_alpha = _alpha[1][1] - _alpha[0][0];	//get the displacement value between adjacent indices
  d_beta = _beta[1][1] - _beta[0][0];		//for both RA and DEC angles

  // Construct u/lambda and v/lambda
  ul_max= 1.0/(2.0*d_alpha);		        //this is our MAX frequency (based on 1/image size)
  vl_max= 1.0/(2.0*d_beta);			//note that this is in units of lambda
  d_ul = (ul_max*2.0)/((double)(nrow));	//now we determine the incremental value for the u and v
  d_vl = (vl_max*2.0)/((double)(ncol));	//conversion so we can make a meshgrid for the u/v plane

  double norm = d_alpha*d_beta; // / (double(nrow)*double(ncol));
  //construct meshgrid for ul and vl
  if (_u.size()!=_v.size() || _v.size()!=nrow)
  {
    _u.resize(nrow);
    _v.resize(nrow);
  }

  std::complex<double> phase_centering_factor;
  for (size_t i=0; i<nrow; ++i)			//go through 2d map
  {
    if (_u[i].size()!=_v[i].size() || _v[i].size()!=ncol)
    {
      _u[i].resize(ncol);
      _v[i].resize(ncol);
    }
    for (size_t j=0; j<ncol; ++j)
    {
      //produce meshgrid recentering now not needed since I changed how the spacing is defined. (Paul Tiede)
      _u[i][j] = -ul_max + d_ul*((double)i);// - 0.5*d_ul;
      _v[i][j] = -vl_max + d_vl*((double)j);// - 0.5*d_vl;
      //we also need to phase center which is slightly annoying since we define intensities at pixel centers
      
      phase_centering_factor = norm*std::exp(-std::complex<double>(0,2.0*M_PI) * (_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );
      _VI[i][j] *= phase_centering_factor;
      _VQ[i][j] *= phase_centering_factor;
      _VU[i][j] *= phase_centering_factor;
      _VV[i][j] *= phase_centering_factor;
    }
  }


  // Reset the interpolation objects
  if (_i2du.size()!=nrow)
    _i2du.resize(nrow);
  if (_i2dv.size()!=ncol)
    _i2dv.resize(ncol);
  if (_i2dVI_r.size()!=_i2dVI_i.size() || _i2dVI_r.size() != nrow*ncol)
  {
    _i2dVI_r.resize(nrow*ncol);
    _i2dVI_i.resize(nrow*ncol);
  }
  if (_i2dVQ_r.size()!=_i2dVQ_i.size()|| _i2dVQ_r.size() != nrow*ncol)
  {
    _i2dVQ_r.resize(nrow*ncol);
    _i2dVQ_i.resize(nrow*ncol);
  }
  if (_i2dVU_r.size()!=_i2dVU_i.size()|| _i2dVU_r.size() != nrow*ncol)
  {
    _i2dVU_r.resize(nrow*ncol);
    _i2dVU_i.resize(nrow*ncol);
  }
  if (_i2dVV_r.size()!=_i2dVV_i.size()|| _i2dVV_r.size() != nrow*ncol)
  {
    _i2dVV_r.resize(nrow*ncol);
    _i2dVV_i.resize(nrow*ncol);
  }

  for (size_t i=0; i<nrow; ++i)
    _i2du[i] = _u[i][0];
  for (size_t j=0; j<ncol; ++j)
    _i2dv[j] = _v[0][j];
  for (size_t j=0; j<ncol; ++j)
    for (size_t i=0; i<nrow; ++i)
    {
      _i2dVI_r[j+ncol*i] = _VI[i][j].real();	//real
      _i2dVI_i[j+ncol*i] = _VI[i][j].imag();	//imaginary
      _i2dVQ_r[j+ncol*i] = _VQ[i][j].real();	//real
      _i2dVQ_i[j+ncol*i] = _VQ[i][j].imag();	//imaginary
      _i2dVU_r[j+ncol*i] = _VU[i][j].real();	//real
      _i2dVU_i[j+ncol*i] = _VU[i][j].imag();	//imaginary
      _i2dVV_r[j+ncol*i] = _VV[i][j].real();	//real
      _i2dVV_i[j+ncol*i] = _VV[i][j].imag();	//imaginary
    }

  _i2D_VIr.set_f(_i2du,_i2dv,_i2dVI_r);
  _i2D_VIi.set_f(_i2du,_i2dv,_i2dVI_i);
  _i2D_VQr.set_f(_i2du,_i2dv,_i2dVQ_r);
  _i2D_VQi.set_f(_i2du,_i2dv,_i2dVQ_i);
  _i2D_VUr.set_f(_i2du,_i2dv,_i2dVU_r);
  _i2D_VUi.set_f(_i2du,_i2dv,_i2dVU_i);
  _i2D_VVr.set_f(_i2du,_i2dv,_i2dVV_r);
  _i2D_VVi.set_f(_i2du,_i2dv,_i2dVV_i);

  _generated_visibilities = true;
}

//takes image and FFTW it
std::vector<std::vector<std::complex<double> > > model_polarized_image::fft_2d(const std::vector<std::vector<double> > &I)
{
  fftw_complex *in, *out;
  fftw_plan p;
  unsigned int rows = I.size();		//number of rows
  unsigned int cols = I[0].size(); 	//number of cols
  unsigned int numelements = rows*cols;
  
  std::vector<std::complex<double> > tmp_out;
  std::vector<std::vector<std::complex<double> > > out_c;		//input (complex var)
  
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  
  //now we do the same thing but we copy over the values from our complex vector
  for (unsigned int i=0; i<rows; i++)
  {
    for (unsigned int j=0; j<cols; j++)
    {
      in[i*cols + j][0] = I[i][j];	//copy it over to our fftw storage
      in[i*cols + j][1] = 0;			//assume no imaginary part for image
    }
  }

  fftw_execute(p); /* repeat as needed */
  
  out_c.clear();
  for (unsigned int i=0; i<rows; i++)
  {
    tmp_out.clear();
    for (unsigned int j=0; j<cols; j++)
    {
      //divide by N for result (include or not to include)
      tmp_out.push_back(std::complex<double>(out[i*cols + j][0], out[i*cols + j][1]));
    }
    out_c.push_back(tmp_out);
  }
  
  //memory cleanup
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
  
  return out_c;
}

//shift the FFTW result so that the low frequency is at center
std::vector<std::vector<std::complex<double> > > model_polarized_image::fft_shift(const std::vector<std::vector<std::complex<double> > > &V)
{
  
  std::vector<std::vector<std::complex<double> > > out_c = V;

  std::rotate(out_c.begin(), out_c.begin() + (out_c.size()+1)/2, out_c.end());
  for ( size_t i = 0; i < out_c.size(); i++ ){
    std::rotate(out_c[i].begin(), out_c[i].begin() + (out_c[i].size()+1)/2, out_c[i].end());
  }
  /*
  unsigned int rows = V.size();		//number of rows
  unsigned int cols = V[0].size(); 	//number of cols
  std::vector<std::complex<double> > tmp_out;
  std::vector<std::vector<std::complex<double> > > out_c;		//input (complex var)
  
  //perform shift here
  tmp_out.clear(); tmp_out.resize(cols,std::complex<double>(0,0));	//first make a 0,0 2d map
  out_c.clear(); out_c.resize(rows,tmp_out);
  unsigned int half_rows = std::floor(rows/2.0);		//if odd, then we are assuming the smaller square is
  unsigned int half_cols = std::floor(cols/2.0);		//in top left
  unsigned int new_i, new_j;						//shifted indices
  
  if (rows == 0 || cols == 0)
  {
    std::cout << "Warning: fft_shift() - V has no columns or rows" << std::endl;
  }
  
  for (unsigned int i=0; i<rows; i++)				//now perform shift
  {
    for (unsigned int j=0; j<cols; j++)
    {
      //this code does a diagonal block swap between quadrants 1 and 4 and 2 and 3
      //refer to code above for insight
      new_i = i + half_rows * (1 + -2*(i >= half_rows));		//set new indices
      new_j = j + half_cols * (1 + -2*(j >= half_cols));
      out_c[new_i][new_j].real(V[i][j].real());			//assign swapped values
      out_c[new_i][new_j].imag(V[i][j].imag());
    }
  }
  */
  
  return out_c;
}

};

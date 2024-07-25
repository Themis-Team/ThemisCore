/*!
  \file model_image.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements image model class.
  \details To be added
*/

#include "model_image.h"
#include "utils.h"
#include <cmath>
#include <valarray>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <fstream>
#include <typeinfo>

#include <ctime>

namespace Themis {

model_image::model_image()
  : _comm(MPI_COMM_WORLD), _generated_model(false), _generated_visibilities(false), _use_spline(false), _position_angle(0.0)
{
}

model_image::~model_image()
{
}


void model_image::write_model_tag_file(std::string tagfilename) const
{
  std::ofstream tagout(tagfilename);
  write_model_tag_file(tagout);
}

void model_image::write_model_tag_file(std::ofstream& tagout) const
{
  tagout << "tagvers-1.0\n" << model_tag() << std::endl;
}

void model_image::generate_model(std::vector<double> parameters)
{
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
    generate_image(parameters,_I,_alpha,_beta);
    
    // Set some boolean flags for what is and is not defined
    _generated_model = true;
    _generated_visibilities = false;
  }
}

void model_image::generate_complex_visibilities()
{
  if (_generated_model){
    if (!_generated_visibilities){
      compute_raw_visibilities();
      _generated_visibilities=true;
    }
  }
  else{
    std::cerr << "model_image::generate_complex_visibilities: Must generate model\n"
	      << "  prior to requesting visibilities.\n";
    std::exit(1);
  }
}

std::complex<double> model_image::visibility(datum_visibility& d, double)
{
  if (_generated_model){
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else{
    std::cerr << "model_image::visibility_amplitude: Must generate model\n"
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
    _i2D_Vr.bicubic_spline(ru,rv,Vr);
    _i2D_Vi.bicubic_spline(ru,rv,Vi);
  }
  else
  {
    _i2D_Vr.bicubic(ru,rv,Vr);
    _i2D_Vi.bicubic(ru,rv,Vi);  
  }
  std::complex<double> V = std::complex<double>(Vr,Vi);
  return (V);
}

double model_image::visibility_amplitude(datum_visibility_amplitude& d, double)
{
  if (_generated_model){
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else{
    std::cerr << "model_image::visibility_amplitude: Must generate model\n"
	      << "  prior to requesting visibility_amplitudes.\n";
    std::exit(1);
  }
  
  // Obtain counter-rotated (after reflection on the sky -- position angle is E of N) u,v coordinates
  double ru = d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle);
  double rv = -d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle);

  // Perform interpolation
  double VM;
  if (_use_spline)
    _i2D_VM.bicubic_spline(ru,rv,VM);
  else
    _i2D_VM.bicubic(ru,rv,VM);
  return ( VM );
}

double model_image::closure_phase(datum_closure_phase& d, double)
{
  if (_generated_model)
  {
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else
  {
    std::cerr << "model_image::closure_phase: Must generate model\n"
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
      _i2D_Vr.bicubic_spline(ru,rv,Vr);
      _i2D_Vi.bicubic_spline(ru,rv,Vi);
    }
    else
    {
      _i2D_Vr.bicubic(ru,rv,Vr);
      _i2D_Vi.bicubic(ru,rv,Vi);  
    }
    V[j] = std::complex<double>(Vr,Vi);
  }
  std::complex<double> V123 = V[0]*V[1]*V[2];
  
  return ( std::imag(std::log(V123))*180.0/M_PI );
}

double model_image::closure_amplitude(datum_closure_amplitude& d, double)
{
  if (_generated_model)
  {
    if (!_generated_visibilities)
      compute_raw_visibilities();
  }
  else
  {
    std::cerr << "model_image::closure_amplitude: Must generate model\n"
	      << "  prior to requesting visibility_amplitudes.\n";
    std::exit(1);
  }

  // Obtain counter-rotated u,v coordinates
  double u[]={d.u1,d.u2,d.u3,d.u4}, v[]={d.v1,d.v2,d.v3,d.v4};
  double ru, rv, VM[4];
  //double c=std::cos(-_position_angle), s=std::sin(-_position_angle);
  double c=std::cos(_position_angle), s=std::sin(_position_angle);
  for (int j=0; j<4; ++j)
  {
    ru = u[j]*c + v[j]*s;
    rv = -u[j]*s + v[j]*c;
    if ( _use_spline)
      _i2D_VM.bicubic_spline(ru,rv,VM[j]);
    else
      _i2D_VM.bicubic(ru,rv,VM[j]);
  }

  return ( (VM[0]*VM[2])/ (VM[1]*VM[3]) );
}


void model_image::output_image(std::string fname, bool rotate)
{
  //First check that image is actually filled
  if (_I.size()==0)
  {
    std::cerr << "model_image::output_image : Intensity grid is empty are you sure you generated the model/image?\n";
    std::exit(1);
  }

  std::vector<std::vector<double> > Irot(_alpha.size(), std::vector<double> (_alpha[0].size(),0.0));
  //Create spline object for rotation
  std::valarray<double> idx(_alpha.size());
  std::valarray<double> idy(_alpha[0].size());
  std::valarray<double> iIrot(_alpha.size()*_alpha[0].size());
  for ( size_t ii = 0; ii < _alpha.size(); ++ii )
    idx[ii] = _alpha[ii][0];
  for ( size_t jj = 0; jj < _alpha[0].size(); ++jj )
    idy[jj] = _beta[0][jj];
  for ( size_t jj = 0; jj < _alpha[0].size(); ++jj )
    for ( size_t ii = 0; ii < _alpha.size(); ++ii )
      iIrot[jj+_alpha.size()*ii] = _I[_alpha.size()-1-ii][jj];
  Interpolator2D I_interp;
  I_interp.set_f(idx,idy,iIrot);
  if (rotate)
  {
    for ( size_t ii = 0; ii < _alpha.size(); ++ii)
      for ( size_t jj = 0; jj < _alpha[0].size(); ++jj)
      {
        double x = std::cos(_position_angle)*_alpha[ii][jj] + std::sin(_position_angle)*_beta[ii][jj];
        double y = -std::sin(_position_angle)*_alpha[ii][jj] + std::cos(_position_angle)*_beta[ii][jj];
        I_interp.bicubic(x,y,Irot[ii][jj]);
      }
  }
  else
  {
    for ( size_t ii = 0; ii < _alpha.size(); ++ii)
      for ( size_t jj = 0; jj < _alpha[0].size(); ++jj)
      {
        Irot[ii][jj] = _I[_alpha.size()-1-ii][jj];
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
        << std::setw(15) << "I (Jy/str)"
        << std::endl;
  //double psize_x = _alpha[1][1] - _alpha[0][0];
  //double psize_y = _beta[1][1] - _beta[0][0];
  for ( size_t ix=0; ix<_alpha[0].size(); ix++ )
    for (size_t iy=0; iy<_alpha.size(); iy++)
      imout << std::setw(5) << iy
            << std::setw(5) << _alpha.size()-1-ix
            << std::setw(15) << Irot[iy][_alpha[0].size()-1-ix] << std::endl;
  imout.close();
  
}


  
void model_image::get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  alpha = _alpha;
  beta = _beta;
  I = _I;
}
void model_image::get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& V) const 
{

  if (!_generated_visibilities){
    std::cerr << "Visibilities not generated yet arrays will be empty\n";
  }
  u = _u;
  v = _v;
  V = _V;
}
void model_image::get_visibility_amplitudes(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<double> >& V) const
{
  if (!_generated_visibilities){
    std::cerr << "Visibilities not generated yet arrays will be empty\n";
  }
  u = _u;
  v = _v;
  V = _V_magnitude;
}

void model_image::use_spline_interp( bool use_spline )
{
  _use_spline = use_spline;
}


void model_image::compute_raw_visibilities()
{
  //time_t start_time;
  //std::time(&start_time);
  //std::cout << "Started in model_image::compute_raw_visibilities()" << std::endl;

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
  size_t ncol, nrow;					//dimensions of the fourier transform map
  std::vector<std::vector<double> > padded_I;		        //padded image
  std::vector<double> tmp_colvec_1, tmp_colvec_2;		//placeholders for inserting each column vector in 2d assignments
  double d_alpha, d_beta, ul_max, vl_max, d_ul, d_vl;	//vars for calculating u/lambda and v/lambda

  //determine number of columns and rows to use (Npad * origina dims)
  nrow = _I.size()*Npad;
  ncol = _I[0].size()*Npad;		//ensures that both column and row sizes are even numbers
  
  // Check sizes of objects
  if (padded_I.size()!=nrow)
  {
    //we now pad the borders of the image with 0 (based on the new size calculated)
    //	padded_I = pad_img(I, ncol, nrow, 0);
    padded_I.resize(nrow);
  }
  for (size_t i=0; i<nrow; ++i)
  {
    for (size_t i=0; i < nrow ; ++i)
      padded_I[i].assign(ncol,0.0);
  }
  for (size_t i=0; i<_I.size(); ++i)
  {
    for (size_t j=0; j<_I[i].size(); ++j)
      // Flip the image to account for the distinction between the image on the sky (so E on the left, N up) and
      // the baselines are defined on the Earth (so E on the right, N up).  Note that this reverses the sign of the
      // position angle.
      padded_I[i][j] = _I[int(_I.size())-1-int(i)][j];
  }
  
  // Get Visibilities by taking the FFT of I and then shifting it from
  // the standard FFT ordering to the standard analytical ordering.
  _V = fft_shift(fft_2d(padded_I));	//fft then shift (need to program this, or find libs)

  // Get step sizes, assumes uniform spacing
  d_alpha = _alpha[1][1] - _alpha[0][0];	//get the displacement value between adjacent indices
  d_beta = _beta[1][1] - _beta[0][0];		//for both RA and DEC angles


  // Construct u/lambda and v/lambda
  ul_max= 1.0/(2.0*d_alpha);		        //this is our MAX frequency (based on 1/image size)
  vl_max= 1.0/(2.0*d_beta);			//note that this is in units of lambda
  d_ul = (ul_max*2.0)/((double)(nrow));	//now we determine the incremental value for the u and v
  d_vl = (vl_max*2.0)/((double)(ncol));	//conversion so we can make a meshgrid for the u/v plane

  double norm = d_alpha*d_beta; // / (double(nrow)*double(ncol));
  //double kj = double(_I[0].size()) / (2.0 * double(ncol));
  //for (size_t i=0; i<nrow; ++i)
  //  for (size_t j=0; j<ncol; ++j)
  //    _V[i][j] *= norm*std::exp( std::complex<double>(0,2.0*M_PI) * (i*ki + j*kj) );
  //construct meshgrid for ul and vl
  if (_u.size()!=_v.size() || _v.size()!=nrow)
  {
    _u.resize(nrow);
    _v.resize(nrow);
  }
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
      //_V[i][j] *= norm*std::exp(-std::complex<double>(0,2.0*M_PI) * (_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );
      _V[i][j] *= norm * utils::fast_img_exp7( -(_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );
    }
  }



  //assign magnitude and phase(log of complex))
  if (_V_magnitude.size()!=nrow)
    _V_magnitude.resize(nrow);
  for (size_t i=0; i<nrow; ++i)
  {
    if (_V_magnitude[i].size()!=ncol)
      _V_magnitude[i].resize(ncol);
    for (size_t j=0; j<ncol; ++j)
      _V_magnitude[i][j] = std::abs(_V[i][j]);
  }

  // Reset the interpolation objects
  if (_i2du.size()!=nrow)
    _i2du.resize(nrow);
  if (_i2dv.size()!=ncol)
    _i2dv.resize(ncol);
  if (_i2dV_r.size()!=_i2dV_i.size() || _i2dV_i.size()!=_i2dV_M.size() || _i2dV_M.size()!=nrow*ncol)
  {
    _i2dV_M.resize(nrow*ncol);
    _i2dV_r.resize(nrow*ncol);
    _i2dV_i.resize(nrow*ncol);
  }

  for (size_t i=0; i<nrow; ++i)
    _i2du[i] = _u[i][0];
  for (size_t j=0; j<ncol; ++j)
    _i2dv[j] = _v[0][j];
  for (size_t j=0; j<ncol; ++j)
    for (size_t i=0; i<nrow; ++i)
    {
      _i2dV_M[j+ncol*i] = _V_magnitude[i][j];	//magnitude
      _i2dV_r[j+ncol*i] = _V[i][j].real();	//real
      _i2dV_i[j+ncol*i] = _V[i][j].imag();	//imaginary
    }

  _i2D_VM.set_f(_i2du,_i2dv,_i2dV_M);
  _i2D_Vr.set_f(_i2du,_i2dv,_i2dV_r);
  _i2D_Vi.set_f(_i2du,_i2dv,_i2dV_i);

  _generated_visibilities = true;
}

//takes image and FFTW it
std::vector<std::vector<std::complex<double> > > model_image::fft_2d(const std::vector<std::vector<double> > &I)
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
std::vector<std::vector<std::complex<double> > > model_image::fft_shift(const std::vector<std::vector<std::complex<double> > > &V)
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

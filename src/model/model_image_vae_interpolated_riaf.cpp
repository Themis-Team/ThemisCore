/*!
  \file model_image_vae_interpolated_riaf.cpp
  \author Ali Sarertoosi, Avery E. Broderick
  \date  January, 2023
  \brief Source file for variational auto-encoder interpolated RIAF model class.
  \details To be added
  \warning Requires Torch to be installed and TORCH_DIR to be specified in Makefile.config.
*/

#ifdef ENABLE_TORCH // Only available if Torch is configured

#include "model_image_vae_interpolated_riaf.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

// #include <fftw3-mpi.h>

// #include "stop_watch.h"

namespace Themis {

model_image_vae_interpolated_riaf::model_image_vae_interpolated_riaf(std::string modeldir, double frequency, double M, double D)
  : _comm(MPI_COMM_WORLD), _frequency(frequency), _M(M), _D(D), _zsize(0), _Npad(2)
{
  read_metadata(modeldir+"/metadata.txt");
  read_range_limits(modeldir+"/maxmin.csv");
  read_model(modeldir+"/model.pt");
  _size = _zsize+3;
  construct_ffts(_Npad);  
}

model_image_vae_interpolated_riaf::model_image_vae_interpolated_riaf(std::string metafile, std::string rangefile, std::string modelfile,
								     double frequency, double M, double D)
  : _comm(MPI_COMM_WORLD), _frequency(frequency), _M(M), _D(D), _zsize(0), _Npad(2)
{
  read_metadata(metafile);
  read_range_limits(rangefile);
  read_model(modelfile);
  _size = _zsize+3;
  construct_ffts(_Npad);  
}

model_image_vae_interpolated_riaf::~model_image_vae_interpolated_riaf()
{
  cleanup_ffts();
}
  
void model_image_vae_interpolated_riaf::read_metadata(std::string fname)
{
  _metafile = fname;
  std::ifstream fin(fname);
  if (fin.is_open()==false)
  {
    std::cerr << "ERROR: Could not open metadata file " << fname
	      << "\n in model_image_vae_interpolated_riaf::read_metadata\n";
    std::exit(1);
  }

  fin >> _xfov; // In M
  fin >> _yfov; // In M
  fin >> _Nx;
  fin >> _Ny;
  fin >> _zsize;
  fin.close();

  set_image_size();
}

void model_image_vae_interpolated_riaf::set_image_size(size_t target_Nx, size_t target_Ny)
{
  if (target_Nx>0)
    _reduction_factor_x = int(std::floor(_Nx/std::min(_Nx,int(target_Nx))));
  else
    _reduction_factor_x = 1;

  if (target_Ny>0)
    _reduction_factor_y = int(std::floor(_Ny/std::min(_Ny,int(target_Ny))));
  else
    _reduction_factor_y = 1;

  std::cerr << "Image size reduction factors: "
	    << std::setw(4) << _reduction_factor_x
	    << std::setw(4) << _reduction_factor_y
	    << '\n';
  
  int Nx = _Nx/_reduction_factor_x;
  int Ny = _Ny/_reduction_factor_y;
  double xfov = _xfov * double(_Nx)/double(_Nx-1) * double(Nx-1)/double(Nx);
  double yfov = _yfov * double(_Ny)/double(_Ny-1) * double(Ny-1)/double(Ny);
  
  _alpha.resize(Nx); _beta.resize(Nx); _I.resize(Nx);
  double Mtorad = _M/_D;
  _I.resize(Nx); _alpha.resize(Nx); _beta.resize(Nx);
  for (int i=0; i<Nx; i++)
  {
    _I[i].resize(Ny); _alpha[i].resize(Ny); _beta[i].resize(Ny);
    for (int j=0; j<Ny; j++)
    {
      _I[i][j] = 0.0;
      _alpha[i][j] = Mtorad*xfov*( double(i)/double(Nx-1) - 0.5 );
      _beta[i][j]  = Mtorad*yfov*( double(j)/double(Ny-1) - 0.5 );
    }
  }

  std::cerr << "  Dims and steps:"
	    << std::setw(4) << Nx
	    << std::setw(15) << xfov
	    << std::setw(15) << Mtorad*xfov/double(Nx-1)
	    << std::setw(4) << Ny
	    << std::setw(15) << yfov
	    << std::setw(15) << Mtorad*yfov/double(Ny-1)
	    << '\n';

  
  _flux_rescale = 1.0;
  _mass_rescale = 1.0;
}


  
void model_image_vae_interpolated_riaf::read_range_limits(std::string fname)
{
  _rangefile = fname;
  
  // Read in the max/min range limits
  //   This is a bad way to do this.
  std::ifstream fin(fname);
  if (fin.is_open()==false)
  {
    std::cerr << "ERROR: Could not open range limits file " << fname
	      << "\n in model_image_vae_interpolated_riaf::read_range_limits\n";
    std::exit(1);
  }
  std::vector<double> mmv;
  double dtmp;
  for (size_t j=0; j<_zsize; ++j)
  {
    fin >> dtmp;
    mmv.push_back(dtmp);

    if (fin.eof())
    {
      std::cerr << "ERROR: End of range limits file " << fname << " found during read"
		<< "\n in model_image_vae_interpolated_riaf::read_range_limits\n";
      std::exit(1);
    }
  }
  
  fin.close();

  // Set the maxmin range limits internally
  // WRONG, NOT RELATED TO LATENT SIZE!  MUST FIX.
  //_maxmins = torch::tensor(mmv).view({1,int(_zsize)});
}

void model_image_vae_interpolated_riaf::read_model(std::string fname)
{
  _modelfile = fname;

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    _module = torch::jit::load(fname);
  }
  catch (const c10::Error& e) {
    std::cerr << "ERROR: Could not load the torch model in " << fname << "\n";
    std::exit(1);
  }
}  
  
std::string model_image_vae_interpolated_riaf::model_tag() const
{
  std::stringstream tag;
  tag << "model_image_vae_interpolated_riaf " << _frequency
      << " " << _M << " " << _D
      << " " << _metafile << " " << _rangefile << " " << _modelfile;
  return tag.str();
}

void model_image_vae_interpolated_riaf::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  // Set the mass and flux rescaling factors
  _flux_rescale = std::exp(parameters[_zsize]);
  _mass_rescale = std::exp(parameters[_zsize+1]); // Because we are going to rescale the *passed* (u,v).
  parameters.pop_back();
  parameters.pop_back();

  // Short circuit if already generated
  if (parameters==_current_latent_parameters)
  {
    // Set to internal values
    alpha = _alpha;
    beta = _beta;
    I = _I;
    return;
  }
  else
  {
    _current_latent_parameters = parameters;
  }
  
  
  // Seems like an inefficient and inelegant solution
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::tensor(parameters).view({1,int(_zsize)}));
  
  torch::manual_seed(42);
  torch::NoGradGuard no_grad;
  auto output = _module.forward(inputs);
  // Execute the model and turn its output into a tensor.
  auto tensor_out1 = output.toTuple() -> elements()[0].toTensor();
  tensor_out1 = tensor_out1.contiguous();
  tensor_out1 = tensor_out1.mul(255).clamp(0, 255).to(c10::DeviceType::CPU);
  std::vector<double> v(tensor_out1.data_ptr<float>(), tensor_out1.data_ptr<float>() + tensor_out1.numel());

  // Get the total flux for renormalization
  double Itot = 0.0;
  for (size_t i=0; i<v.size(); ++i)
    Itot += v[i];
  Itot *= (_alpha[1][1]-_alpha[0][0])*(_beta[1][1]-_beta[0][0]); // / (_reduction_factor_x*_reduction_factor_y);
  double renorm = 1.0/Itot;
  
  // Rearrange the data and fill the desired output
  for (size_t i=0; i<_I.size(); i++)
    for (size_t j=0, j2=0; j<_I[i].size(); j++)
    {
      j2 = _I[i].size()-1-j; // Should always be in range 0 to _I[i].size()-1.
      //_I[i][j] = renorm*v[i+_I.size()*j];
      _I[i][j] = 0.0;
      //_I[i][j2] = 0.0;
      // While explicitly works, does sometimes result in significant difference in likelihoods
      for (size_t ii=0; ii<size_t(_reduction_factor_x); ii++)
	for (size_t jj=0; jj<size_t(_reduction_factor_y); jj++)
	  _I[i][j] += renorm*v[(i*_reduction_factor_x+ii)+(_I.size()*_reduction_factor_x)*(j*_reduction_factor_y+jj)];
	  //_I[i][j2] += renorm*v[(i*_reduction_factor_x+ii)+(_I.size()*_reduction_factor_x)*(j*_reduction_factor_y+jj)];
    }
  
  // Set to internal values
  alpha = _alpha;
  beta = _beta;
  I = _I;
}


std::complex<double> model_image_vae_interpolated_riaf::visibility(datum_visibility& d, double)
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
  double ru = ( d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle) )*_mass_rescale;
  double rv = (-d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle) )*_mass_rescale;

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
  return ( V*_flux_rescale );
}

double model_image_vae_interpolated_riaf::visibility_amplitude(datum_visibility_amplitude& d, double)
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
  double ru = ( d.u*std::cos(_position_angle) + d.v*std::sin(_position_angle) )*_mass_rescale;
  double rv = (-d.u*std::sin(_position_angle) + d.v*std::cos(_position_angle) )*_mass_rescale;

  // Perform interpolation
  double VM;
  if (_use_spline)
    _i2D_VM.bicubic_spline(ru,rv,VM);
  else
    _i2D_VM.bicubic(ru,rv,VM);
  return ( VM*_flux_rescale );
}

double model_image_vae_interpolated_riaf::closure_phase(datum_closure_phase& d, double)
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
  double c=_mass_rescale*std::cos(_position_angle), s=_mass_rescale*std::sin(_position_angle);
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

double model_image_vae_interpolated_riaf::closure_amplitude(datum_closure_amplitude& d, double)
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
  double c=_mass_rescale*std::cos(_position_angle), s=_mass_rescale*std::sin(_position_angle);
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

void model_image_vae_interpolated_riaf::construct_ffts(size_t Npad)
{
  // Initialize
  size_t ncol, nrow;					//dimensions of the fourier transform map
  double d_alpha, d_beta, ul_max, vl_max, d_ul, d_vl;	//vars for calculating u/lambda and v/lambda

  // Determine number of columns and rows to use (Npad * origina dims)
  nrow = _I.size()*_Npad;
  ncol = _I[0].size()*_Npad;		//ensures that both column and row sizes are even numbers
  
  // We now pad the borders of the image with 0 (based on the new size calculated)
  _padded_I.resize(nrow);
  for (size_t i=0; i<nrow; ++i)
    for (size_t i=0; i < nrow ; ++i)
      _padded_I[i].assign(ncol,0.0);

  // Get step sizes, assumes uniform spacing
  d_alpha = _alpha[1][1] - _alpha[0][0];	//get the displacement value between adjacent indices
  d_beta = _beta[1][1] - _beta[0][0];		//for both RA and DEC angles

  // Construct u/lambda and v/lambda
  ul_max= 1.0/(2.0*d_alpha);		        //this is our MAX frequency (based on 1/image size)
  vl_max= 1.0/(2.0*d_beta);			//note that this is in units of lambda
  d_ul = (ul_max*2.0)/((double)(nrow));	//now we determine the incremental value for the u and v
  d_vl = (vl_max*2.0)/((double)(ncol));	//conversion so we can make a meshgrid for the u/v plane

  // Set the normalization constant for FFTs
  _fft_norm = d_alpha*d_beta;

  // Construct meshgrid for ul and vl
  _u.resize(nrow);
  _v.resize(nrow);
  for (size_t i=0; i<nrow; ++i)			//go through 2d map
  {
    _u[i].resize(ncol);
    _v[i].resize(ncol);
    for (size_t j=0; j<ncol; ++j)
    {
      //produce meshgrid recentering now not needed since I changed how the spacing is defined. (Paul Tiede)
      _u[i][j] = -ul_max + d_ul*((double)i);// - 0.5*d_ul;
      _v[i][j] = -vl_max + d_vl*((double)j);// - 0.5*d_vl;
    }
  }

  // Make space for magnitude table
  _V_magnitude.resize(nrow);
  _V.resize(nrow);
  for (size_t i=0; i<nrow; ++i)
  {
      _V_magnitude[i].resize(ncol);
      _V[i].resize(ncol);
  }

  // Allocate space for the interpolation objects
  _i2du.resize(nrow);
  _i2dv.resize(ncol);
  _i2dV_M.resize(nrow*ncol);
  _i2dV_r.resize(nrow*ncol);
  _i2dV_i.resize(nrow*ncol);

  // Set the u,v interpolation tables
  for (size_t i=0; i<nrow; ++i)
    _i2du[i] = _u[i][0];
  for (size_t j=0; j<ncol; ++j)
    _i2dv[j] = _v[0][j];

  // Set up the FFTW buffers and plans
  unsigned int numelements = nrow*ncol;
  _in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  _out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numelements);
  _p = fftw_plan_dft_2d(nrow, ncol, _in, _out, FFTW_FORWARD, FFTW_ESTIMATE);

  //fftw_mpi_init();
  //_p = fftw_mpi_plan_dft_2d(nrow, ncol, _in, _out, _comm, FFTW_FORWARD, FFTW_ESTIMATE);
  // MUCH HARDER, REQUIRES ADJUSTING THE DATA ARRANGMENT AS WELL.
}

void model_image_vae_interpolated_riaf::cleanup_ffts()
{
  // Memory cleanup
  fftw_destroy_plan(_p);
  fftw_free(_in); fftw_free(_out);
}

void model_image_vae_interpolated_riaf::compute_raw_visibilities()
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

  size_t nrow = _padded_I.size();
  size_t ncol = _padded_I[0].size();

  for (size_t i=0; i<_I.size(); ++i)
  {
    for (size_t j=0; j<_I[i].size(); ++j)
      // Flip the image to account for the distinction between the image on the sky (so E on the left, N up) and
      // the baselines are defined on the Earth (so E on the right, N up).  Note that this reverses the sign of the
      // position angle.
      _padded_I[i][j] = _I[int(_I.size())-1-int(i)][j];
  }
  
  // Get Visibilities by taking the FFT of I and then shifting it from
  // the standard FFT ordering to the standard analytical ordering.
  _V = fft_shift(fft_2d(_padded_I));	//fft then shift (need to program this, or find libs)

  for (size_t i=0; i<nrow; ++i)			//go through 2d map
    for (size_t j=0; j<ncol; ++j)
      // We also need to phase center which is slightly annoying since we define intensities at pixel centers
      //_V[i][j] *= norm*std::exp(-std::complex<double>(0,2.0*M_PI) * (_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );
      _V[i][j] *= _fft_norm * utils::fast_img_exp7( -(_u[i][j]*_alpha[0][0] + _v[i][j]*_beta[0][0]) );

  // Assign magnitude and phase(log of complex))
  for (size_t i=0; i<nrow; ++i)
    for (size_t j=0; j<ncol; ++j)
      _V_magnitude[i][j] = std::abs(_V[i][j]);

  // Reset the interpolation objects
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
std::vector<std::vector<std::complex<double> > > model_image_vae_interpolated_riaf::fft_2d(const std::vector<std::vector<double> > &I)
{
  unsigned int rows = I.size();		//number of rows
  unsigned int cols = I[0].size(); 	//number of cols
    
  //now we do the same thing but we copy over the values from our complex vector
  for (unsigned int i=0; i<rows; i++)
  {
    for (unsigned int j=0; j<cols; j++)
    {
      _in[i*cols + j][0] = I[i][j];	//copy it over to our fftw storage
      _in[i*cols + j][1] = 0;			//assume no imaginary part for image
    }
  }

  fftw_execute(_p); /* repeat as needed */
  
  for (unsigned int i=0; i<rows; i++)
    for (unsigned int j=0; j<cols; j++)
      //divide by N for result (include or not to include)
      //tmp_out.push_back(std::complex<double>(out[i*cols + j][0], out[i*cols + j][1]));
      _V[i][j] = std::complex<double>(_out[i*cols + j][0], _out[i*cols + j][1]);
   
  return _V;
}

//shift the FFTW result so that the low frequency is at center
std::vector<std::vector<std::complex<double> > > model_image_vae_interpolated_riaf::fft_shift(const std::vector<std::vector<std::complex<double> > > &V)
{
  std::vector<std::vector<std::complex<double> > > out_c = V;
  std::rotate(out_c.begin(), out_c.begin() + (out_c.size()+1)/2, out_c.end());
  for ( size_t i = 0; i < out_c.size(); i++ ){
    std::rotate(out_c[i].begin(), out_c[i].begin() + (out_c[i].size()+1)/2, out_c[i].end());
  }
  return out_c;
}


void model_image_vae_interpolated_riaf::set_mpi_communicator(MPI_Comm comm)
{
  //std::cout << "model_image_sed_fitted_riaf proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
  _comm=comm;
  //open_error_streams();
  //cleanup_ffts();  
  //construct_ffts(_Npad);  
}

};

#endif

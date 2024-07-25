/*!
  \file model_image_sed_fitted_riaf_intensity.cpp
  \author Avery E. Broderick, Paul Tiede
  \date  August 8, 2019
  \brief Implements SED-fitted RIAF model class.
  \details To be added
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "model_image_sed_fitted_riaf_intensity.h"
#include "stop_watch.h"

namespace Themis {

model_image_sed_fitted_riaf_intensity::model_image_sed_fitted_riaf_intensity(std::string sed_fit_parameter_file, double frequency)
  : _comm(MPI_COMM_WORLD), _sdmp(sed_fit_parameter_file,2,3), _D(VRT2::VRT2_Constants::D_SgrA_cm), _frequency(frequency), _Nray(128), _imgR(15)
{
  open_error_streams();
}

void model_image_sed_fitted_riaf_intensity::open_error_streams()
{
#if 0
  if (_merr.is_open())
    _merr.close();
  
  int rank;
  MPI_Comm_rank(_comm,&rank);
  if (rank==0) {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&global_rank);
    std::stringstream fname;
    fname << "merr." << std::setfill('0') << std::setw(4) << global_rank;
    _merr.open(fname.str().c_str());
  }
#endif
}

void model_image_sed_fitted_riaf_intensity::use_small_images()
{
  set_image_resolution(32);
}

void model_image_sed_fitted_riaf_intensity::set_screen_size(double imgR)
{
  _imgR=imgR;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_sed_fitted_riaf_intensity: Rank " << rank << " using screen size " << _imgR << std::endl;
}

void model_image_sed_fitted_riaf_intensity::set_image_resolution(int Nray)
{
  _Nray=Nray;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_sed_fitted_riaf_intensity: Rank " << rank << " using image resolution " << _Nray << std::endl;
}

std::string model_image_sed_fitted_riaf_intensity::model_tag() const
{
  std::stringstream tag;
  tag << "model_image_sed_fitted_riaf_intensity " << _frequency << " " << _D;
  return tag.str();
}


void model_image_sed_fitted_riaf_intensity::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  /*
  std::cout << "model_image_sed_fitted_riaf_intensity::generate_image"
	    << std::setw(5) << MPI::COMM_WORLD.Get_rank()
	    << std::setw(15) << parameters[0]
	    << std::setw(15) << parameters[1]
	    << std::endl;
  */


  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "model_image_sed_fitted_riaf_intensity::generate_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }

  
  /////////////////////////////////////////////////////////
  // Set various parameters:

  // Assign names to given parameters
  const double density_factor = parameters[0];
  const double mass_cm = parameters[1]; //mass of black hole in cm
  const double a = std::min(std::max(0.0,parameters[2]),0.998);      // Spin magnitude
  //const double THETA = parameters[1];  // Inclination Angle
  //const double THETA = std::acos(parameters[1]) * 180.0/M_PI;  // Inclination Angle
  const double THETA = std::acos(std::min(std::max(parameters[3],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle
  const double alphaR = parameters[4]; //Infall parameter
  const double subKep = parameters[5]; //sub keplerian factor
  
  // Get model parameters from SED fit  <-- These can be provided as free parameters in the future (Note that -THETA has the same SED at +THETA and sets a floor of 1 degree)
  _sdmp.reset(a,std::max(std::acos(std::fabs(parameters[3])) * 180.0/M_PI,1.0));
  //_sdmp.reset(a,THETA);
  //  "Thermal" electron density normalization, radial power-law index, and h/r
  const double ne_norm = density_factor * _sdmp.ne_norm(); 
  const double ne_index = _sdmp.ne_index();
  const double ne_height = _sdmp.ne_height();
  //  "Thermal" electron temperature normalization, radial power-law index, and h/r
  const double Te_norm = _sdmp.Te_norm(); 
  const double Te_index = _sdmp.Te_index();
  const double Te_height = _sdmp.Te_height();
  //  "Nonthermal" electron density normalization, radial power-law index, and h/r
  const double nnth_norm = density_factor * _sdmp.nnth_norm(); 
  const double nnth_index = _sdmp.nnth_index();
  const double nnth_height = _sdmp.nnth_height();
  const double nnth_alpha = 1.25;
  const double nnth_gammamin = 100.0;
  //  Plasma beta for the magnetic field
  const double B_beta = 10.0;

  /*
  std::cerr << "density_factor " << density_factor << std::endl
            << "mass in cm     " << mass_cm << std::endl
            << "spin of bh     " << a << std::endl
            << "cos Inc        " << THETA << std::endl
            << "alphaR         " << alphaR << std::endl
            << "subkep         " << subKep << std::endl
            << "imgR           " << _imgR << std::endl;

  */
  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();

  // Choose when to stop rays
  VRT2::StopCondition stop(g,100,1.01);

  // Choose accretion flow velocity
  VRT2::AFV_ShearingInflow afv(g,risco, alphaR, subKep); // switches to free-fall at ISCO
  
  // CHOOSE RADIATIVE TRANSFER EFFECTS
  std::vector<VRT2::RadiativeTransfer*> rts;
  VRT2::RT_PW_PA rt_geometric(g,THETA*M_PI/180.0);
  rts.push_back(&rt_geometric);
  
  // Create disk model
  VRT2::ED_RPL ne_disk(ne_norm,ne_index,ne_height);
  VRT2::T_RPL Te_disk(Te_norm,Te_index,Te_height); // Note that Te_height is not used
  VRT2::ED_RPL ne_hotdisk(nnth_norm,nnth_index,nnth_height);
		       
  // Disk thermal synchrotron
  VRT2::MF_ToroidalBeta B(g,afv,ne_disk,B_beta);
  VRT2::RT_ThermalSynchrotron rt_thsync(g,ne_disk,Te_disk,afv,B);
  // Disk power-law component
  VRT2::RT_PowerLawSynchrotron rt_plsync_hd(g,ne_hotdisk,afv,B,nnth_alpha,nnth_gammamin);
  rts.push_back(&rt_thsync);
  rts.push_back(&rt_plsync_hd);
  
  // DEFINE AGGREGATE RT OBJECT
  VRT2::RT_Multi rt(g,rts);
  rt.set_length_scale(mass_cm);
  rt.set_frequency_scale(1.0);
  
  // DEFINE RAY AND POLARIZATION MAP
  VRT2::NullGeodesic ray(g,rt,stop);
  VRT2::PolarizationMap pmap(g,ray,mass_cm,_D,_comm,VERBOSITY);
  // Set frequency scale
  pmap.set_f0(_frequency);
  // Set perspective
  const double R = 2000; // Distance to image plane in M
  pmap.set_R_THETA(R,THETA);

#if (VERBOSITY>0)
  pmap.set_progress_stream("progress");
#endif

  double delta = 1e-4;	
  pmap.generate(-_imgR-delta,_imgR,_Nray,-_imgR-delta,_imgR,_Nray);
  pmap.integrate();

  double Mtorad = mass_cm/_D;
  I.resize(_Nray); alpha.resize(_Nray); beta.resize(_Nray);
  for (int i=0; i<_Nray; i++)
  {
    I[i].resize(_Nray); alpha[i].resize(_Nray); beta[i].resize(_Nray);
    for (int j=0; j<_Nray; j++)
    {
      I[i][j] = pmap.I(i,j) / std::pow(Mtorad*(2.0*_imgR-delta)/double(_Nray-1),2);
      alpha[i][j] = Mtorad*( (2.0*_imgR-delta)/double(_Nray-1)*double(i) - _imgR-delta );
      if (THETA>0)
	beta[i][j]  = Mtorad*( (2.0*_imgR-delta)/double(_Nray-1)*double(j) - _imgR-delta );
      else
	beta[i][_Nray-1-j]  = -Mtorad*( (2.0*_imgR-delta)/double(_Nray-1)*double(j) - _imgR-delta );
    }
  }



  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_image_sed_fitted_riaf_intensity::generate_image finished at:";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.print_time(_merr,"    Total Time:");
  }
}

};

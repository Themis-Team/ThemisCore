/*!
  \file model_image_sed_fitted_force_free_jet.cpp
  \author Paul Tiede
  \date  Sep, 2018
  \brief Implements SED-fitted force free jet model class.
  \details To be added
*/

#include "model_image_sed_fitted_force_free_jet.h"
#include <algorithm>
#include <cmath>

#include <iostream>
#include <iomanip>

#include "stop_watch.h"

namespace Themis {

model_image_sed_fitted_force_free_jet::model_image_sed_fitted_force_free_jet(std::string sed_fit_parameter_file, double M87_mass_cm, double M87_distance_cm, double frequency)
  : _comm(MPI_COMM_WORLD), _sdmp(sed_fit_parameter_file,3,3,3), _M(M87_mass_cm), _D(M87_distance_cm), _frequency(frequency), _xNray(128), _yNray(128), _xlow(-60), _xhigh(80), _ylow(-60), _yhigh(80)
{
  open_error_streams();
}

void model_image_sed_fitted_force_free_jet::open_error_streams()
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

void model_image_sed_fitted_force_free_jet::use_small_images()
{
  set_image_resolution(32,32);
}

void model_image_sed_fitted_force_free_jet::set_image_resolution(int xNray, int yNray)
{
  _xNray=xNray;
  _yNray=yNray;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_sed_fitted_force_free_jet: Rank " << rank << " using image resolution " 
            << _xNray << " x " << _yNray << std::endl;
}

void model_image_sed_fitted_force_free_jet::set_image_dimensions(double xlow, double xhigh, double ylow, double yhigh)
{
  _xlow = xlow;
  _xhigh = xhigh;
  _ylow = ylow;
  _yhigh = yhigh;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_sed_fitted_force_free_jet: Rank " << rank << " using image dimensions:\n" 
            << "x: ( " << _xlow << "," << _xhigh << " ),\n" 
            << "y: ( " << _ylow << "," << _yhigh << " )" 
            << std::endl;
}


double model_image_sed_fitted_force_free_jet::generate_renormalized_image(double density_factor, int xNray, int yNray, std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  /*
  int myrank, mysize;
  MPI_Comm_rank(_comm,&myrank);
  MPI_Comm_size(_comm,&mysize);
  std::cout << "model_image_sed_fitted_force_free_jet::generate_renormalized_image"
	    << std::setw(5) << myrank
	    << std::setw(15) << parameters[0]
	    << std::setw(15) << parameters[1]
	    << std::setw(5) << myrank
	    << std::setw(5) << mysize
	    << std::endl;
  */

  
  /////////////////////////////////////////////////////////
  // Set various parameters:

  // Assign names to given parameters

  const double a = std::min(std::max(0.0,parameters[1]),0.998);      // Spin magnitude
  //const double THETA = parameters[1];  // Inclination Angle
  //const double THETA = std::acos(parameters[1]) * 180.0/M_PI;  // Inclination Angle
  const double THETA = std::acos(std::min(std::max(parameters[2],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle


  const double jet_p = parameters[3];
  const double jet_open_angle = parameters[4];
  const double rLoad = parameters[5];
  const double gamma_max = parameters[6];

  /*
  std::cout << std::setw(10) << "spin"
            << std::setw(10) << "THETA"
            << std::setw(10) << "jet_p"
            << std::setw(10) << "jet_open"
            << std::setw(10) << "rLoad"
            << std::setw(10) << "gammaMax" << std::endl;
  std::cout << std::setw(10) << a
            << std::setw(10) << THETA
            << std::setw(10) << jet_p
            << std::setw(10) << jet_open_angle
            << std::setw(10) << rLoad
            << std::setw(10) << gamma_max << std::endl;
  */

  _sdmp.reset(1.0, a, rLoad);
  //  "Thermal" electron density normalization, radial power-law index, and h/r

  const double jet_density = _sdmp.nj_norm()*density_factor; 

  const double jet_alpha = _sdmp.nj_index();
  const double Bj = _sdmp.bj_norm();
  
  const double jet_gammamin = 100.0;
  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();
  const double jet_disk_edge = risco;

  // Choose when to stop rays
  VRT2::StopCondition stop(g,100,1.01);

 
  // CHOOSE RADIATIVE TRANSFER EFFECTS
  std::vector<VRT2::RadiativeTransfer*> rts;
  VRT2::RT_PW_PA rt_geometric(g,THETA*M_PI/180.0);
  rts.push_back(&rt_geometric);

  //Create jet
  VRT2::ForceFreeJet jet(g, jet_p, jet_disk_edge, jet_open_angle, rLoad, Bj, jet_density, gamma_max);
  VRT2::RT_PowerLawSynchrotron jet_plsync(g, jet.ed(), jet.afv(), jet.mf(), jet_alpha, jet_gammamin);
  rts.push_back(&jet_plsync);
 
  // DEFINE AGGREGATE RT OBJECT
  VRT2::RT_Multi rt(g,rts);
  rt.set_length_scale(_M);
  rt.set_frequency_scale(1.0);
  
  // DEFINE RAY AND POLARIZATION MAP
  VRT2::NullGeodesic ray(g,rt,stop);
  VRT2::PolarizationMap pmap(g,ray,_M,_D,_comm,VERBOSITY);
  // Set frequency scale
  pmap.set_f0(_frequency);
  // Set perspective
  const double R = 2000; // Distance to image plane in M
  pmap.set_R_THETA(R,THETA);

#if (VERBOSITY>0)
  pmap.set_progress_stream("progress");
#endif

  
  double delta = 1e-4;
	

  pmap.generate(_xlow-delta,_xhigh,xNray,_ylow-delta,_yhigh,yNray);

  double dx = _xhigh - _xlow;
  double dy = _yhigh - _ylow;
  pmap.integrate();

  double Mtorad = _M/_D;
  I.resize(xNray); alpha.resize(xNray); beta.resize(xNray);
  for (int i=0; i<xNray; i++)
  {
    I[i].resize(yNray); alpha[i].resize(yNray); beta[i].resize(yNray);
    for (int j=0; j<yNray; j++)
    {
      I[i][j] = pmap.I(i,j) / (Mtorad*dx/double(xNray-1) * Mtorad*dy/double(yNray-1) );
      alpha[i][j] = Mtorad*( dx/double(xNray-1)*double(i) + _xlow - delta );
      if (THETA>0)
	      beta[i][j]  = Mtorad*( dy/double(yNray-1)*double(j) + _ylow - delta );
      else
	      beta[i][yNray-1-j]  = -Mtorad*( dy/double(yNray-1)*double(j)  + _ylow - delta );
    }
  }

  return pmap.I_int();
}

void model_image_sed_fitted_force_free_jet::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  
  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "model_image_sed_fitted_force_free_jet::generate_renormalized_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }


  // Renormalize all disk fluxes to some target Jy, somewhat less than used 
  // for the SED fitting, but comparable to that on observed epochs
  // thus far.
  int xNdfit = _xNray/2;
  int yNdfit = _yNray/2;

  const double jet_flux_target=parameters[0];
  std::vector<double> dfactors(3), fluxes(3);
  dfactors[0]=0.6;
  dfactors[1]=0.9;
  dfactors[2]=1.1;
  for (size_t iF=0; iF<dfactors.size(); ++iF)
    fluxes[iF] = generate_renormalized_image(dfactors[iF], xNdfit, yNdfit, parameters,I,alpha,beta);
  // Do interpolations in log-log
  for (size_t iF=0; iF<dfactors.size(); ++iF)
  {
    fluxes[iF] = std::log(fluxes[iF]);
    dfactors[iF] = std::log(dfactors[iF]);
  }
  // Get best guess from quadratic interpolation, usually good to
  // better than 1% or so.
  double ldf = std::log(jet_flux_target);
  double dfactor = std::exp(
		       (ldf-fluxes[0])*(ldf-fluxes[1])*dfactors[2]/( (fluxes[2]-fluxes[0])*(fluxes[2]-fluxes[1]) )
		       +
		       (ldf-fluxes[1])*(ldf-fluxes[2])*dfactors[0]/( (fluxes[0]-fluxes[1])*(fluxes[0]-fluxes[2]) )
		       +
		       (ldf-fluxes[2])*(ldf-fluxes[0])*dfactors[1]/( (fluxes[1]-fluxes[2])*(fluxes[1]-fluxes[0]) )
		       );

  // Generate the image
  double flux = generate_renormalized_image(dfactor, _xNray, _yNray, parameters, I, alpha, beta);  
  
  if (myrank==0)
    std::cout << "Flux = "<< flux << " at spin = " << parameters[1] 
              << ", rload = " << parameters[5] << std::endl;
  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_image_sed_fitted_riaf::generate_image finished at:";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.print_time(_merr,"    Total Time:");
  }

}


}

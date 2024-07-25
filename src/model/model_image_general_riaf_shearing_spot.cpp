/*!
  \file model_image_general_riaf_shearing_spot.cpp
  \author Paul Tiede
  \date  August, 2019
  \brief Implements Shearing spot image class.
*/

#include "model_image_general_riaf_shearing_spot.h"
#include <algorithm>
#include <cmath>

#include <iostream>
#include <iomanip>
#include <sstream>

#include "stop_watch.h"
#include "constants.h"

namespace Themis {

model_image_general_riaf_shearing_spot::model_image_general_riaf_shearing_spot(double start_obs, double tobs, double frequency, double D)
  : _comm(MPI_COMM_WORLD),_start_obs(start_obs), _tobs(tobs),  _frequency(frequency), _D(D), _Nray(128), _Rmax(15)
{
  open_error_streams();
}

void model_image_general_riaf_shearing_spot::open_error_streams()
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

void model_image_general_riaf_shearing_spot::set_image_resolution(int Nray)
{
  _Nray=Nray;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_general_riaf_shearing_spot: Rank " << rank << " using image resolution " << _Nray << std::endl;
}

void model_image_general_riaf_shearing_spot::set_screen_size(double Rmax)
{
	_Rmax = Rmax;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_general_riaf_shearing_spot: Rank " << rank << " using screen size " << _Rmax << std::endl;

}

std::string model_image_general_riaf_shearing_spot::model_tag() const
{
  std::stringstream tag;
  tag << "model_image_general_riaf_shearing_spot " << _frequency << " " << _D;
  return tag.str();
}

void model_image_general_riaf_shearing_spot::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_image_general_riaf_shearing_spot::generate_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }
  /////////////////////////////////////////////////////////
  // Set various parameters:

  // Assign names to given parameters
  const double mass_cm = parameters[0]*constants::G*constants::Msun/(constants::c*constants::c);
  const double a = std::min(std::max(0.0,parameters[1]),0.998);      // Spin magnitude
  //const double THETA = parameters[1];  // Inclination Angle
  const double THETA = std::acos(std::min(std::max(parameters[2],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle

  //Spot density factor
  const double nspot = parameters[3]; //Density factor for renormalized image
  //Spot width
  const double rspot = parameters[4]; 
  double c = constants::c;
  //Initial spot location
  const double t0 = c*parameters[5]/mass_cm-2000; //Spot start time measured in M taking off an estimate of the initial shift since screen is 2000M away.
	
  
  const double tobsM = c*_tobs/mass_cm; //Spot observed converting to units of M to cgs
  const double tSM = c*_start_obs/mass_cm;

  const double r0 = parameters[6];
  const double outer_radius = r0 + rspot*30.0;
  const double phi0 = parameters[7];
  //Accretion flow parameters
  //
  //  "Thermal" electron density normalization, radial power-law index, and h/r
  const double ne_norm = parameters[8];
  const double ne_index = parameters[9];
  const double ne_height = parameters[10];
  //  "Thermal" electron temperature normalization, radial power-law index, and h/r
  const double Te_norm = parameters[11];
  const double Te_index = parameters[12];
  const double Te_height = 1.0; // Not used!
  //  "Nonthermal" electron density normalization, radial power-law index, and h/r
  const double nnth_norm = parameters[13];
  const double nnth_index = parameters[14];
  const double nnth_height = parameters[15];
  const double nnth_alpha = 1.25;
  const double nnth_gammamin = 100;
  //  Plasma beta for the magnetic field
  const double B_beta = 10;
  //Infall parameter
  
  const double alphaR = parameters[16];
  const double subKep = parameters[17];


  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();

  // Choose when to stop rays
  VRT2::StopCondition stop(g,100,1.01);

  // Choose accretion flow velocity
  VRT2::AFV_ShearingInflow afv(g,risco, alphaR, subKep);
  
  // CHOOSE RADIATIVE TRANSFER EFFECTS
  std::vector<VRT2::RadiativeTransfer*> rts;
  VRT2::RT_PW_PA rt_geometric(g,THETA*M_PI/180.0);
  rts.push_back(&rt_geometric);
  
  // Create disk model for background magnetic field
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

  //Create shearing spot model
  VRT2::ED_shearing_infall_spot ne_spot(g, nspot, rspot, risco, 
                                        afv, alphaR, subKep, outer_radius, 
					t0, r0, M_PI/2.0, phi0);
  //Set table
  ne_spot.generate_table();
  
  //Set offset time negative since want to push initial time back
  ne_spot.set_toffset(tSM - tobsM);
		
  //RAdiative transfer object for the spot
  const double specIndx = 1.08;
  const double gammaMin = 100;
  VRT2::RT_PowerLawSynchrotron rt_spot(g, ne_spot, afv, B, specIndx, gammaMin);
  rts.push_back(&rt_spot);


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

  double r_grid = _Rmax;
  double delta = 1e-4;
	
  pmap.generate(-r_grid-delta,r_grid,_Nray,-r_grid-delta,r_grid,_Nray);
  pmap.integrate();

  double Mtorad = mass_cm/_D;
  I.resize(_Nray); alpha.resize(_Nray); beta.resize(_Nray);
  //Check if image is nan'd because ray tracing failed. 
  //If failed fill image with a constant value array of 1Jy/px
  if (std::isnan(pmap.I_int()))
  {
    std::cerr << "model_image_general_riaf_shearing_spot: Image nan'd setting image to a constant of 1Jy/px.\n";

    for (int i=0; i<_Nray; i++)
    {
      I[i].resize(_Nray); alpha[i].resize(_Nray); beta[i].resize(_Nray);
      for (int j=0; j<_Nray; j++)
      {
        I[i][j] = 1.0 / std::pow(Mtorad*(2.0*r_grid-delta)/double(_Nray-1),2);
        alpha[i][j] = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(i) - r_grid-delta );
        if (THETA>0)
          beta[i][j]  = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
        else
	  beta[i][_Nray-1-j]  = -Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
      }

    }
  }
  else
  {
    for (int i=0; i<_Nray; i++)
    {
      I[i].resize(_Nray); alpha[i].resize(_Nray); beta[i].resize(_Nray);
      for (int j=0; j<_Nray; j++)
      {
        I[i][j] = pmap.I(i,j) / std::pow(Mtorad*(2.0*r_grid-delta)/double(_Nray-1),2);
        alpha[i][j] = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(i) - r_grid-delta );
        if (THETA>0)
          beta[i][j]  = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
        else
	  beta[i][_Nray-1-j]  = -Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
      }
    }
  }

  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_image_general_riaf_shearing_spot::generate_image finished at:";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.print_time(_merr,"    Total Time:");
  }
}

};

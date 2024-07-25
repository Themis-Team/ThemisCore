/*!
  \file model_image_force_free_jet.cpp
  \author Paul Tiede
  \date  Sep, 2018
  \brief Implements SED-fitted force free jet model class.
  \details To be added
*/

#include "model_image_force_free_jet.h"
#include <algorithm>
#include <cmath>

#include <iostream>
#include <iomanip>

#include "stop_watch.h"

namespace Themis {

model_image_force_free_jet::model_image_force_free_jet(double frequency)
  : _comm(MPI_COMM_WORLD), _frequency(frequency), _Nray_base(64),_Nray(64),_number_of_refines(0), _Rmax(20)
{
  open_error_streams();
}

void model_image_force_free_jet::open_error_streams()
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

void model_image_force_free_jet::use_small_images()
{
  set_image_resolution(32);
}

void model_image_force_free_jet::set_image_resolution(int Nray, int number_of_refines)
{
  _Nray_base=Nray;
  _Nray = _Nray_base;
  _number_of_refines = number_of_refines;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_ffj: Rank " << rank << " using image resolution " << _Nray << std::endl;
  
  if (number_of_refines!=0)
    std::cout << "model_image_ffj: VRT2 refine turned on! Refining base map " 
              << number_of_refines << " times on rank " << rank << std::endl;
}

void model_image_force_free_jet::set_screen_size(double Rmax)
{
  _Rmax = Rmax;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_image_force_free_jet: Rank " << rank << " using " 
            << 2*Rmax << "M field of view."  
            << std::endl;
}


void model_image_force_free_jet::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  /*
  int myrank, mysize;
  MPI_Comm_rank(_comm,&myrank);
  MPI_Comm_size(_comm,&mysize);
  std::cout << "model_image_force_free_jet::generate_renormalized_image"
	    << std::setw(5) << myrank
	    << std::setw(15) << parameters[0]
	    << std::setw(15) << parameters[1]
	    << std::setw(5) << myrank
	    << std::setw(5) << mysize
	    << std::endl;
  */

  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "model_image_force_free_jet::generate_renormalized_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }

  
  /////////////////////////////////////////////////////////
  // Set various parameters:

  // Assign names to given parameters
  const double mass_cm = parameters[0]*constants::G*constants::Msun/(constants::c*constants::c);
  const double distance_cm = parameters[1]*constants::pc*1e3; 
  const double a = std::min(std::max(0.0,parameters[2]),0.998);      // Spin magnitude
  const double THETA = std::acos(std::min(std::max(parameters[3],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle

  const double jet_p = parameters[4];
  const double jet_open_angle = parameters[5];
  const double rLoad = parameters[6];
  const double gamma_max = parameters[7];
  
  const double jet_density = parameters[8]; 
  const double jet_alpha = parameters[9];
  const double Bj = parameters[10];
  
  const double jet_gammamin = parameters[11];

  /*
  for ( int i = 0; i < parameters.size(); i++)
    std::cout << std::setw(15) << parameters[i];
  std::cout << std::endl;
  */

  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();
  const double jet_disk_edge = parameters[12]*risco;

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
  rt.set_length_scale(mass_cm);
  rt.set_frequency_scale(1.0);
  
  // DEFINE RAY AND POLARIZATION MAP
  VRT2::NullGeodesic ray(g,rt,stop);
  VRT2::PolarizationMap pmap(g,ray,mass_cm,distance_cm,_comm,VERBOSITY);
  // Set frequency scale
  pmap.set_f0(_frequency);
  // Set perspective
  const double R = 2000; // Distance to image plane in M
  pmap.set_R_THETA(R,THETA);

#if (VERBOSITY>0)
  pmap.set_progress_stream("progress");
#endif

  
  double delta = 1e-4;
  double r_grid = _Rmax;
	
  pmap.generate(-r_grid-delta, r_grid, _Nray_base, -r_grid-delta, r_grid, _Nray_base);
  pmap.integrate();

  for ( int ir = 0; ir < _number_of_refines; ir++ )
    pmap.refine();
  
  int Nray_x = pmap.xi_size();
  int Nray_y = pmap.eta_size();
  _Nray = std::min(Nray_x,Nray_y);


  double Mtorad = mass_cm/distance_cm;
  I.resize(_Nray); alpha.resize(_Nray); beta.resize(_Nray);
  if (!std::isnan(pmap.I_int()))
  {
    for (int i=0; i<_Nray; i++)
    {
        I[i].resize(_Nray); alpha[i].resize(_Nray); beta[i].resize(_Nray);
        for (int j=0; j<_Nray; j++)
        {
          I[i][j] = std::max(pmap.I(i,j),0.0) / std::pow(Mtorad*(2.0*r_grid-delta)/double(_Nray-1),2);
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
    std::cerr << "Image is nan'd, filling image with constant flux (1Jy/px)\n";
    std::cerr << "Image parameters: \n";
    for ( size_t i = 0; i < parameters.size(); ++i)
      std::cerr << std::setw(5) << parameters[i] << std::endl;


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
}


double model_image_force_free_jet::generate_flux_estimate(double accuracy, std::vector<double> parameters)
{
  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "model_image_force_free_jet::generate_renormalized_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }

  
  /////////////////////////////////////////////////////////
  // Set various parameters:

  // Assign names to given parameters
  const double mass_cm = parameters[0] = constants::G*constants::Msun/(constants::c*constants::c);
  const double distance_cm = parameters[1]*constants::pc*1e3; 
  const double a = std::min(std::max(0.0,parameters[2]),0.998);      // Spin magnitude
  const double THETA = std::acos(std::min(std::max(parameters[3],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle

  const double jet_p = parameters[4];
  const double jet_open_angle = parameters[5];
  const double rLoad = parameters[6];
  const double gamma_max = parameters[7];
  
  const double jet_density = parameters[8]; 
  const double jet_alpha = parameters[9];
  const double Bj = parameters[10];
  
  const double jet_gammamin = parameters[11];

  /*
  for ( int i = 0; i < parameters.size(); i++)
    std::cout << std::setw(15) << parameters[i];
  std::cout << std::endl;
  */

  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();
  const double jet_disk_edge = parameters[12]*risco;

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
  rt.set_length_scale(mass_cm);
  rt.set_frequency_scale(1.0);
  
  // DEFINE RAY AND POLARIZATION MAP
  VRT2::NullGeodesic ray(g,rt,stop);
  VRT2::PolarizationMap pmap(g,ray,mass_cm,distance_cm,_comm,VERBOSITY);
  // Set frequency scale
  pmap.set_f0(_frequency);
  // Set perspective
  const double R = 2000; // Distance to image plane in M
  pmap.set_R_THETA(R,THETA);

#if (VERBOSITY>0)
  pmap.set_progress_stream("progress");
#endif

  int itermax = 500;
  double safety = 1.2;
  double rmin = 2.0;
  double dtheta = 0.5;
  double Icut_fraction = 1e-3;

  double r_grid = estimate_image_size_log_spiral(pmap, rmin, dtheta, Icut_fraction, itermax, safety);

  ///////////
  //
  // Generate and refine until the flux varies by less than
  // of the best guess 5% and 20% of the relevant errorbars
  // or has refined 5 levels (this takes NxN to [2^5(N-1)+1]x[2^5(N-1)+1],
  // i.e., starting off with 5x5 this gives a maximum effective resolution
  // of 129x129, starting with 10x10 givs 289x289, etc.)
  //
  // Conditions:  1) refined level is less than our max # of refined levels AND
  //              2) our flux difference (between refinements) over our flux error 
  //                 is greater than "accuracy" OR our flux didn't change all that
  //                 much between refinements.
  //
  int Nrays_start = 10;
  double delta = 1e-4;

  pmap.generate(-r_grid-delta,r_grid,Nrays_start,-r_grid-delta,r_grid,Nrays_start);
  
  double refine_ratio   = 1e-1;       //refine amount (ratio)
  int refinement_level  = 0;          //current refine level
  int number_of_refines = 3;          //total levels to refine
  double I_old, I_new;                //intensity placeholders
  pmap.integrate();                   //"sum" up intensities for each grid point
  I_new = pmap.I_int();               //our flux value is the intensity sum for the grid points

  //std::cerr << "model_image_riaf::generate_flux_estimate : generated first image " << I_new << "\n";

  do                                  //now we keep refining until certain conditions are met
  {
    I_old = I_new;                    //set old intensity/flux
    pmap.refine(refine_ratio);        //refine our map
    pmap.integrate();                 //find new flux
    I_new = pmap.I_int();             //set new flux
    refinement_level++;               //increment our current level of refinement

    
    /*std::cerr << "model_image_riaf::generate_flux_estimate : " << refinement_level
	      << std::setw(15) << I_old 
	      << std::setw(15) << I_new << "\n";*/
    
  } 
  while ( (refinement_level<=number_of_refines) && (std::fabs(I_new-I_old)>accuracy || std::fabs(I_new-I_old)>0.05*std::fabs(I_new+I_old)) ); // There is a minimum accuracy that is being specified here that may need to be readdressed at a later date.

  return I_new;
}


double model_image_force_free_jet::estimate_image_size_log_spiral(VRT2::PolarizationMap &pmap, double initial_r, double theta_step, 
							double percent_from_max_I, unsigned int max_iter, double safety_factor)
{
  /////////////////////////////////////////////////////
  // Use a log spiral outwards, keep going until we reach a max number of iterations
  // or until we find a point (cx,cy) which has an intensity cutoff of some ratio of our max intensity.
  // Then we ensure that the spiral goes another loop to see if there's no intensity value
  // greater than our ratio*max_intensity and set our (cx,cy) as the size if there isn't.
  // Returns the length of (cx,cy) * safety_factor if a size was found, otherwise returns -1.
  /////////////////////////////////////////////////////

  double max_size = 0, max_intensity = 0;  //maximum size, and maximum intensity
  double c_intensity, c_x, c_y;		   //current intensity at our current x,y coord
  double c_r, c_theta;			   //corresponding r and theta for our x,y coord
  bool found_size = false;		   //did we find a size?
  bool theta_chk;			   //used to determine if we scanned an entire revolution (incase we get a false size estimate)
  double theta_chk_start=0.0;		   //used in conjunction with theta_chk, if >2*pi, then we say that our size estimate is valid
  unsigned int i;			   //counter


  c_theta = 0;								//initialize our theta
  c_r = initial_r;
  //loop until we reach the maximum # of iterations OR if we find a size
  i = 0;										//init counter
  theta_chk = false;							//no size candidate initially
  c_intensity = 0; 	c_x = 0; 	c_y = 0;	//start with some initial values
  while (i < max_iter && !found_size)
  {
    //convert from polar to cartesian
    c_x = c_r*std::cos(c_theta);
    c_y = c_r*std::sin(c_theta);
    pmap.generate(c_x,c_x,1,c_y,c_y,1);	//use pmap to generate a IQUV value at our c_x,c_y point in the camera
    c_intensity = pmap.I(0,0);			//find our intensity at x,y
    if (c_intensity > max_intensity)	//if we have a larger intensity at a point further out
    {
      max_intensity = c_intensity;	//set our "new" max intensity to compare with our percentage limit
    }
    else if(max_intensity*percent_from_max_I >= c_intensity) //we reached our limit/criterion?
    {
      //must check to see if we are in "theta_chk" mode
      if (theta_chk)	//if we are..
      {
	if (c_theta-theta_chk_start >= 2*M_PI)	//we've completed one rev to scan for any higher intensity
	{
	  found_size = true;		//we assume our max size is a good estimate
	}
      }
      else			//start a new theta_chk routine
      {
	theta_chk = true;			//start "theta check" routine
	theta_chk_start = c_theta;	//record our theta (and accept the max size if we go +2pi distance)
	max_size = c_r;				//assume our maximum size is our current r value
      }
    }
    else	//keep going!
    {
      theta_chk = false;	//we stop our theta_chk routine
    }
    
    //calculate our new radius and theta position
    //keep dtheta fixed and approximate our log spiral
    c_theta = ((double)(i+1))*theta_step;
    // c_r = a*std::exp((std::pow(theta_step,2.0))*((double)(i+1))/2.0/M_PI);
    c_r = initial_r*std::exp(theta_step*c_theta/(2.0*M_PI));
    i = i + 1;

    /*
    std::cerr << "model_image_riaf::estimate_image_size_log_spiral : "
	      << std::setw(15) << max_intensity
	      << std::setw(15) << c_intensity
	      << std::setw(15) << c_r
	      << std::setw(15) << c_theta
	      << '\n';
    */
  }
  
  if (found_size)		//did we find a size estimate?
  {
    return max_size * safety_factor;
  }
  else
  {
    //we weren't able to find a size after <max_iter> iterations
    return -1;		//return -1 to indicate this
  }
}


  

};

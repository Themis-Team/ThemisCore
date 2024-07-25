/*!
  \file model_polarized_image_general_riaf.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements the RIAF image model class.
  \details To be added
*/

#include "model_polarized_image_general_riaf.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include "constants.h"
#include "stop_watch.h"

namespace Themis {

model_polarized_image_general_riaf::model_polarized_image_general_riaf(double frequency, double D)
  : _comm(MPI_COMM_WORLD), _D(D), _frequency(frequency), _Rmax(15.0), _Nray_base(128), _Nray(128), _number_of_refines(0)
{
  open_error_streams();
}

void model_polarized_image_general_riaf::open_error_streams()
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

void model_polarized_image_general_riaf::set_screen_size(double imgR)
{
  _Rmax=imgR;

  int rank;
  MPI_Comm_rank(_comm,&rank);
  std::cout << "model_polarized_image_general_riaf: Rank " << rank << " using screen size " << _Rmax << std::endl;

}


void model_polarized_image_general_riaf::set_image_resolution(int Nray, int number_of_refines)
{
  _Nray_base=Nray;
  _Nray = _Nray_base;
  _number_of_refines = number_of_refines;
  int rank;
  MPI_Comm_rank(_comm,&rank);
  //std::cout << "model_polarized_image_general_riaf: Rank " << rank << " using image resolution " << _Nray << std::endl;
  
  //if (number_of_refines!=0)
  //  std::cout << "model_polarized_image_general_riaf: VRT2 refine turned on! Refining base map " 
  //            << number_of_refines << " times on rank " << rank << std::endl;
}
  
std::string model_polarized_image_general_riaf::model_tag() const
{
  std::stringstream tag;
  tag << "model_polarized_image_general_riaf " << _frequency << " " << _D;
  return tag.str();
}
  
void model_polarized_image_general_riaf::generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
{
  int myrank;
  MPI_Comm_rank(_comm,&myrank);
  Themis::StopWatch sw;
  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_polarized_image_general_riaf::generate_polarized_image started at: ";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.start();
  }
  // Assign names to given parameters
  const double mass_cm = parameters[0]*constants::G*constants::Msun/(constants::c*constants::c);
  const double a = parameters[1];//std::min(std::max(0.0,parameters[1]),0.998);      // Spin magnitude
  //const double THETA = parameters[1];  // Inclination Angle
  const double THETA = std::acos(std::min(std::max(parameters[2],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle

  //  "Thermal" electron density normalization, radial power-law index, and h/r
  const double ne_norm = parameters[3];
  const double ne_index = parameters[4];
  const double ne_height = parameters[5];
  //  "Thermal" electron temperature normalization, radial power-law index, and h/r
  const double Te_norm = parameters[6];
  const double Te_index = parameters[7];
  const double Te_height = 1.0; // Not used!
  //  "Nonthermal" electron density normalization, radial power-law index, and h/r
  const double nnth_norm = parameters[8];
  const double nnth_index = parameters[9];
  const double nnth_height = parameters[10];
  const double nnth_alpha = 1.25;
  const double nnth_gammamin = 100;
  //  Plasma beta for the magnetic field
  const double B_beta = 10;
  //Infall parameter
  const double infallR = parameters[11];
  //  Sub-Keplerian Fraction
  const double subk_fac = parameters[12];



  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();

  // Choose when to stop rays
  VRT2::StopCondition stop(g,100,1.01);

  // Choose accretion flow velocity
  VRT2::AFV_ShearingInflow afv(g,risco,infallR,subk_fac); // switches to free-fall at ISCO
  

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

  double r_grid = _Rmax;
  double delta = 1e-4;
	
  pmap.generate(-r_grid-delta,r_grid,_Nray_base,-r_grid-delta,r_grid,_Nray_base);
  pmap.integrate();

  for ( int ir = 0; ir < _number_of_refines; ir++)
    pmap.refine();

  int Nray_x = pmap.xi_size();
  int Nray_y = pmap.eta_size();
  _Nray = std::min(Nray_x,Nray_y);
  _flux = pmap.I_int();
  
 
  double Mtorad = mass_cm/_D;
  I.resize(_Nray); alpha.resize(_Nray); beta.resize(_Nray);
  Q.resize(_Nray);U.resize(_Nray);V.resize(_Nray);

  if (!std::isnan(pmap.I_int()))
  {
    for (int i=0; i<_Nray; i++)
    {
        I[i].resize(_Nray); alpha[i].resize(_Nray); beta[i].resize(_Nray);
        Q[i].resize(_Nray);U[i].resize(_Nray);V[i].resize(_Nray);
        for (int j=0; j<_Nray; j++)
        {
          I[i][j] = std::max(pmap.I(i,j),0.0) / std::pow(Mtorad*(2.0*r_grid-delta)/double(_Nray-1),2);
          Q[i][j] = pmap.Q(i,j)*I[i][j];
          U[i][j] = pmap.U(i,j)*I[i][j];
          V[i][j] = pmap.V(i,j)*I[i][j];
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
          Q[i][j] = 0.0; 
          U[i][j] = 0.0;
          V[i][j] = 0.0;
          alpha[i][j] = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(i) - r_grid-delta );
          if (THETA>0)
            beta[i][j]  = Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
          else
	    beta[i][_Nray-1-j]  = -Mtorad*( (2.0*r_grid-delta)/double(_Nray-1)*double(j) - r_grid-delta );
        }
    }

  }


  if (myrank==0 && _merr.is_open()) {
    _merr << "------------ model_polarized_image_general_riaf::generate_image finished at:";
    for (size_t j=0; j<parameters.size(); ++j)
      _merr << std::setw(15) << parameters[j];
    _merr << std::endl;
    sw.print_time(_merr,"    Total Time:");
  }
}

double model_polarized_image_general_riaf::generate_flux_estimate(double accuracy, std::vector<double> parameters)
{
  /////////////////////////////////////////////////////////
  // Set various parameters:
  // Assign names to given parameters
  const double mass_cm = parameters[0]*Themis::constants::G*Themis::constants::Msun/(Themis::constants::c*Themis::constants::c);
  const double a = std::min(std::max(0.0,parameters[1]),0.998);      // Spin magnitude
  //const double THETA = parameters[1];  // Inclination Angle
  const double THETA = std::acos(std::min(std::max(parameters[2],-0.9999999),0.9999999)) * 180.0/M_PI;  // Inclination Angle

  //  "Thermal" electron density normalization, radial power-law index, and h/r
  const double ne_norm = parameters[3];
  const double ne_index = parameters[4];
  const double ne_height = parameters[5];
  //  "Thermal" electron temperature normalization, radial power-law index, and h/r
  const double Te_norm = parameters[6];
  const double Te_index = parameters[7];
  const double Te_height = 1.0; // Not used!
  //  "Nonthermal" electron density normalization, radial power-law index, and h/r
  const double nnth_norm = parameters[8];
  const double nnth_index = parameters[9];
  const double nnth_height = parameters[10];
  const double nnth_alpha = 1.25;
  const double nnth_gammamin = 100;
  //  Plasma beta for the magnetic field
  const double B_beta = 10;
  //Infall parameter
  const double infallR = parameters[11];
  //  Sub-Keplerian Fraction
  const double subk_fac = parameters[12];



  // Choose metric
  VRT2::Kerr g(1.0,a);
  double risco = g.rISCO();

  // Choose when to stop rays
  VRT2::StopCondition stop(g,100,1.01);

  // Choose accretion flow velocity
  //VRT2::AFV_Keplerian afv(g,risco); // switches to free-fall at ISCO
  VRT2::AFV_ShearingInflow afv(g,risco,infallR,subk_fac); // switches to free-fall at ISCO
  

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

  //std::cerr << "model_polarized_image_general_riaf::generate_flux_estimate : generated first image " << I_new << "\n";

  do                                  //now we keep refining until certain conditions are met
  {
    I_old = I_new;                    //set old intensity/flux
    pmap.refine(refine_ratio);        //refine our map
    pmap.integrate();                 //find new flux
    I_new = pmap.I_int();             //set new flux
    refinement_level++;               //increment our current level of refinement

    
    /*std::cerr << "model_polarized_image_general_riaf::generate_flux_estimate : " << refinement_level
	      << std::setw(15) << I_old 
	      << std::setw(15) << I_new << "\n";*/
    
  } 
  while ( (refinement_level<=number_of_refines) && (std::fabs(I_new-I_old)>accuracy || std::fabs(I_new-I_old)>0.05*std::fabs(I_new+I_old)) ); // There is a minimum accuracy that is being specified here that may need to be readdressed at a later date.

  return I_new;
}

double model_polarized_image_general_riaf::estimate_image_size_log_spiral(VRT2::PolarizationMap &pmap, double initial_r, double theta_step, 
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
    std::cerr << "model_polarized_image_general_riaf::estimate_image_size_log_spiral : "
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

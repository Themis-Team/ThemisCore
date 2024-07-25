/*!
  \file likelihood_matched_filter.cpp
  \author Paul Tiede
  \date  April, 2017
  \brief Implementation file for the matched filter likelihood class
*/

#include "likelihood_matched_filter.h"
#include <iomanip>

namespace Themis
{

likelihood_matched_filter::likelihood_matched_filter(double lreg, double treg, double I_clip,  std::string image_fname, std::string README_fname, std::string image_type)
  :_lreg(lreg),_treg(treg),_I_clip(I_clip)
{

  std::cout << "Setting compression factor to " << treg <<std::endl;
  if (image_type=="GRMHD"){
    std::cout << "Reading in GRMHD image type\n";
    GRMHD_read(image_fname, README_fname);
  }
  else if (image_type=="test"){
    std::cout << "Using test image\n";
    test_read();
  }
  else if (image_type=="SMILI"){
    std::cout << "Reading in SMILI image type\n";
    SMILI_read(image_fname, README_fname);
  }
  else if (image_type=="hpu"){
    std::cout << "Reading in hpu image type\n";
    hpu_read(image_fname);
  }
  else{
    std::cerr << "Image type not specified\n"
              << "Use one of: \n"
              << "   - GRMHD\n"
              << "   - test\n"
              << "   - SMILI\n"
              << "   - hpu\n";
    std::exit(1);
  }

  
}
  
void likelihood_matched_filter::set_mpi_communicator(MPI_Comm comm)
{
  _comm=comm;
  initialize_mpi();
}

double likelihood_matched_filter::find_min_distance_square(double a, double b, double x, double y, double eps, int nmax)
{
  //First check if x,y if close to zero
  if ( x*x + y*y < 1e-8 ){
    return std::min(b*b, a*a);
  }
  //Redefine point x,y so that they live in the first quadrant for simplicity
  double px = std::fabs(x);
  double py = std::fabs(y);

  double radEll = (x/a)*(x/a) + (y/b)*(y/b);
  double t;
  if (radEll < 1)
    t = std::atan2(px,py);
  else
    t = std::atan2(py,px);
  //t = M_PI/4.0;

  double err = 1.0;
  int N = 0;
  double xMin = a*std::cos(t);
  double yMin = b*std::sin(t);
  //Find the minimum
  while (err > eps){

    double xR = xMin;
    double yR = yMin;

    double eX = (a*a - b*b)*std::cos(t)*std::cos(t)*std::cos(t)/a;
    double eY = (b*b - a*a)*std::sin(t)*std::sin(t)*std::sin(t)/b;

    double rx = xR - eX;
    double ry = yR - eY;

    double qx = px - eX;
    double qy = py - eY;

    //Find radius
    double r = std::sqrt(rx*rx + ry*ry);
    double q = std::sqrt(qx*qx + qy*qy);

    double deltaC = r*std::asin((rx*qy - ry*qx)/(r*q));
    double deltaT = deltaC/std::sqrt(a*a + b*b - xR*xR - yR*yR + 1e-20);

    t += deltaT;
    t = std::min(M_PI/2.0, std::max(0.0,t));
    xMin = a*std::cos(t);
    yMin = b*std::sin(t);
    err = std::sqrt( (xMin - xR)*(xMin - xR) + 
                     (yMin - yR)*(yMin - yR) );

    /*
    std::cerr << std::setw(15) << xR
              << std::setw(15) << yR
              << std::setw(15) << xMin
              << std::setw(15) << yMin
              << std::setw(15) << t
              << std::setw(15) << deltaT
              << std::setw(15) << err << std::endl;
  */
     
   
    N+=1;
      if (N>nmax){
        std::cerr << "likelihood_matched_filter.cpp: Max number of iterations reached, answer may be bunk\n" << N << std::endl;
        std::cerr << "x = " << x << std::endl
                  << "y = " << y << std::endl
                  << "err = " << err << std::endl;
        return (xMin-px)*(xMin-px) + (yMin-py)*(yMin-py);
      }
  }

  return (xMin-px)*(xMin-px)+(yMin-py)*(yMin-py);
  
}

double likelihood_matched_filter::operator()(std::vector<double>& x)
{
  //Find gaussian ring model
  std::vector<std::vector<double> > model = gaussian_ring_model(x);
  double sum = 0.0;
  double ptot = 0.0;
  double Itot = 0.0;
  double dx = _alpha[1][1]-_alpha[0][0];
  for (size_t i = 0; i < model.size(); ++i)
    for (size_t j = 0; j < model[0].size(); ++j)
    {
        sum += std::sqrt(model[i][j]*_I[i][j])*dx*dx;
        ptot += model[i][j]*dx*dx;
        Itot += _I[i][j]*dx*dx;
      
    }

  /*
  for ( size_t ii = 0; ii < x.size(); ii++){
    std::cerr << std::setw(15) << x[ii];
  }
  std::cerr << std::endl;
  std::cerr << "ptot " << ptot << std::endl;
  std::cerr << "Itot " << Itot << std::endl;
  std::cerr << "sum " << sum << std::endl;
  */

  return _lreg*std::log(sum);
}

double likelihood_matched_filter::chi_squared(std::vector<double>& x)
{
  //std::cout << "Flux Chi2: " << ( -2.0*operator()(x) ) << std::endl;
  return ( -2.0*operator()(x) );
}

void likelihood_matched_filter::output(std::ostream& out)
{
};

std::vector<std::vector<double> > likelihood_matched_filter::gaussian_ring_model(std::vector<double> parameters)
{

  //First I will construct the ring where the slash is along the x-axis
  double sigma = parameters[1]/(2*std::sqrt(2*std::log(2)));
  double r0 = parameters[0]/2.0;
  double ecc = parameters[2];
  double phi0 = parameters[3];
  double f = parameters[4];
  double a,b;
  if ( ecc >= 0 ){
    a = r0;
    b = r0*std::sqrt(1-ecc*ecc);
  }
  else{
    a = r0*std::sqrt(1-ecc*ecc);
    b = r0;
  }
  double xC = parameters[5];
  double yC = parameters[6];
  std::vector<std::vector<double> > Iring;
  Iring.resize(_I.size());
  for ( size_t i = 0; i < _I[0].size(); ++i )
    Iring[i].resize(_I[0].size());
  
  double sum = 0;
  for ( size_t i =0; i< _alpha.size(); ++i )
    for (size_t j = 0; j<_alpha[0].size(); ++j)
    {
      double x = (_alpha[i][j]-xC)*std::cos(phi0) + (_beta[i][j]-yC)*std::sin(phi0);
      double y = -(_alpha[i][j]-xC)*std::sin(phi0) + (_beta[i][j]-yC)*std::cos(phi0);

      double dr2 = find_min_distance_square(a,b,x,y);

      
      
      //Find the slash
      double A;
      if ( x > -a )
        A = std::max( 1.0-f*(x+a)/(2*a), 0.0);
      else
        A = 1;
      

      //Azimuthally symmetric slash
      //double phi = std::atan2(y,x) + M_PI;
      //double A = 1.0 - f*(1.0-std::fabs(M_PI-phi)/M_PI); 

      Iring[i][j] = A*std::exp(-0.5*dr2/(sigma*sigma)) + 1e-50;
      sum += Iring[i][j];
    }
  
  //Normalize the ring
  double dalpha = _alpha[1][1]-_alpha[0][0];
  double dbeta = _beta[1][1]-_beta[0][0];
  double renormalization_factor = 1.0/( sum*(dalpha)*dbeta);
  for (size_t i=0; i<_alpha.size(); i++)
    for (size_t j=0; j<_alpha[0].size(); j++)
      Iring[i][j] *= renormalization_factor;
  
  return Iring;
 
}

void likelihood_matched_filter::GRMHD_read(std::string image_file_name, std::string README_file_name)
{
  /////////////////////////
  // Get the image particulars (masses, fov, etc.)
  std::ifstream rin(README_file_name.c_str());
  if (!rin.is_open())
  {
    std::cerr << "model_image_score: Could not open " << README_file_name << "\n";
    std::exit(1);
  }
  double frequency;
  std::string stmp;
  rin.ignore(4096,'\n'); // Name
  rin.ignore(4096,'\n'); // Cadence
  rin >> stmp >> stmp >> frequency;
  rin.ignore(4096,'\n');
  // Get the fov's
  double fovx, fovy;
  rin >> stmp >> stmp >> fovx;
  rin.ignore(4096,'\n');
  rin >> stmp >> stmp >> fovy;
  rin.ignore(4096,'\n');
  // Get the dimensions
  size_t Nx, Ny;
  rin >> stmp >> stmp >> Nx;
  rin.ignore(4096,'\n');
  rin >> stmp >> stmp >> Ny;
  rin.ignore(4096,'\n');

  rin.ignore(4096,'\n'); // Spin

  // Get the mass in Msun
  double M,D;
  rin >> stmp >> stmp >> M;
  rin.ignore(4096,'\n');
  // Get the distance in pc
  rin >> stmp >> stmp >> D;
  rin.ignore(4096,'\n');

  
  std::cout << image_file_name << std::endl; 
    
  ////////////////////////////
  // Read in the image (ONLY READS IN INTENSITY FOR NOW)
  std::ifstream iin(image_file_name.c_str());
  if (!iin.is_open())
  {
    std::cerr << "model_image_score: Could not open " << image_file_name << "\n";
    std::exit(1);
  }
    
  _I.resize(Nx);
  _alpha.resize(Nx);
  _beta.resize(Nx);
  for (size_t i=0; i<Nx; i++)
  {
    _I[i].resize(Ny);
    _alpha[i].resize(Ny);
    _beta[i].resize(Ny);
  }

  // Renorm I to 1 Jy

  double dtmp;
  size_t ix,iy;
  double ix2rad = fovx/(Nx);
  double iy2rad = fovy/(Ny);
  double Imax = -1;
  double dItmp;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      iin >> ix >> iy;
      _alpha[ix][iy] = ix2rad * ( ix - 0.5*Nx );
      _beta[ix][iy] = iy2rad * ( iy - 0.5*Nx );
      iin >> dtmp >> dItmp;
      _I[ix][iy] = dItmp;
      if (Imax < dItmp)
        Imax = dItmp;
      iin.ignore(4096,'\n');
      }
  iin.close();
  
  if ( Imax == -1){
    std::cerr << "Well you done goofed\n";
    std::exit(1);
  }
  
  double Itotal=0.0;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      if (_I[i][j]/Imax < _I_clip)
        _I[i][j] = 0.0;
      else{
        _I[i][j] = std::pow(_I[i][j]/Imax, 1.0/_treg);
      }
      Itotal += _I[i][j];
    }

  double renormalization_factor = 1.0/( Itotal*(ix2rad*iy2rad) );
  //double renormalization_factor = 1.0/( Itotal );
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
      _I[i][j] *= renormalization_factor;

}

void likelihood_matched_filter::test_read()
{
  double xmin = -50;
  double xmax = 50;

  double ymin = -50;
  double ymax = 50;
  int Nx = 500;
  int Ny = 500;

  double dx = (xmax-xmin)/(Nx-1);
  double dy = (ymax-ymin)/(Ny-1);

  std::vector<std::vector<double> > alpha, beta, I;
  _alpha.resize(Nx);
  _beta.resize(Nx);
  _I.resize(Nx);

  for (int i=0; i<Nx; i++)
  {
    _I[i].resize(Ny);
    _alpha[i].resize(Ny);
    _beta[i].resize(Ny);
  }

  for (int i = 0; i < Nx; i++)
    for (int j = 0; j<Ny; j++)
    {
      _alpha[i][j] = xmin + dx*i;
      _beta[i][j] = ymin + dy*j;
    }

  std::vector<double> p;
  p.push_back(40.0);
  p.push_back(5);
  p.push_back(-0.5);
  p.push_back(M_PI/2.0);
  p.push_back(1.0);
  p.push_back(10.0);
  p.push_back(5.0);

  
  _I = gaussian_ring_model(p);  


  std::cerr << _alpha.size() << std::endl;
  for (size_t i = 0; i < _alpha[0].size(); ++i)
    for ( size_t j = 0; j < _alpha.size(); ++j )
      std::cerr << std::setw(15) << _alpha[i][j]
                << std::setw(15) << _beta[i][j]
                << std::setw(15) << _I[i][j] << std::endl;
  

}

void likelihood_matched_filter::SMILI_read(std::string image_fname, std::string README_fname)
{

  ///////////////////////////////////////
  //Get the image particulars (pixel size and number of pixels)
  
  std::ifstream rin(README_fname.c_str());
  if (!rin.is_open())
  {
    std::cerr << "likelihood_matched_filter: Could not open " << README_fname << "\n";
    std::exit(1);
  }

  //Skip the first line since it is a header
  std::string dummy;
  std::getline(rin, dummy); 
  size_t Nx,Ny;
  double dx,dy;
  rin >> Nx;
  rin >> Ny;
  rin >> dx;
  rin >> dy;

  rin.close();

  
  ////////////////////////////
  // Read in the image (ONLY READS IN INTENSITY FOR NOW)
  std::ifstream iin(image_fname.c_str());
  if (!iin.is_open())
  {
    std::cerr << "model_image_score: Could not open " << image_fname << "\n";
    std::exit(1);
  }
  //Now read in the image
  _I.resize(Nx);
  _alpha.resize(Nx);
  _beta.resize(Nx);
  for (size_t i=0; i<Nx; i++)
  {
    _I[i].resize(Ny);
    _alpha[i].resize(Ny);
    _beta[i].resize(Ny);
  }

  size_t ix,iy;
  double dItmp;
  double Imax = -1;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      iin >> ix >> iy;
      _alpha[ix][iy] = dx * ( ix - 0.5*Nx );
      _beta[ix][iy] = dy * ( iy - 0.5*Nx );
      iin >> dItmp;
      if ( Imax < dItmp)
        Imax = dItmp; 
      _I[ix][iy] = dItmp;
      
    }
  if ( Imax == -1){
    std::cerr << "Well you done goofed\n";
    std::exit(1);
  }
  iin.close();

  //Now renormalize the image so that its maximum is unity
  //and apply the compression factor
  double Itotal=0.0;
  std::cout << Imax << std::endl;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      if (_I[i][j]/Imax < _I_clip)
        _I[i][j] = 0.0;
      else{
        _I[i][j] = std::pow(_I[i][j]/Imax, 1.0/_treg);
        Itotal += _I[i][j];
      }
      //std::cerr << std::setw(15) << _alpha[i][j]
      //          << std::setw(15) << _beta[i][j]
      //          << std::setw(15) << _I[i][j] << std::endl;
    }
  std::cout << Itotal << std::endl;
  std::cout << _I_clip << std::endl;
  double renormalization_factor = 1.0/( Itotal*(dx*dy) );
  //double renormalization_factor = 1.0/( Itotal );
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
      _I[i][j] *= renormalization_factor;

  
  
}


void likelihood_matched_filter::hpu_read(std::string image_fname)
{

  ///////////////////////////////////////
  //Get the image particulars (pixel size and number of pixels)
  
  std::ifstream rin(image_fname.c_str());
  if (!rin.is_open())
  {
    std::cerr << "likelihood_matched_filter: Could not open " << image_fname << "\n";
    std::exit(1);
  }

  //Skip the first line since it is a header
  std::string dummy;
  rin.ignore(4096,'\n'); //SRC
  rin.ignore(4096,'\n'); //RA
  rin.ignore(4096,'\n'); //DEC
  rin.ignore(4096,'\n'); //MJD
  rin.ignore(4096,'\n'); //RF
  size_t Nx,Ny;
  double dx,dy;
  rin >> dummy >> dummy >> Nx >> dummy >> dx;
  rin.ignore(4096,'\n'); 
  rin >> dummy >> dummy >> Ny >> dummy >> dy;

  
  
  
  //convert to uas
  dx = 1e6*dx/Nx;
  dy = 1e6*dy/Ny;

  std::cout << "Nx: " << Nx << std::endl
            << "Ny: " << Ny << std::endl
            << "dx: " << dx << std::endl
            << "dy: " << dy << std::endl;
  
  std::getline(rin,dummy);
  std::getline(rin,dummy);
  std::getline(rin,dummy);

  //Now read in the image
  _I.resize(Nx);
  _alpha.resize(Nx);
  _beta.resize(Nx);
  for (size_t i=0; i<Nx; i++)
  {
    _I[i].resize(Ny);
    _alpha[i].resize(Ny);
    _beta[i].resize(Ny);
  }

  double xtemp,ytemp;
  double dItmp;
  double Imax = -1;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      rin >> xtemp >> ytemp >> dItmp;
    
      _alpha[i][j] = xtemp*1e6;
      _beta[i][j] = ytemp*1e6; 
      _I[i][j] = dItmp;
      if ( Imax < dItmp)
        Imax = dItmp; 
      
      rin.ignore(4096,'\n');
      /*
      std::cerr << std::setw(15) << _alpha[i][j]
                << std::setw(15) << _beta[i][j]
                << std::setw(15) << _I[i][j] << std::endl;
      */
    }
  if ( Imax == -1){
    std::cerr << "Well you done goofed\n";
    std::exit(1);
  }
  rin.close();

  //Now renormalize the image so that its maximum is unity
  //and apply the compression factor
  double Itotal=0.0;
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
    {
      if (_I[i][j]/Imax < _I_clip)
        _I[i][j] = 0.0;
      else{
        _I[i][j] = std::pow(_I[i][j]/Imax, 1.0/_treg);
        Itotal += _I[i][j];
      }

      /*
      std::cerr << std::setw(15) << _alpha[i][j]
                << std::setw(15) << _beta[i][j]
                << std::setw(15) << _I[i][j] << std::endl;
     */ 
      
    }

  std::cout << Imax << std::endl;
  std::cout << Itotal << std::endl;
  double renormalization_factor = 1.0/( Itotal*(dx*dy) );
  //double renormalization_factor = 1.0/( Itotal );
  for (size_t i=0; i<Nx; i++)
    for (size_t j=0; j<Ny; j++)
      _I[i][j] *= renormalization_factor;

  
  
}




}

#include "diskmap.h"

namespace VRT2 {

#define CPV(x,y) ( y!=0 ? (x)/(y) : 0.0 )
#define MAX(x,y) ( x<y ? y : x )

// Define if there are mpi buffer issues.  Performs one send/recv per
//  point per quantity.  Otherwise, reduces the whole thing at once.
//#define MPI_BUFF_LIMITS

DiskMap::DiskMap(Metric& g, Ray& ray, SC_DiskMap& stop, int verbosity)
  : _g(g), _ray(ray), _stop(stop), _verbosity(verbosity), _frequency0(1.0)
{
  set_R_THETA(500.0,30.0);
  set_progress_stream(std::cerr);

#ifdef MPI_MAP
  _rank = MPI::COMM_WORLD.Get_rank();
  _size = MPI::COMM_WORLD.Get_size();
#else
  _rank = 0;
  _size = 1;
#endif
}

void DiskMap::set_R_THETA(double R, double THETA) // Theta in degrees
{
  _R = R;
  _THETA = THETA*VRT2_Constants::pi/180.0;
}

void DiskMap::set_progress_stream(std::ostream& pstream)
{
  _pstream = &pstream;
}

void DiskMap::set_progress_stream(std::string pbase)
{
#ifndef MPI_MAP
  if(_pstream==&_lpstream)
    _lpstream.close();
  _lpstream.open(pbase.c_str(), std::ios::out);
  _pstream = &_lpstream;
#else
  std::stringstream pname;
  pname << pbase << '.' << MPI::COMM_WORLD.Get_rank();
  if (_pstream==&_lpstream)
    _lpstream.close();
  _lpstream.open(pname.str().c_str());
  _pstream = &_lpstream;
#endif
}

void DiskMap::generate(double xi_lo,double xi_hi,int N_xi,
			       double eta_lo,double eta_hi,int N_eta)
{
  generate(_frequency0,xi_lo,xi_hi,N_xi,eta_lo,eta_hi,N_eta);
}

void DiskMap::generate(double frequency,
			       double xi_lo,double xi_hi,int N_xi,
			       double eta_lo,double eta_hi,int N_eta)
{
  /*** Assign frequency ***/
  _frequency = frequency;

  /*** Assign Limits ***/
  _xi_lo = xi_lo;
  _xi_hi = xi_hi;
  _N_xi = N_xi; 
  _eta_lo = eta_lo;
  _eta_hi = eta_hi;
  _N_eta = N_eta;

  /*** Resize vectors (implicit slicing) ***/
  int dim=_N_xi*_N_eta;
  _tau.resize(dim);
  _D.resize(dim);
  _I.resize(dim);
  _Q.resize(dim);
  _U.resize(dim);
  _V.resize(dim);

  /*** Zero vectors ***/
  _tau.assign(dim,0.0);
  _D.assign(dim,0.0);
  _I.assign(dim,0.0);
  _Q.assign(dim,0.0);
  _U.assign(dim,0.0);
  _V.assign(dim,0.0);


  // eta and xi stepsize
  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  // eta and xi declaration
  double xi, eta;
  // slicing index
  int index;
  // individual iquv vectors
  std::valarray<double> iquv(0.0,4);
  // Initial values      
  FourVector<double> x0(_g), k0(_g);

  //  if (_rank==0)
  //  std::cerr << "generate: N_to_I = " << N_to_I << std::endl;

  /*** Progress indicator start ***/
  ProgressCounter pc( (*_pstream) );
  (*_pstream) << "Done with ";
  pc.start();
  std::string backup(3+4+1+4+1,'\b');

  /*** Begin Loop Over Region ***/
  for (int i_eta=0; i_eta<_N_eta; ++i_eta){
    eta = _eta_lo + i_eta*eta_step;
    for (int i_xi=0; i_xi<_N_xi; ++i_xi){
      xi = _xi_lo + i_xi*xi_step;
      
      // Slicing index
      index = i_eta*_N_xi+i_xi;

      if ( (index-_rank)%_size == 0 ) {
	(*_pstream) << "  ("
		    << std::setw(4) << i_xi
		    << ','
		    << std::setw(4) << i_eta
		    << ')';

	
	init_conds(_frequency,xi,eta,x0,k0);
	
	// Propagate
	_stop.reset();
	_ray.reinitialize(x0,k0);
	_ray.propagate(1.0,"!");
	
	
	// Get polarization
	iquv = _ray.IQUV(); 
	
	_tau[index] = _ray.tau();
	_D[index] = _ray.D();

	_I[index] = iquv[0];
	_Q[index] = iquv[1];
	_U[index] = iquv[2];
	_V[index] = iquv[3];
	
	// Progress indicator increment
	(*_pstream) << backup;
	pc.increment(double(index+1)/double(dim));
      }
    }
  }
  (*_pstream) << backup;
  pc.finish();
  (*_pstream) << "  ("
	      << std::setw(4) << _N_xi
	      << ','
	      << std::setw(4) << _N_eta
	      << ')';

#ifdef MPI_MAP
  collect(0);
#endif
}


void DiskMap::eta_section(double eta_hi, double xi_lo, double xi_hi, int N_xi)
{
  eta_section(_frequency0,eta_hi,xi_lo,xi_hi,N_xi);
}

void DiskMap::eta_section(double frequency, double eta_hi, double xi_lo, double xi_hi, int N_xi)
{
  /*** Assign frequency ***/
  _frequency = frequency;

  /*** Assign Limits ***/
  _xi_lo = xi_lo;
  _xi_hi = xi_hi;
  _N_xi = N_xi; 
  _N_eta = 1;
  _eta_hi = eta_hi;  // Use high levels to denote the static slice value

  /*** Resize vectors (implicit slicing) ***/
  int dim=_N_xi*_N_eta;
  _tau.resize(dim);
  _D.resize(dim);
  _I.resize(dim);
  _Q.resize(dim);
  _U.resize(dim);
  _V.resize(dim);

  /*** Open smfile ***/
  std::ofstream smfile("../data_vis/smfiles/eta_section.sm");  
  smfile << "ctype default\n"
	 << "ltype 0\n"
	 << "expand 1\nwindow 1 1 1 1\n"
	 << "lweight 2\n"
	 << "location 3700 31000 3700 31000\n\n"
	 << "define Sqr_Size 20 !plot size\n\n"
	 << "expand 1.5\n"
	 << "limits -$Sqr_Size $Sqr_Size -$Sqr_Size $Sqr_Size\n"
	 << "box\n"
	 << "expand 2\n"
	 << "ylabel x(M)\nxlabel y(M)\n"
	 << "expand 1\n"
	 << "! Draw the ergosphere\n"
	 << "set phi=0,6.3,0.1\n"
	 << "set cth=cos(phi)*cos(" << _THETA << ") \n"
	 << "set rergo=" << _g.mass()
	 << "+sqrt(" << _g.mass() << "**2 - "
	 << _g.ang_mom() << "**2 * cth**2 )\n"
	 << "lweight 1\n"
	 << "set x=rergo*cos(phi)\n"
	 << "set y=rergo*sin(phi)\n"
	 << "angle 45\n"
	 << "shade 300 y x\n"
	 << "angle 0\n"
	 << "connect y x"
	 << "! Draw horizon\n"
	 << "set x=" << _g.horizon() << "*cos(phi)\n"
	 << "set y=" << _g.horizon() << "*sin(phi)\n"
	 << "shade 0 y x\n\n"
	 << "lweight 3\n";

  /*** Open matlab file ***/
  std::ofstream mfile("../data_vis/matlab_scripts/rays/eta_section.m");
  mfile << "% Clear Figure\n"
	<< "%clf;\n"
	<< "fig=figure;\n"
	<< "set(fig,'color','cyan','name','eta section','numbertitle'"
	<< ",'off','renderer','OpenGL');\n"
	<< "%Plot All Together\n"
	<< "hold on;\n"
	<< "% BH Properties\n"
	<< "M=" << 1 << ";\n"  //_g.mass() << ";\n"
	<< "a=" << _g.ang_mom() << ";\n"
	<< "% Plot the Hole\n"
	<< "[horiz,ergo]=kerr_hole(M,a);\n"
	<< "% Plasma Stuff\n"
	<< "%density_plot3d('../../disk.dat');\n"
	<< "% Read and Plot rays\n";

  // slicing index
  int index;
  // individual iquv vectors
  std::valarray<double> iquv(0.0,4);
  // eta and xi stepsize
  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  // eta and xi declaration
  double xi, eta=_eta_hi;
  // Initial values      
  FourVector<double> x0(_g), k0(_g);

  //std::cout << "N_to_I = " << N_to_I << std::endl;

  // Output File name
  std::string pre="../ray_data/eta_";

  /*** Progress indicator start ***/
  (*_pstream).setf(std::ios::fixed);
  (*_pstream) << std::setprecision(1);
  (*_pstream) << std::endl << std::endl;
  (*_pstream) << "Done with   0.0%";

  /*** Begin Loop Over Region ***/
  for (int i_xi=0; i_xi<_N_xi; i_xi++){
    xi = _xi_lo + i_xi*xi_step;

    // Slicing index
    index = i_xi;
    
    // Get initial values
    init_conds(_frequency,xi,eta,x0,k0);

    // Get output
    std::ostringstream out_name;
    out_name << pre << i_xi << '_';

    // Propagate
    _stop.reset();
    _ray.reinitialize(x0,k0);
    std::vector<std::string> ray_out = _ray.propagate(1.0,out_name.str());

    // Get polarization
    iquv = _ray.IQUV();

    _tau[index] = _ray.tau();
    _D[index] = _ray.D();
    _I[index] = iquv[0];
    _Q[index] = iquv[1];
    _U[index] = iquv[2];
    _V[index] = iquv[3];
            
    for (unsigned int i=0;i<ray_out.size();++i){
      smfile << "data ../" << ray_out[i] << "\nlines 3 100000\n"
	     << "read {x 3 y 4 z 5}\n";
      mfile << "ray=plot_ray('../../" << ray_out[i] << "');\n"
	    << "set(ray,";
      
      smfile << "lweight 2\n"
	     << "ltype 0\n";
      mfile << "'color','red','linewidth',1);\n";
      
      smfile << "connect x y\n\n";
    }
    
    // Progress indicator start
    (*_pstream) << "\b\b\b\b\b\b" << std::setw(5)
	 << 100.0*double(i_xi+1)/double(N_xi) << '%';
    
  }
  (*_pstream) << std::endl << std::endl << std::endl;  
  
  /*** Close SM file ***/
  smfile << "identification\n";
  smfile.close();
  
  /*** Close matlab file ***/
  mfile << "% New Plotting Parameters\n"
	<< "axis([-100 100 -100 100 -100 100]);\n"
	<< "axis('off','equal');\n"
	<< "camtarget([0 0 1]);\n"
	<< "%camproj('perspective');\n"
	<< "camproj('orthographic');\n"
	<< "% Next plotting command will erase the plot\n"
	<< "hold off;";
  mfile.close();
}


void DiskMap::xi_section(double xi_hi, double eta_lo, double eta_hi, int N_eta)
{
  xi_section(_frequency0,xi_hi,eta_lo,eta_hi,N_eta);
}

void DiskMap::xi_section(double frequency, double xi_hi, double eta_lo, double eta_hi, int N_eta)
{
  /*** Assign frequency ***/
  _frequency = frequency;

  /*** Assign Limits ***/
  _eta_lo = eta_lo;
  _eta_hi = eta_hi;
  _N_xi = 1;
  _N_eta = N_eta; 
  _xi_hi = xi_hi;  // Use high levels to denote the static slice value

  /*** Resize vectors (implicit slicing) ***/
  int dim=_N_xi*_N_eta;
  _tau.resize(dim);
  _D.resize(dim);
  _I.resize(dim);
  _Q.resize(dim);
  _U.resize(dim);
  _V.resize(dim);

  /*** Open smfile ***/
  std::ofstream smfile("../data_vis/smfiles/xi_section.sm");
  smfile << "ctype default\n"
	 << "expand 1.0001\nwindow 1 1 1 1\n"
	 << "lweight 2\n"
	 << "location 3500 31000 3500 31000\n\n"
	 << "define Sqr_Size 30 !plot size\n\n"
	 << "limits -$Sqr_Size $Sqr_Size -$Sqr_Size $Sqr_Size\n"
	 << "box\nylabel x(M)\nxlabel y(M)\n"
	 << "! Draw the ergosphere\n"
	 << "set theta=0,6.3,0.1\n"
	 << "set cth=cos(theta)\n"
	 << "set rergo=" << _g.mass()
	 << "+sqrt(" << _g.mass() << "**2 - "
	 << _g.ang_mom() << "**2 * cth**2 )\n"
	 << "lweight 1\n"
	 << "set x=rergo*cos(theta)\n"
	 << "set y=rergo*sin(theta)\n"
	 << "shade 300 y x\n"
	 << "connect y x"
	 << "! Draw horizon\n"
	 << "set x=" << _g.horizon() << "*cos(theta)\n"
	 << "set y=" << _g.horizon() << "*sin(theta)\n"
	 << "shade 0 y x\n\n"
	 << "lweight 3\n";

  /*** Open matlab file ***/
  std::ofstream mfile("../data_vis/matlab_scripts/rays/xi_section.m");
  mfile << "% Clear Figure\n"
	<< "%clf;\n"
	<< "fig=figure;\n"
	<< "set(fig,'color','cyan','name','xi section','numbertitle','off','renderer','OpenGL');\n"
	<< "%Plot All Together\n"
	<< "hold on;\n"
	<< "% BH Properties\n"
	<< "M=" << _g.mass() << ";\n"
	<< "a=" << _g.ang_mom() << ";\n"
	<< "% Plot the Hole\n"
	<< "[horiz,ergo]=kerr_hole(M,a);\n"
	<< "% Plasma Stuff\n"
	<< "%density_plot3d('../../disk.dat');\n"
	<< "% Read and Plot rays\n";

  // slicing index
  int index;
  // individual iquv vectors
  std::valarray<double> iquv(0.0,5);
  // eta and xi stepsize
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  // eta and xi declaration
  double eta, xi=_xi_hi;
  // Initial values      
  FourVector<double> x0(_g), k0(_g);

  // Output File name
  std::string pre="../ray_data/xi_";

  /*** Progress indicator start ***/
  (*_pstream).setf(std::ios::fixed);
  (*_pstream) << std::setprecision(1);
  (*_pstream) << std::endl << std::endl;
  (*_pstream) << "Done with   0.0%";

  /*** Begin Loop Over Region ***/
  for (int i_eta=0; i_eta<_N_eta; i_eta++){
    eta = _eta_lo + i_eta*eta_step;

    // Slicing index
    index = i_eta*_N_xi;
    
    // Get initial values      
    init_conds(_frequency,xi,eta,x0,k0);

    // Get output
    std::ostringstream out_name;
    out_name << pre << i_eta << '_';

    // Propagate
    _stop.reset();
    _ray.reinitialize(x0,k0);
    std::vector<std::string> ray_out = _ray.propagate(1.0,out_name.str());

    // Get polarization
    iquv = _ray.IQUV();

    _tau[index] = _ray.tau();
    _D[index] = _ray.D();
    _I[index] = iquv[0];
    _Q[index] = iquv[1];
    _U[index] = iquv[2];
    _V[index] = iquv[3];

    for (unsigned int i=0;i<ray_out.size();++i){
      smfile << "data ../" << ray_out[i] << "\nlines 2 100000\n"
	     << "read {x 3 y 4 z 5}\n";
      mfile << "ray=plot_ray('../../" << ray_out[i] << "');\n"
	    << "set(ray,";
      
      smfile << "lweight 2\n"
	     << "ltype 0\n";
      mfile << "'color','red','linewidth',1);\n";
      
      smfile << "connect x z\n\n";
    }

    // Progress indicator start
    (*_pstream) << "\b\b\b\b\b\b" << std::setw(5)
	 << 100.0*double(i_eta+1)/double(N_eta) << '%';
  }
  (*_pstream) << std::endl << std::endl << std::endl;  

  /*** Close SM file ***/
  smfile << "identification\n";
  smfile.close();
  
  /*** Close matlab file ***/
  mfile << "% New Plotting Parameters\n"
	<< "axis([-100 100 -100 100 -100 100]);\n"
	<< "axis('off','equal');\n"
	<< "camtarget([0 0 1]);\n"
	<< "%camproj('perspective');\n"
	<< "camproj('orthographic');\n"
	<< "% Next plotting command will erase the plot\n"
	<< "hold off;";
  mfile.close();
}


/*** Calculates Initial Conditions for x0 on a plane with closest
     approach at _R, _THETA, and phi=0, and k0 perpendicular to
     the plane.  Good to go. ***/

int DiskMap::init_conds(double frequency, double xi, double eta,
		    FourVector<double> &x, FourVector<double> &k)
{
  double x0[4], k0[4];

  // Set x0
  x0[0] = 0.0;
  x0[1] = sqrt( xi*xi + eta*eta + _R*_R );
  x0[2] = std::fabs( std::atan2( std::sqrt(std::pow(_R*std::sin(_THETA)-eta*std::cos(_THETA),2) + xi*xi) ,
     _R*std::cos(_THETA)+eta*std::sin(_THETA) ) );
  x0[3] = std::fmod( std::atan2( xi, _R*std::sin(_THETA)-eta*std::cos(_THETA) ) + 2.0*VRT2_Constants::pi
		     , 2.0*VRT2_Constants::pi);

  // Set metric
  _g.reset(x0);
 
  // Set k0
  k0[0] = -2.0*VRT2_Constants::pi * _frequency0; // covariant component -> going backwards in time
  k0[1] = std::fabs(k0[0]) * ( std::sin(_THETA)*std::sin(x0[2])*std::cos(x0[3])
			       + std::cos(_THETA)*std::cos(x0[2]) );
  k0[2] = - std::fabs(k0[0]) * x0[1] * ( std::cos(_THETA)*std::sin(x0[2])
					 - std::sin(_THETA)*std::cos(x0[2])*std::cos(x0[3]) );
  k0[3] = - std::fabs(k0[0]) * x0[1]*std::sin(x0[2]) * std::sin(_THETA)*std::sin(x0[3]);
 

  x.mkcon(x0);
  k.mkcov(k0);

  return 0;
}  

// Stokes Parameters
double DiskMap::I(int i_xi,int i_eta)
{
  return ( _I[i_eta*_N_xi+i_xi] );
}
double DiskMap::Q(int i_xi,int i_eta)
{
  return ( _Q[i_eta*_N_xi+i_xi] );
}
double DiskMap::U(int i_xi,int i_eta)
{
  return ( _U[i_eta*_N_xi+i_xi] );
}
double DiskMap::V(int i_xi,int i_eta)
{
  return ( _V[i_eta*_N_xi+i_xi] );
}

// Checks and Other Functions
double DiskMap::tau(int i_xi,int i_eta)
{
  return ( _tau[i_eta*_N_xi+i_xi] );
}
double DiskMap::D(int i_xi,int i_eta)
{
  return ( _D[i_eta*_N_xi+i_xi] );
}


void DiskMap::output_map(std::string fname)
{
  if (_rank==0) {
    std::ofstream DiskMap_out(fname.c_str());

    if (!DiskMap_out.is_open())
      std::cerr << "Couldn't open " << fname << " to output map\n";

    output_map(DiskMap_out);
  }
}

void DiskMap::output_map(std::ostream& DiskMap_out)
{
  if (_rank==0) {
    DiskMap_out.setf(std::ios::scientific);
    DiskMap_out << std::setprecision(5);

    // Info for SM
    DiskMap_out << "xi: " << _xi_lo << ' ' << _xi_hi << ' ' << _N_xi << '\n';
    DiskMap_out << "eta: " << _eta_lo << ' ' << _eta_hi << ' ' << _N_eta << '\n';
    
    // Headers
    DiskMap_out << std::setw(7) << "i_xi"
			<< std::setw(7) << "i_eta"
			<< std::setw(15) << "I"
			<< std::setw(15) << "Q"
			<< std::setw(15) << "U"
			<< std::setw(15) << "V"
			<< std::setw(15) << "tau"
			<< std::setw(15) << "D\n";

    for (int i_eta=0; i_eta<_N_eta; ++i_eta){
      for (int i_xi=0; i_xi<_N_xi; ++i_xi){
	DiskMap_out << std::setw(7) << i_xi
			    << std::setw(7) << i_eta
			    << std::setw(15) << I(i_xi,i_eta)
			    << std::setw(15) << Q(i_xi,i_eta)
			    << std::setw(15) << U(i_xi,i_eta)
			    << std::setw(15) << V(i_xi,i_eta)
			    << std::setw(15) << tau(i_xi,i_eta)
			    << std::setw(15) << D(i_xi,i_eta)
			    << '\n';
      }
    }
  }
}
#undef CPV

#ifdef MPI_MAP
void DiskMap::collect(int rank)
{

#ifdef MPI_BUFF_LIMITS
  // Assumes that this process is involved iff (index-_rank)%_size == 0
  MPI::COMM_WORLD.Barrier();
  /*** Begin Loop Over Region ***/
  int index;
  int sender_rank;
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
    for (int i_xi=0; i_xi<_N_xi; ++i_xi) {
      
      // Slicing index
      index = i_eta*_N_xi+i_xi;

      sender_rank = index%_size;

      if (sender_rank != rank) { // Don't self send
	if (_rank == sender_rank) { // If I'm the sender, then send
	  MPI::COMM_WORLD.Send(&_I[index],1,MPI::DOUBLE,rank,20);
	  MPI::COMM_WORLD.Send(&_Q[index],1,MPI::DOUBLE,rank,21);
	  MPI::COMM_WORLD.Send(&_U[index],1,MPI::DOUBLE,rank,22);
	  MPI::COMM_WORLD.Send(&_V[index],1,MPI::DOUBLE,rank,23);
	  MPI::COMM_WORLD.Send(&_tau[index],1,MPI::DOUBLE,rank,24);
	  MPI::COMM_WORLD.Send(&_D[index],1,MPI::DOUBLE,rank,25);
	}
	else if (_rank == rank) { // If I'm the root, then recieve
	  MPI::COMM_WORLD.Recv(&_I[index],1,MPI::DOUBLE,sender_rank,20);
	  MPI::COMM_WORLD.Recv(&_Q[index],1,MPI::DOUBLE,sender_rank,21);
	  MPI::COMM_WORLD.Recv(&_U[index],1,MPI::DOUBLE,sender_rank,22);
	  MPI::COMM_WORLD.Recv(&_V[index],1,MPI::DOUBLE,sender_rank,23);
	  MPI::COMM_WORLD.Recv(&_tau[index],1,MPI::DOUBLE,sender_rank,24);
	  MPI::COMM_WORLD.Recv(&_D[index],1,MPI::DOUBLE,sender_rank,25);
	}
      }
    }
#else
  double *send = new double[_N_xi*_N_eta];
  double *recv = new double[_N_xi*_N_eta];

  // I
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _I[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _I[i + _N_xi*j] = recv[i + _N_xi*j];
  
  // Q
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _Q[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _Q[i + _N_xi*j] = recv[i + _N_xi*j];

  // U
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _U[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _U[i + _N_xi*j] = recv[i + _N_xi*j];

  // V
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _V[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _V[i + _N_xi*j] = recv[i + _N_xi*j];

  // tau
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _tau[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _tau[i + _N_xi*j] = recv[i + _N_xi*j];

  // D
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _D[i + _N_xi*j];
  MPI::COMM_WORLD.Reduce(&send[0], &recv[0], _N_xi*_N_eta, MPI::DOUBLE, MPI::SUM, rank);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _D[i + _N_xi*j] = recv[i + _N_xi*j];
  
  delete[] send;
  delete[] recv;
#endif
}
#endif
#undef MAX
};

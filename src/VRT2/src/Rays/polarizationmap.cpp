#include "polarizationmap.h"


namespace VRT2 {


#define CPV(x,y) ( y!=0 ? (x)/(y) : 0.0 )
#define MAX(x,y) ( x<y ? y : x )

// Define if there are mpi buffer issues.  Performs one send/recv per
//  point per quantity.  Otherwise, reduces the whole thing at once.
//#define MPI_BUFF_LIMITS

int first_leg;

PolarizationMap::PolarizationMap(Metric& g, Ray& ray, double M, double D, int verbosity, bool rescale_intensity)
  : _g(g), _ray(ray), _verbosity(verbosity), _frequency0(1.0),  _BHM(M), _BHD(D), _rI(rescale_intensity)
{
  set_R_THETA(500.0,30.0);
  set_progress_stream(std::cerr);
  
  unset_background_intensity();
  
#ifdef VRT2_USE_MPI_MAP
  // Get the rank and size in the original communicator
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int color = world_rank; // Determine color based on row

  // Split the communicator based on the color and use the
  // original rank for ordering
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &_pmap_communicator);
  _created_pmap_communicator=true;

  MPI_Comm_rank(_pmap_communicator,&_rank);
  MPI_Comm_size(_pmap_communicator,&_size);
#else
  _rank = 0;
  _size = 1;
#endif
}

#ifdef VRT2_USE_MPI_MAP
PolarizationMap::PolarizationMap(Metric& g, Ray& ray, double M, double D, MPI_Comm& comm, int verbosity, bool rescale_intensity)
  : _g(g), _ray(ray), _verbosity(verbosity), _frequency0(1.0),  _BHM(M), _BHD(D), _rI(rescale_intensity), _pmap_communicator(comm)
{
  set_R_THETA(500.0,30.0);
  set_progress_stream(std::cerr);
  
  unset_background_intensity();

  _created_pmap_communicator=false;

  MPI_Comm_rank(_pmap_communicator,&_rank);
  MPI_Comm_size(_pmap_communicator,&_size);
}

PolarizationMap::~PolarizationMap()
{
  if (_created_pmap_communicator)
    MPI_Comm_free(&_pmap_communicator);
}
#endif

void PolarizationMap::set_f0(double frequency0)
{
  _frequency0 = frequency0;
}

void PolarizationMap::set_R_THETA(double R, double THETA) // Theta in degrees
{
  _R = R;
  _THETA = THETA*VRT2_Constants::pi/180.0;
}

void PolarizationMap::set_progress_stream(std::ostream& pstream)
{
  _pstream = &pstream;
}

void PolarizationMap::set_progress_stream(std::string pbase)
{
#ifndef VRT2_USE_MPI_MAP
  if(_pstream==&_lpstream)
    _lpstream.close();
  _lpstream.open(pbase.c_str(), std::ios::out);
  _pstream = &_lpstream;
#else
  std::stringstream pname;
  int rank;
  MPI_Comm_rank(_pmap_communicator,&rank);
  pname << pbase << '.' << rank;
  if (_pstream==&_lpstream)
    _lpstream.close();
  _lpstream.open(pname.str().c_str());
  _pstream = &_lpstream;
#endif
}

double PolarizationMap::get_N_to_I()
{
  if (_rI) {
    double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 1.0);
    double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 1.0);
    double area_pixel = xi_step*eta_step * _BHM*_BHM; // cm^2
    double gc_distance = _BHD; // cm
    double solid_angle = area_pixel/(gc_distance*gc_distance);
    
    // Constant to take N to I in Jy:
    return ( std::pow( 2.0*VRT2_Constants::pi*_frequency, 3 ) // N -> erg/s/cm^2/Hz
	     * 1.0E-7 * 1.0E4            // -> J/s/m^2/Hz
	     * 1.0E26                    // -> Jy_source
	     * solid_angle               // -> Jy_observer
	     );  
  } else {
    return 1.0;
  }
}

void PolarizationMap::generate(double xi_lo,double xi_hi,int N_xi,
			       double eta_lo,double eta_hi,int N_eta)
{
  generate(_frequency0,xi_lo,xi_hi,N_xi,eta_lo,eta_hi,N_eta);
}

void PolarizationMap::generate(double frequency,
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
  _tau.resize(dim,0.0);
  _D.resize(dim,0.0);
  _I.resize(dim,0.0);
  _Q.resize(dim,0.0);
  _U.resize(dim,0.0);
  _V.resize(dim,0.0);

  for (size_t i=0; i<_I.size(); ++i)
  {
    _tau[i] = 0.0;
    _D[i] = 0.0;
    _I[i] = 0.0;
    _Q[i] = 0.0;
    _U[i] = 0.0;
    _V[i] = 0.0;
  }

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
  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();			//???

  //  if (_rank==0)
  //  std::cerr << "generate: N_to_I = " << N_to_I << std::endl;

//// CARLOS EDIT: REMOVE PROGRESS TO SAVE SPACE
//  /*** Progress indicator start ***/
  ProgressCounter pc( (*_pstream) );
  std::string backup(3+4+1+4+1,'\b');
  if (_verbosity>0)
  {
    (*_pstream) << "Done with ";
    pc.start();
  }


  /*** Begin Loop Over Region ***/
  //xi and eta are our respective "positions" on the camera
  for (int i_eta=0; i_eta<_N_eta; ++i_eta){	//each eta
    eta = _eta_lo + i_eta*eta_step;
    for (int i_xi=0; i_xi<_N_xi; ++i_xi){	//each xi
      xi = _xi_lo + i_xi*xi_step;

      // Slicing index
      index = i_eta*_N_xi+i_xi;

      if ( (index-_rank)%_size == 0 ) {
	////CARLOS EDIT
	if (_verbosity>0)
	(*_pstream) << "  ("
		    << std::setw(4) << i_xi
		    << ','
		    << std::setw(4) << i_eta
		    << ')';

	
	init_conds(_frequency,xi,eta,x0,k0);
	
	// Propagate
	_ray.reinitialize(x0,k0);		//start from our xi and eta position, then propagate towards black hole
									//and back so we get our flux value through camera pixel (polarization map)
	_ray.propagate(1.0,"!");
	
	
	// Get polarization
	iquv = _ray.IQUV(); 
	
	_tau[index] = _ray.tau();
	_D[index] = _ray.D();

	_I[index] = iquv[0] * N_to_I;
	_Q[index] = CPV( iquv[1], iquv[0] );
	_U[index] = CPV( iquv[2], iquv[0] );
	_V[index] = CPV( iquv[3], iquv[0] );
	
	if (vrt2_isnan(_I[index]))
	  _I[index]=0.0;
	if (vrt2_isnan(_Q[index]))
	  _Q[index]=0.0;
	if (vrt2_isnan(_U[index]))
	  _U[index]=0.0;
	if (vrt2_isnan(_V[index]))
	  _V[index]=0.0;      
	
	// Make sure that RT is okay
	if (_verbosity>0 && (_I[index]<0.0 || std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)) {
	  std::cout << std::endl
		    << "Bad RT at i_xi, i_eta = "
		    << std::setw(5) << i_xi
		    << std::setw(5) << i_eta
		    << std::setw(15) << xi
		    << std::setw(15) << eta
		    << std::setw(15) << _I[index]
		    << std::setw(15) << _Q[index]
		    << std::setw(15) << _U[index]
		    << std::setw(15) << _V[index]
		    << std::endl;
  	}
//// CARLOS EDIT: REMOVE PRGORESS INDICATOR
	if (_verbosity>0)
	{
	  // Progress indicator increment
	  (*_pstream) << backup;
	  pc.increment(double(index+1)/double(dim));
	}
      }
    }
  }
//// CARLOS INDICATOR: REMOVE PROGRESS INDICATOR
  if (_verbosity>0)
  {
    (*_pstream) << backup;
    pc.finish();
    (*_pstream) << "  ("
		<< std::setw(4) << _N_xi
		<< ','
		<< std::setw(4) << _N_eta
		<< ')';
  }

#ifdef VRT2_USE_MPI_MAP
  collect(0);
#endif
}



// Set current map to background intensity for refining
void PolarizationMap::set_background_intensity()
{
  double ooN_to_I = 1.0/get_N_to_I();
  _Ibg.resize(_I.size());
  _Ibg = _I*ooN_to_I;  // Get atual intensities (not fluxes!)

  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);

  _xibg.resize(_N_xi);
  for (int i_xi=0; i_xi<_N_xi; ++i_xi)
    _xibg[i_xi] = _xi_lo + i_xi*xi_step;
      
  _etabg.resize(_N_eta);
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
    _etabg[i_eta] = _eta_lo + i_eta*eta_step;
}
void PolarizationMap::unset_background_intensity()
{
  _Ibg.resize(4,0.0);
  _xibg.resize(2);
  _xibg[0] = -100;
  _xibg[1] = 100;
  _etabg.resize(2);
  _etabg[0] = -100;
  _etabg[1] = 100;
}


// Refines the map with an adaptive step refinement
void PolarizationMap::refine(double refine_factor)
{
  // Get N_to_I correction (flux to intensity)
  double ooN_to_I = 1.0/get_N_to_I();

  // Save previous values
  int Nxilo = _N_xi;
  int Netalo = _N_eta;
  double xi_step_lo = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step_lo = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  std::valarray<double> xilo, etalo, Ilo(_I.size()), Qlo(_Q.size()), Ulo(_U.size()), Vlo(_V.size()), taulo(_tau.size()), Dlo(_D.size());
  Ilo = _I * ooN_to_I; // Get intensities not fluxes!
  Qlo = _Q;
  Ulo = _U;
  Vlo = _V;
  taulo = _tau;
  Dlo = _D;
  xilo.resize(_N_xi);
  for (int i_xi=0; i_xi<_N_xi; ++i_xi)
    xilo[i_xi] = _xi_lo + i_xi*xi_step_lo;
  etalo.resize(_N_eta);
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
    etalo[i_eta] = _eta_lo + i_eta*eta_step_lo;
  

  // Double the size of the new arrays in each direction (but keep the edges the same)
  _N_xi = 2*Nxilo-1;
  _N_eta = 2*Netalo-1;
  int newsize = _N_xi*_N_eta;
  _I.resize(newsize,0.0);
  _Q.resize(newsize,0.0);
  _U.resize(newsize,0.0);
  _V.resize(newsize,0.0);
  _tau.resize(newsize,0.0);
  _D.resize(newsize,0.0);

  for (size_t i=0; i<_I.size(); ++i)
  {
    _tau[i] = 0.0;
    _D[i] = 0.0;
    _I[i] = 0.0;
    _Q[i] = 0.0;
    _U[i] = 0.0;
    _V[i] = 0.0;
  }


  // Fill in existing values and define refine switch (0->interpolates, 1-> computes refined value)
  // new stepsizes
  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  // eta and xi declaration
  double xi, eta;
  //int indexlo;
  int index;
  //double refine_factor=5e-3;
  double interp_value, interp_value_lo;
  double interp_bg_value, interp_bg_value_lo;
  std::vector<int> refine_switch(newsize,0);
  for (size_t i=0; i<refine_switch.size(); ++i)
      refine_switch[i] = 0;
  bool set_refine_switch;
  double Imax = 0.0; // Absolute scale to compare to.




  // Define an interpolator and set the derivatives (so that we can use bicubic interpolation
  Interpolator2D interp_bgI(_etabg,_xibg,_Ibg);
  interp_bgI.use_forward_difference();

  Interpolator2D interp_I(etalo,xilo,Ilo);
  interp_I.use_forward_difference();
  Interpolator2D interp_Q(etalo,xilo,Qlo*Ilo);
  interp_Q.use_forward_difference();
  Interpolator2D interp_U(etalo,xilo,Ulo*Ilo);
  interp_U.use_forward_difference();
  Interpolator2D interp_V(etalo,xilo,Vlo*Ilo);
  interp_V.use_forward_difference();



  Imax = 0.0;
/*
  for (size_t i=0; i<Ilo.size(); ++i)
      Imax += Ilo[i];
  Imax *= _Ibg.size()/Ilo.size();
  for (size_t i=0; i<_Ibg.size(); ++i)
      Imax -= _Ibg[i];
  Imax /= _Ibg.size();
*/
  for (int i_eta=0; i_eta<Netalo; ++i_eta)
  {
    for (int i_xi=0; i_xi<Nxilo; ++i_xi)
    {
      eta = _eta_lo + i_eta*eta_step_lo;
      xi = _xi_lo + i_xi*xi_step_lo;
  
      interp_I.bicubic(eta,xi,interp_value_lo);
      interp_bgI.bicubic(eta,xi,interp_bg_value_lo);
      interp_value_lo -= interp_bg_value_lo;

      if (interp_value_lo>Imax)
	Imax = interp_value_lo;
    }
  }
/*
  if (_rank==0)
      std::cout << "\n\n";
*/
  for (int i_eta=0; i_eta<Netalo; ++i_eta)
  {
    for (int i_xi=0; i_xi<Nxilo; ++i_xi)
    {
      // Slicing indicies
      //indexlo = i_eta*Nxilo+i_xi;
      index = (2*i_eta)*_N_xi+(2*i_xi);

      // Define refine switch (compare value to that linearly interpolated from points on either side, if different by
      //  "factor" then refine around this point)  Note that this only adds points, but never the already computed points!
      eta = _eta_lo + i_eta*eta_step_lo;
      xi = _xi_lo + i_xi*xi_step_lo;

      if ( i_xi>0 && i_eta>0 && i_xi<Nxilo-1 && i_eta<Netalo-1 )
      {

	// Check I
	//interp_value = 0.25*( Ilo[indexlo+1] + Ilo[indexlo-1] + Ilo[indexlo+Nxilo] + Ilo[indexlo-Nxilo] );
	double sum=0.0;
	for (int ix=-1; ix<=1; ix+=2)
	{
	  // Linear interp in eta-dir
	  interp_I.bicubic(   eta + ix*eta_step_lo , xi , interp_value);
	  interp_bgI.bicubic( eta + ix*eta_step_lo , xi , interp_bg_value);
	  sum += interp_value-interp_bg_value;
	  // Linear interp in xi-dir
	  interp_I.bicubic(   eta , xi + ix*xi_step_lo , interp_value);
	  interp_bgI.bicubic( eta , xi + ix*xi_step_lo , interp_bg_value);
	  sum += interp_value-interp_bg_value;
	}
	sum *= 0.25;
	interp_value = sum;

	interp_I.bicubic(eta,xi,interp_value_lo);
	interp_bgI.bicubic(eta,xi,interp_bg_value_lo);
	interp_value_lo -= interp_bg_value_lo;



	set_refine_switch = bool( std::max(interp_value,interp_value_lo) > (refine_factor*Imax) );
	//set_refine_switch = bool( std::fabs((interp_value_lo-interp_value)/interp_value) > refine_factor )  && bool( std::max(interp_value,interp_value_lo) > (refine_factor*Imax) );
	//set_refine_switch = bool( std::fabs((interp_value_lo-interp_value)/interp_value) > refine_factor )  && bool( std::max(interp_value,interp_value_lo) > (refine_factor*Imax/double(_N_xi*_N_eta)) );
	

	//set_refine_switch = bool( std::fabs((Ilo[indexlo]-interp_value)/interp_value) > refine_factor )  && bool( std::max(interp_value,Ilo[indexlo]) > (1.0e-3*Imax/double(_N_xi*_N_eta)) );

	//set_refine_switch = bool( std::max(interp_value,Ilo[indexlo]) > (1.0e-3*Imax/double(_N_xi*_N_eta)) ) ;
	/*
	// Check Q
	interp_value = 0.25*( Qlo[indexlo+1] + Qlo[indexlo-1] + Qlo[indexlo+Nxilo] + Qlo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Qlo[indexlo]-interp_value)/interp_value) > refine_factor );
	// Check U
	interp_value = 0.25*( Ulo[indexlo+1] + Ulo[indexlo-1] + Ulo[indexlo+Nxilo] + Ulo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Ulo[indexlo]-interp_value)/interp_value) > refine_factor );
	// Check V
	interp_value = 0.25*( Vlo[indexlo+1] + Vlo[indexlo-1] + Vlo[indexlo+Nxilo] + Vlo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Vlo[indexlo]-interp_value)/interp_value) > refine_factor );
	*/
	if (set_refine_switch)
	{
	  // Set the switch at each neighboring point to true
	  refine_switch[index-1-_N_xi] = 1;    
	  refine_switch[index-1] = 1;    
	  refine_switch[index-1+_N_xi] = 1;

	  refine_switch[index-_N_xi] = 1;
	  refine_switch[index+_N_xi] = 1;

	  refine_switch[index+1-_N_xi] = 1;    
	  refine_switch[index+1] = 1;    
	  refine_switch[index+1+_N_xi] = 1;
	}

/*
	if (_rank==0)
	{
	    double tmp;
	    interp_I.bicubic(eta, xi, tmp);
	    interp_bgI.bicubic(eta, xi, interp_bg_value);
	    std::cout << std::setw(15) << xi
		      << std::setw(15) << eta
		      << std::setw(15) << tmp
		      << std::setw(15) << interp_bg_value
		      << std::setw(15) << interp_value
		      << std::setw(15) << interp_value_lo
		      << std::setw(15) << std::max(interp_value,interp_value_lo)
		      << std::setw(15) << refine_factor*Imax
		      << std::endl;
	}
*/
      }
    }
/*
    if (_rank==0)
	std::cout << std::endl;
*/
  }
/*
  if (_rank==0)
      std::cout << std::endl;
*/
#ifdef VRT2_USE_MPI_MAP
  int *send = new int[refine_switch.size()];

  // Redistribute refine_switch
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = refine_switch[i + _N_xi*j];

  MPI_Bcast(&send[0], refine_switch.size(), MPI_INT, 0, _pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      refine_switch[i + _N_xi*j] = send[i + _N_xi*j];

  delete[] send;
#endif

  std::vector<int> i_xi_refine, i_eta_refine;
  std::vector<int> i_xi_interp, i_eta_interp;
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
   for (int i_xi=0; i_xi<_N_xi; ++i_xi)
   {
     // Slicing index
     index = i_eta*_N_xi+i_xi;

     if (refine_switch[index])
     {
	 i_xi_refine.push_back(i_xi);
	 i_eta_refine.push_back(i_eta);
     }
     else
     {
	 i_xi_interp.push_back(i_xi);
	 i_eta_interp.push_back(i_eta);
     }
   }

  // Count up new points and output efficiencies
  //  int newpts=i_xi_refine.size();
//// CARLOS EDIT: REMOVE PI
//  (*_pstream) << "New points: " << std::setw(10) << newpts
//	      << "    Efficiency: " << std::setw(15) << (newpts/double(_N_xi*_N_eta))
//	      << std::endl;



  // Now compute values at new points
  // individual iquv vectors
  std::valarray<double> iquv(0.0,4);
  // Initial values      
  FourVector<double> x0(_g), k0(_g);

  //  if (_rank==0)
  //  std::cerr << "generate: N_to_I = " << N_to_I << std::endl;

//// CARLOS EDIT: REMOVE PROGRESS TO SAVE SPACE
//  /*** Progress indicator start ***/
//  ProgressCounter pc( (*_pstream) );
//  //(*_pstream) << "Done with ";
//  pc.start();
//  std::string backup(3+4+1+4+1,'\b');

  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();


  /*** Get new rays ***/
  for (int i=0; i<int(i_xi_refine.size()); ++i)
  {
      int i_eta = i_eta_refine[i];
      int i_xi = i_xi_refine[i];

      eta = _eta_lo + i_eta*eta_step;
      xi = _xi_lo + i_xi*xi_step;
      
      // Slicing index
      index = i_eta*_N_xi+i_xi;
      //indexlo = int(i_eta/2)*Nxilo + int(i_xi/2);

      if ( (i-_rank)%_size == 0 )
      {
	/*
	(*_pstream) << "  ("
	            << std::setw(4) << i_xi
		    << ','
		    << std::setw(4) << i_eta
		    << ')';
	*/
	  
	init_conds(_frequency,xi,eta,x0,k0);
	  
	// Propagate
	_ray.reinitialize(x0,k0);
	_ray.propagate(1.0,"!");
	  
	  
	// Get polarization
	iquv = _ray.IQUV(); 
	  
	_tau[index] = _ray.tau();
	_D[index] = _ray.D();
	  
	_I[index] = iquv[0] * N_to_I;
	_Q[index] = CPV( iquv[1], iquv[0] );
	_U[index] = CPV( iquv[2], iquv[0] );
	_V[index] = CPV( iquv[3], iquv[0] );
	  

	  if (vrt2_isnan(_I[index]))
	    _I[index]=0.0;
	  if (vrt2_isnan(_Q[index]))
	    _Q[index]=0.0;
	  if (vrt2_isnan(_U[index]))
	    _U[index]=0.0;
	  if (vrt2_isnan(_V[index]))
	    _V[index]=0.0;      
	  
	  // Make sure that RT is okay
	  if (_verbosity>0 && (_I[index]<0.0 || std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)) {
	    std::cout << std::endl
		      << "Bad RT at i_xi, i_eta = "
		      << std::setw(5) << i_xi
		      << std::setw(5) << i_eta
		      << std::setw(15) << xi
		      << std::setw(15) << eta
		      << std::setw(15) << _I[index]
		      << std::setw(15) << _Q[index]
		      << std::setw(15) << _U[index]
		      << std::setw(15) << _V[index]
		      << std::endl;
	  }
	  // Progress indicator increment
	  //(*_pstream) << backup;
	  //pc.increment(double(index+1)/double(newsize));

	  // Marker for refined cells
	  _D[index] = 1;
      }
  }

  /*** Get Interpolated Rays ***/
  for (int i=0; i<int(i_xi_interp.size()); ++i)
  {
      int i_eta = i_eta_interp[i];
      int i_xi = i_xi_interp[i];

      eta = _eta_lo + i_eta*eta_step;
      xi = _xi_lo + i_xi*xi_step;
      
      // Slicing index
      index = i_eta*_N_xi+i_xi;
      //indexlo = int(i_eta/2)*Nxilo + int(i_xi/2);

      if ( (i-_rank)%_size == 0 )
      {  
	  interp_I.bicubic(eta,xi,interp_value);
	  _I[index] = N_to_I * interp_value;
	  interp_Q.bicubic(eta,xi,interp_value);
	  _Q[index] = CPV( N_to_I * interp_value, _I[index]);
	  interp_U.bicubic(eta,xi,interp_value);
	  _U[index] = CPV( N_to_I * interp_value, _I[index]);
	  interp_V.bicubic(eta,xi,interp_value);
	  _V[index] = CPV( N_to_I * interp_value, _I[index]);


	  // Marker for un-refined cells
	  _D[index] = -1;
      }
  }

  
  //(*_pstream) << backup;
//// CARLOS EDIT: REMOVE PI
//  pc.finish();
//  /*
//  (*_pstream) << "  ("
//	      << std::setw(4) << _N_xi
//	      << ','
//	      << std::setw(4) << _N_eta
//	      << ')';
//  */



#ifdef VRT2_USE_MPI_MAP
  collect(0);
#endif

/*
  if (_rank==0)
  {
      std::cout << "\n\n";
      for (int i_eta=0; i_eta<_N_eta; ++i_eta){
	  eta = _eta_lo + i_eta*eta_step;
	  for (int i_xi=0; i_xi<_N_xi; ++i_xi){
	      xi = _xi_lo + i_xi*xi_step;
	      
	      std::cout << std::setw(15) << eta
			<< std::setw(15) << xi
			<< std::setw(15) << refine_switch[i_eta*_N_xi+i_xi]
			<< std::setw(15) << _D[i_eta*_N_xi+i_xi]
			<< std::endl;
	  }
	  std::cout << std::endl;
      }
      std::cout << std::endl;
  }
*/


}

#if 1
// Refines the map with an adaptive step refinement
//void PolarizationMap::background_subtracted_refine(std::vector<double>& xib, std::vector<double>& etab, std::vector<double>& Ib, std::vector<double>& Qb, std::vector<double>& Ub, std::vector<double>& Vb)
void PolarizationMap::old_refine()
{
  // Get N_to_I correction
  double N_to_I_correction = get_N_to_I();

  // Save previous values
  int Nxilo = _N_xi;
  int Netalo = _N_eta;
  std::valarray<double> Ilo(_I.size()), Qlo(_Q.size()), Ulo(_U.size()), Vlo(_V.size()), taulo(_tau.size()), Dlo(_D.size());
  Ilo = _I;
  Qlo = _Q;
  Ulo = _U;
  Vlo = _V;
  taulo = _tau;
  Dlo = _D;

  // Double the size of the new arrays in each direction (but keep the edges the same)
  _N_xi = 2*Nxilo-1;
  _N_eta = 2*Netalo-1;
  int newsize = _N_xi*_N_eta;
  _I.resize(newsize,0.0);
  _Q.resize(newsize,0.0);
  _U.resize(newsize,0.0);
  _V.resize(newsize,0.0);
  _tau.resize(newsize,0.0);
  _D.resize(newsize,0.0);


  // Fill in existing values and define refine switch (0->interpolates, 1-> computes refined value)
  int indexlo, index;
  double refine_factor=5e-2;
  double interp_value;
  std::vector<int> refine_switch(newsize,0);
  bool set_refine_switch;
  double Imax = 0.0; // Absolute scale to compare to.
  for (int i_eta=0; i_eta<Netalo; ++i_eta)
    for (int i_xi=0; i_xi<Nxilo; ++i_xi)
    {
      // Slicing indicies
      indexlo = i_eta*Nxilo+i_xi;
      index = (2*i_eta)*_N_xi+(2*i_xi);

      // Fill proper locations in new arrays
      _I[index] = Ilo[indexlo];
      _Q[index] = Qlo[indexlo];
      _U[index] = Ulo[indexlo];
      _V[index] = Vlo[indexlo];
      _tau[index] = taulo[indexlo];
      _D[index] = Dlo[indexlo];
      if (_I[index]>Imax)
	Imax = _I[index];
    }
  for (int i_eta=0; i_eta<Netalo; ++i_eta)
    for (int i_xi=0; i_xi<Nxilo; ++i_xi)
    {
      // Slicing indicies
      indexlo = i_eta*Nxilo+i_xi;
      index = (2*i_eta)*_N_xi+(2*i_xi);

      // Define refine switch (compare value to that linearly interpolated from points on either side, if different by
      //  "factor" then refine around this point)  Note that this only adds points, but never the already computed points!
      if ( i_xi>0 && i_eta>0 && i_xi<Nxilo-1 && i_eta<Netalo-1 )
      {
	// Check I
	interp_value = 0.25*( Ilo[indexlo+1] + Ilo[indexlo-1] + Ilo[indexlo+Nxilo] + Ilo[indexlo-Nxilo] );
	set_refine_switch = bool( std::fabs((Ilo[indexlo]-interp_value)/interp_value) > refine_factor )  && bool( std::max(interp_value,Ilo[indexlo]) > (1.0e-3*Imax/double(_N_xi*_N_eta)) );
	//set_refine_switch = bool( std::max(interp_value,Ilo[indexlo]) > (1.0e-3*Imax/double(_N_xi*_N_eta)) ) ;
	/*
	// Check Q
	interp_value = 0.25*( Qlo[indexlo+1] + Qlo[indexlo-1] + Qlo[indexlo+Nxilo] + Qlo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Qlo[indexlo]-interp_value)/interp_value) > refine_factor );
	// Check U
	interp_value = 0.25*( Ulo[indexlo+1] + Ulo[indexlo-1] + Ulo[indexlo+Nxilo] + Ulo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Ulo[indexlo]-interp_value)/interp_value) > refine_factor );
	// Check V
	interp_value = 0.25*( Vlo[indexlo+1] + Vlo[indexlo-1] + Vlo[indexlo+Nxilo] + Vlo[indexlo-Nxilo] );
	set_refine_switch = set_refine_switch || bool( std::fabs((Vlo[indexlo]-interp_value)/interp_value) > refine_factor );
	*/
	if (set_refine_switch)
	{
	  // Set the switch at each neighboring point to true
	  refine_switch[index-1-_N_xi] = 1;    
	  refine_switch[index-1] = 1;    
	  refine_switch[index-1+_N_xi] = 1;
	  refine_switch[index-_N_xi] = 1;
	  refine_switch[index+_N_xi] = 1;
	  refine_switch[index-1] = 1;
	  refine_switch[index+1-_N_xi] = 1;    
	  refine_switch[index+1] = 1;    
	  refine_switch[index+1+_N_xi] = 1;
	}
      }
    }

  // Count up new points and output efficiencies
  int newpts=0;
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
    for (int i_xi=0; i_xi<_N_xi; ++i_xi)
      newpts += refine_switch[ i_eta*_N_xi+i_xi ];
  (*_pstream) << "New points: " << std::setw(10) << newpts
	      << "    Efficiency: " << std::setw(15) << (newpts/double(_N_xi*_N_eta))
	      << std::endl;



  // Now compute values at new points

  // eta and xi stepsize
  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  // eta and xi declaration
  double xi, eta;
  // individual iquv vectors
  std::valarray<double> iquv(0.0,4);
  // Initial values      
  FourVector<double> x0(_g), k0(_g);
  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();
  N_to_I_correction = N_to_I/N_to_I_correction;

  //  if (_rank==0)
  //  std::cerr << "generate: N_to_I = " << N_to_I << std::endl;

//// CARLOS EDIT: REMOVE PROGRESS TO SAVE SPACE
//  /*** Progress indicator start ***/
//  ProgressCounter pc( (*_pstream) );
//  (*_pstream) << "Done with ";
//  pc.start();
//  std::string backup(3+4+1+4+1,'\b');

  /*** Begin Loop Over Region ***/
  for (int i_eta=0; i_eta<_N_eta; ++i_eta){
    eta = _eta_lo + i_eta*eta_step;
    for (int i_xi=0; i_xi<_N_xi; ++i_xi){
      xi = _xi_lo + i_xi*xi_step;
      
      // Slicing index
      index = i_eta*_N_xi+i_xi;
      indexlo = int(i_eta/2)*Nxilo + int(i_xi/2);

      if ( (index-_rank)%_size == 0 )
      {
	if (refine_switch[index]) // compute actual value
	{
	  (*_pstream) << "  ("
		      << std::setw(4) << i_xi
		      << ','
		      << std::setw(4) << i_eta
		      << ')';
	  
	  
	  init_conds(_frequency,xi,eta,x0,k0);
	  
	  // Propagate
	  _ray.reinitialize(x0,k0);
	  _ray.propagate(1.0,"!");
	  
	  
	  // Get polarization
	  iquv = _ray.IQUV(); 
	  
	  _tau[index] = _ray.tau();
	  _D[index] = _ray.D();
	  
	  _I[index] = iquv[0] * N_to_I;
	  _Q[index] = CPV( iquv[1], iquv[0] );
	  _U[index] = CPV( iquv[2], iquv[0] );
	  _V[index] = CPV( iquv[3], iquv[0] );
	  
	  if (vrt2_isnan(_I[index]))
	    _I[index]=0.0;
	  if (vrt2_isnan(_Q[index]))
	    _Q[index]=0.0;
	  if (vrt2_isnan(_U[index]))
	    _U[index]=0.0;
	  if (vrt2_isnan(_V[index]))
	    _V[index]=0.0;      
	  
	  // Make sure that RT is okay
	  if (_verbosity>0 && (_I[index]<0.0 || std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)) {
	    std::cout << std::endl
		      << "Bad RT at i_xi, i_eta = "
		      << std::setw(5) << i_xi
		      << std::setw(5) << i_eta
		      << std::setw(15) << xi
		      << std::setw(15) << eta
		      << std::setw(15) << _I[index]
		      << std::setw(15) << _Q[index]
		      << std::setw(15) << _U[index]
		      << std::setw(15) << _V[index]
		      << std::endl;
	  }
	  // Progress indicator increment
//// CARLOS EDIT: REMOVE PI
//	  (*_pstream) << backup;
//	  pc.increment(double(index+1)/double(newsize));

	  // Marker for refined cells
	  _D[index] = 1;
	}
	else // interpolate
	{
	  if ( (i_eta%2) && (i_xi%2) ) // If a new point in both directions, i.e., not i_eta, i_xi = 0 + N*2
	  {
	    _I[index] = N_to_I_correction * 0.25*( Ilo[indexlo] + Ilo[indexlo+1] + Ilo[indexlo+Nxilo] + Ilo[indexlo+Nxilo+1] );
	    _Q[index] = CPV( 0.25*N_to_I_correction * ( Qlo[indexlo]*Ilo[indexlo] + Qlo[indexlo+1]*Ilo[indexlo+1] + Qlo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] + Qlo[indexlo+Nxilo+1]*Ilo[indexlo+Nxilo+1] ) , _I[index]);
	    _U[index] = CPV( 0.25*N_to_I_correction * ( Ulo[indexlo]*Ilo[indexlo] + Ulo[indexlo+1]*Ilo[indexlo+1] + Ulo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] + Ulo[indexlo+Nxilo+1]*Ilo[indexlo+Nxilo+1] ) , _I[index]);
	    _V[index] = CPV( 0.25*N_to_I_correction * ( Vlo[indexlo]*Ilo[indexlo] + Vlo[indexlo+1]*Ilo[indexlo+1] + Vlo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] + Vlo[indexlo+Nxilo+1]*Ilo[indexlo+Nxilo+1] ) , _I[index]);
	    _tau[index] = 0.25*(taulo[indexlo]+taulo[indexlo+1]+taulo[indexlo+Nxilo]+taulo[indexlo+Nxilo+1]);
	    _D[index] = 0.25*(Dlo[indexlo]+Dlo[indexlo+1]+Dlo[indexlo+Nxilo]+Dlo[indexlo+Nxilo+1]);  // marker for where we DID NOT refine
	  }
	  else if ( (i_eta%2) ) // If a new point in the eta direction
	  {
	    _I[index] = N_to_I_correction * 0.5*(Ilo[indexlo]+Ilo[indexlo+Nxilo]);
	    _Q[index] = CPV( 0.5*N_to_I_correction * ( Qlo[indexlo]*Ilo[indexlo] + Qlo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] ), _I[index] );
	    _U[index] = CPV( 0.5*N_to_I_correction * ( Ulo[indexlo]*Ilo[indexlo] + Ulo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] ), _I[index] );
	    _V[index] = CPV( 0.5*N_to_I_correction * ( Vlo[indexlo]*Ilo[indexlo] + Vlo[indexlo+Nxilo]*Ilo[indexlo+Nxilo] ), _I[index] );
	    _tau[index] = 0.5*(taulo[indexlo]+taulo[indexlo+Nxilo]);
	    _D[index] = 0.5*(Dlo[indexlo]+Dlo[indexlo+Nxilo]);
	  }
	  else if ( (i_xi%2) ) // If a new point in the xi direction
	  {
	    _I[index] = N_to_I_correction * 0.5*(Ilo[indexlo]+Ilo[indexlo+1]);
	    _Q[index] = CPV( 0.5*N_to_I_correction * ( Qlo[indexlo]*Ilo[indexlo] + Qlo[indexlo+1]*Ilo[indexlo+1] ) , _I[index] );
	    _U[index] = CPV( 0.5*N_to_I_correction * ( Ulo[indexlo]*Ilo[indexlo] + Ulo[indexlo+1]*Ilo[indexlo+1] ) , _I[index] );
	    _V[index] = CPV( 0.5*N_to_I_correction * ( Vlo[indexlo]*Ilo[indexlo] + Vlo[indexlo+1]*Ilo[indexlo+1] ) , _I[index] );
	    _tau[index] = 0.5*(taulo[indexlo]+taulo[indexlo+1]);
	    _D[index] = 0.5*(Dlo[indexlo]+Dlo[indexlo+1]);
	  }
	  else
	  {
	    _I[index] = N_to_I_correction * Ilo[indexlo];
	    _Q[index] = Qlo[indexlo];
	    _U[index] = Ulo[indexlo];
	    _V[index] = Vlo[indexlo];
	    _tau[index] = taulo[indexlo];
	    _D[index] = Dlo[indexlo];
	  }
	  // Marker for un-refined cells
	  _D[index] = -1;
	}
      }
    }
  }
//// CARLOS EDIT : REMOVE PI
//  (*_pstream) << backup;
//  pc.finish();
  (*_pstream) << "  ("
	      << std::setw(4) << _N_xi
	      << ','
	      << std::setw(4) << _N_eta
	      << ')';

#ifdef VRT2_USE_MPI_MAP
  collect(0);
#endif
}
#endif

// Reimmanian integration (EXPECTS THAT IT HAS ALREADY BEEN COLLECTED!)
std::valarray<double> PolarizationMap::integrate()
{
  _IQUV_int.resize(4,0.0);
  for (int i=0; i<4; i++)
    _IQUV_int[i] = 0.0;

  for (int i=0; i<_N_xi*_N_eta; i++){
    _IQUV_int[0] += _I[i];
    _IQUV_int[1] += _I[i]*_Q[i];
    _IQUV_int[2] += _I[i]*_U[i];
    _IQUV_int[3] += _I[i]*_V[i];
  }
  _IQUV_int[1] = CPV(_IQUV_int[1],_IQUV_int[0]);
  _IQUV_int[2] = CPV(_IQUV_int[2],_IQUV_int[0]);
  _IQUV_int[3] = CPV(_IQUV_int[3],_IQUV_int[0]);

  return (_IQUV_int);

/*
#ifndef VRT2_USE_MPI_MAP
    for (int i=0; i<_N_xi*_N_eta; i++){
      _IQUV_int[0] += _I[i];
      _IQUV_int[1] += _I[i]*_Q[i];
      _IQUV_int[2] += _I[i]*_U[i];
      _IQUV_int[3] += _I[i]*_V[i];
    }
    _IQUV_int[1] = CPV(_IQUV_int[1],_IQUV_int[0]);
    _IQUV_int[2] = CPV(_IQUV_int[2],_IQUV_int[0]);
    _IQUV_int[3] = CPV(_IQUV_int[3],_IQUV_int[0]);
#else
    double iquv_int_local[]={0.0, 0.0, 0.0, 0.0};
    double iquv_int_global[]={0.0, 0.0, 0.0, 0.0};
    for (int i=0; i<_N_xi*_N_eta; i++){
      iquv_int_local[0] += _I[i];
      iquv_int_local[1] += _I[i]*_Q[i];
      iquv_int_local[2] += _I[i]*_U[i];
      iquv_int_local[3] += _I[i]*_V[i];
    }
    MPI_Reduce(&iquv_int_local[0], &iquv_int_global[0], 4, MPI_DOUBLE, MPI_SUM, 0, _pmap_communicator);
 
    _IQUV_int[0] = iquv_int_global[0];  
    _IQUV_int[1] = iquv_int_global[1]; //CPV(iquv_int_global[1],iquv_int_global[0]);
    _IQUV_int[2] = iquv_int_global[2]; //CPV(iquv_int_global[2],iquv_int_global[0]);
    _IQUV_int[3] = iquv_int_global[3]; //CPV(iquv_int_global[3],iquv_int_global[0]);
#endif
*/
  return (_IQUV_int);
}

double PolarizationMap::I_int() { return(_IQUV_int[0]); }
double PolarizationMap::Q_int() { return(_IQUV_int[1]); }
double PolarizationMap::U_int() { return(_IQUV_int[2]); }
double PolarizationMap::V_int() { return(_IQUV_int[3]); }


void PolarizationMap::eta_section(double eta_hi, double xi_lo, double xi_hi, int N_xi)
{
  eta_section(_frequency0,eta_hi,xi_lo,xi_hi,N_xi);
}

void PolarizationMap::eta_section(double frequency, double eta_hi, double xi_lo, double xi_hi, int N_xi)
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
  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();

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
    _ray.reinitialize(x0,k0);
    std::vector<std::string> ray_out = _ray.propagate(1.0,out_name.str());

    // Get polarization
    iquv = _ray.IQUV();

    _tau[index] = _ray.tau();
    _D[index] = _ray.D();
    _I[index] = iquv[0] * N_to_I;
    _Q[index] = CPV( iquv[1], iquv[0] );
    _U[index] = CPV( iquv[2], iquv[0] );
    _V[index] = CPV( iquv[3], iquv[0] );
    
    if (vrt2_isnan(_I[index]))
      _I[index]=0.0;
    if (vrt2_isnan(_Q[index]))
      _Q[index]=0.0;
    if (vrt2_isnan(_U[index]))
      _U[index]=0.0;
    if (vrt2_isnan(_V[index]))
      _V[index]=0.0;      
    
    // Make sure that RT is okay
    if (std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)
      std::cerr << std::endl
		<< "Bad RT at i_xi, i_eta = "
		<< std::setw(5) << i_xi
		<< std::setw(5) << 0
		<< std::setw(15) << _I[index]
		<< std::setw(15) << _Q[index]
		<< std::setw(15) << _U[index]
		<< std::setw(15) << _V[index]
		<< std::endl;
    
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
//// CARLOS EDIT    
//    // Progress indicator start
//    (*_pstream) << "\b\b\b\b\b\b" << std::setw(5)
//	 << 100.0*double(i_xi+1)/double(N_xi) << '%';
    
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

void PolarizationMap::xi_section(double xi_hi, double eta_lo, double eta_hi, int N_eta)
{
  xi_section(_frequency0,xi_hi,eta_lo,eta_hi,N_eta);
}

void PolarizationMap::xi_section(double frequency, double xi_hi, double eta_lo, double eta_hi, int N_eta)
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
  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();

  //std::cout << "N_to_I = " << N_to_I << std::endl;

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
    _ray.reinitialize(x0,k0);
    std::vector<std::string> ray_out = _ray.propagate(1.0,out_name.str());

    // Get polarization
    iquv = _ray.IQUV();

    _tau[index] = _ray.tau();
    _D[index] = _ray.D();
    _I[index] = iquv[0] * N_to_I;
    _Q[index] = CPV( iquv[1], iquv[0] );
    _U[index] = CPV( iquv[2], iquv[0] );
    _V[index] = CPV( iquv[3], iquv[0] );
    
    if (vrt2_isnan(_I[index]))
      _I[index]=0.0;
    if (vrt2_isnan(_Q[index]))
      _Q[index]=0.0;
    if (vrt2_isnan(_U[index]))
      _U[index]=0.0;
    if (vrt2_isnan(_V[index]))
      _V[index]=0.0;      
    
    // Make sure that RT is okay
    if (std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)
      (*_pstream) << std::endl
	   << "Bad RT at i_xi, i_eta = "
	   << std::setw(5) << 0
	   << std::setw(5) << i_eta
	   << std::setw(15) << _I[index]
	   << std::setw(15) << _Q[index]
	   << std::setw(15) << _U[index]
	   << std::setw(15) << _V[index]
	   << std::endl;
    
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

//// CARLOS EDIT
//    // Progress indicator start
//    (*_pstream) << "\b\b\b\b\b\b" << std::setw(5)
//	 << 100.0*double(i_eta+1)/double(N_eta) << '%';

    smfile.flush();
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



void PolarizationMap::eta_point_section(double x[], int N)
{
  eta_point_section(_frequency0,x,N);
}

void PolarizationMap::eta_point_section(double frequency, double x[], int N)
{
  /*** Assign frequency ***/
  _frequency = frequency;

  /*** Assign Limits ***/
  _xi_hi = 1.0;
  _eta_hi = 1.0;  // Use unity as arbitrary values
  _N_xi = N;
  _N_eta = 1;

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
	 << "location 0 32767 0 32767\n\n"
    //<< "location 3700 31000 3700 31000\n\n"
	 << "define Sqr_Size 5 !plot size\n\n"
	 << "expand 1.5\n"
	 << "limits -$Sqr_Size $Sqr_Size -$Sqr_Size $Sqr_Size\n"
    //<< "box\n"
    //<< "expand 2\n"
    //<< "ylabel x(M)\nxlabel y(M)\n"
    //<< "expand 1\n"
    //<< "! Draw the ergosphere\n"
    //<< "set phi=0,6.3,0.1\n"
    //<< "set cth=cos(phi)*cos(" << _THETA << ") \n"
    //<< "set rergo=" << _g.mass()
    //<< "+sqrt(" << _g.mass() << "**2 - "
    //<< _g.ang_mom() << "**2 * cth**2 )\n"
    //<< "lweight 1\n"
    //<< "set x=rergo*cos(phi)\n"
    //<< "set y=rergo*sin(phi)\n"
    //<< "angle 45\n"
    //<< "shade 300 y x\n"
    //<< "angle 0\n"
    //<< "connect y x"
	 << "lweight 3\n\n"
	 << "define rw (5)\n\n";
  //<< "! Draw horizon\n"
  //<< "set phi=0,6.3,0.1\n"
  //<< "set x=" << _g.horizon() << "*cos(phi)\n"
  //<< "set y=" << _g.horizon() << "*sin(phi)\n"
  //<< "shade 0 y x\n\n"
  //<< "lweight 3\n";

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
  // Initial values      
  FourVector<double> x0(_g), k0(_g);
  // Constant to take N to I in Jy
  double N_to_I = get_N_to_I();

  //std::cout << "N_to_I = " << N_to_I << std::endl;

  // Output File name
  std::string pre="../ray_data/eta_";

  /*** Progress indicator start ***/
  (*_pstream).setf(std::ios::fixed);
  (*_pstream) << std::setprecision(1);
  (*_pstream) << std::endl << std::endl;
  (*_pstream) << "Done with   0.0%";

  /*** Begin Loop Over Region ***/
  double alpha;
  for (int i_xi=0; i_xi<_N_xi; i_xi++){
    alpha = 2.0*M_PI*double(i_xi)/double(_N_xi);

    // Slicing index
    index = i_xi;
    
    // Get initial values
    point_init_conds(_frequency,alpha,x,x0,k0);

    // Get output
    std::ostringstream out_name;
    out_name << pre << i_xi << '_';

    // Propagate
    _ray.reinitialize(x0,k0);
    std::vector<std::string> ray_out = _ray.propagate(1.0,out_name.str());

    // Get polarization
    iquv = _ray.IQUV();

    _tau[index] = _ray.tau();
    _D[index] = _ray.D();
    _I[index] = iquv[0] * N_to_I;
    _Q[index] = CPV( iquv[1], iquv[0] );
    _U[index] = CPV( iquv[2], iquv[0] );
    _V[index] = CPV( iquv[3], iquv[0] );
    
    if (vrt2_isnan(_I[index]))
      _I[index]=0.0;
    if (vrt2_isnan(_Q[index]))
      _Q[index]=0.0;
    if (vrt2_isnan(_U[index]))
      _U[index]=0.0;
    if (vrt2_isnan(_V[index]))
      _V[index]=0.0;      
    
    // Make sure that RT is okay
    if (std::fabs(_Q[index])>1.0 || std::fabs(_U[index])>1.0 || std::fabs(_V[index])>1.0)
      std::cerr << std::endl
		<< "Bad RT at i_xi, i_eta = "
		<< std::setw(5) << i_xi
		<< std::setw(5) << 0
		<< std::setw(15) << _I[index]
		<< std::setw(15) << _Q[index]
		<< std::setw(15) << _U[index]
		<< std::setw(15) << _V[index]
		<< std::endl;
    
    for (unsigned int i=0;i<ray_out.size();++i){
      double alpha_prev = 2.0*M_PI*double(i_xi-1)/double(_N_xi);
      double alpha_next = 2.0*M_PI*double(i_xi+1)/double(_N_xi);


      smfile << "data ../" << ray_out[i] << "\nlines 3 100000\n"
	     << "read {x 3 y 4 z 5}\n"
	     << "lweight $rw\n"
	     << "ltype 0\n";
      if ( (std::cos(alpha)*std::cos(alpha_next)<=0.0
	    &&
	    std::fabs(std::cos(alpha))<std::fabs(std::cos(alpha_next)))
	   ||
	   (std::cos(alpha)*std::cos(alpha_prev)<=0.0
	    &&
	    std::fabs(std::cos(alpha))<std::fabs(std::cos(alpha_prev)))
	 )
      {
	smfile << "ctype green\n"
	       << "connect x y\n"
	       << "ctype default\n\n";
      }
      else if (std::cos(alpha)>=0)
      {
	smfile << "ctype red\n"
	       << "connect x y\n"
	       << "ctype default\n\n";
      }
      else
      {
	smfile << "ctype blue\n"
	       << "connect x y\n"
	       << "ctype default\n\n";
      }



      mfile << "ray=plot_ray('../../" << ray_out[i] << "');\n"
	    << "set(ray,";

      mfile << "'color','red','linewidth',1);\n";
    }
    
//// CARLOS EDIT
//    // Progress indicator start
//    (*_pstream) << "\b\b\b\b\b\b" << std::setw(5)
//		<< 100.0*double(i_xi+1)/double(_N_xi) << '%';
    
  }
  (*_pstream) << std::endl << std::endl << std::endl;  
  
  /*** Close SM file ***/
  smfile << "! Draw horizon\n"
	 << "set phi=0,6.3,0.1\n"
	 << "set x=" << _g.horizon() << "*cos(phi)\n"
	 << "set y=" << _g.horizon() << "*sin(phi)\n"
	 << "shade 0 y x\n\n"
	 << "lweight 3\n\n"
	 << "! Draw photon orbit\n"
	 << "set x=3*cos(phi)\n"
	 << "set y=3*sin(phi)\n"
	 << "ltype 3\n"
	 << "connect x y\n\n";

  //smfile << "identification\n";
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

int PolarizationMap::init_conds(double frequency, double xi, double eta,
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


int PolarizationMap::point_init_conds(double frequency, double alpha,
				      double x0[],
				      FourVector<double> &x, FourVector<double> &k)
{
  double kx,ky,kz;
  double k0[4];

  x.mkcon(x0);

  // Set metric
  _g.reset(x0);


  // Set k0
  kz = 0.0;
  kx = std::cos(alpha+x0[3]);
  ky = std::sin(alpha+x0[3]);
    
  // ASSUMES THE METRIC IS DIAGONAL!!!!
  k0[0] = -2.0*VRT2_Constants::pi * _frequency0; // covariant component -> going backwards in time
  k0[1] = std::fabs(k0[0]) * ( std::sin(x0[2])*std::cos(x0[3])*kx + std::sin(x0[2])*std::sin(x0[3])*ky + std::cos(x0[2])*kz ) * std::sqrt(-_g.ginv(0,0)/_g.ginv(1,1));
  k0[2] = std::fabs(k0[0]) * ( x0[1]*std::cos(x0[2])*std::cos(x0[3])*kx + x0[1]*std::cos(x0[2])*std::sin(x0[3])*ky - x0[1]*std::sin(x0[2])*kz ) * std::sqrt(-_g.ginv(0,0));
  k0[3] = std::fabs(k0[0]) * ( - x0[1]*std::sin(x0[2])*std::sin(x0[3])*kx + x0[1]*std::sin(x0[2])*std::cos(x0[3])*ky ) * std::sqrt(-_g.ginv(0,0));


  // sin(x0[2])=1, cos(x0[2])=0, sin(x0[0])=0, cos(x0[0])=1
  // k0_r  = k0_t*( kx*sqrt(-g^tt/g^rr) )
  // k0_th = k0_t*( 0 )
  // k0_ph = k0_t*( ky*sqrt(-g^tt/g^pp) )
  //
  // --> k^2 = g^tt k_t^2 + g^rr k_r^2 + g^pp k_p^2
  //         = k_t^2 ( g^tt + g^rr [-g^tt kx^2/g^rr] + g^pp [-g^tt ky^2/g^pp] )
  //         = k_t^2 g^tt ( 1 - kx^2 - ky^2 )
  //         = k_t^2 g^tt ( 1 - 1 ) = 0


  k.mkcov(k0);
  //std::cout << "k:\n" << k << '\n';
  std::cout << "k^2 = " << std::setw(15) << (k*k) << '\n';

  return 0;
}  

// Stokes Parameters
double PolarizationMap::I(int i_xi,int i_eta)		//x,y (col then row)
{
  return ( _I[i_eta*_N_xi+i_xi] );
}
double PolarizationMap::Q(int i_xi,int i_eta)
{
  return ( _Q[i_eta*_N_xi+i_xi] );
}
double PolarizationMap::U(int i_xi,int i_eta)
{
  return ( _U[i_eta*_N_xi+i_xi] );
}
double PolarizationMap::V(int i_xi,int i_eta)
{
  return ( _V[i_eta*_N_xi+i_xi] );
}

// Checks and Other Functions
double PolarizationMap::tau(int i_xi,int i_eta)
{
  return ( _tau[i_eta*_N_xi+i_xi] );
}
double PolarizationMap::D(int i_xi,int i_eta)
{
  return ( _D[i_eta*_N_xi+i_xi] );
}

void PolarizationMap::get_map(std::valarray<double>& xi, std::valarray<double>& eta, std::valarray<double>& I, std::valarray<double>& Q, std::valarray<double>& U, std::valarray<double>& V)
{
  xi.resize(_I.size());
  eta.resize(_I.size());
  
  int index;
  double xi_step = ( _N_xi>1 ? (_xi_hi-_xi_lo)/(_N_xi-1) : 0.0);
  double eta_step = ( _N_eta>1 ? (_eta_hi-_eta_lo)/(_N_eta-1) : 0.0);
  for (int i_eta=0; i_eta<_N_eta; ++i_eta)
    for (int i_xi=0; i_xi<_N_xi; ++i_xi)
    {
      index = i_eta*_N_xi+i_xi;
      eta[index] = _eta_lo + i_eta*eta_step;
      xi[index] = _xi_lo + i_xi*xi_step;
    }

  I = _I;
  Q = _Q;
  U = _U;
  V = _V;
}

void PolarizationMap::output_map(std::string fname)
{
  if (_rank==0) {
    std::ofstream PolarizationMap_out(fname.c_str());

    if (!PolarizationMap_out.is_open())
      std::cerr << "Couldn't open " << fname << " to output map\n";

    output_map(PolarizationMap_out);
  }
}

void PolarizationMap::output_map(std::ostream& PolarizationMap_out)
{
  if (_rank==0) {
    PolarizationMap_out.setf(std::ios::scientific);
    PolarizationMap_out << std::setprecision(5);

    // Info for SM
    PolarizationMap_out << "xi: " << _xi_lo << ' ' << _xi_hi << ' ' << _N_xi << '\n';
    PolarizationMap_out << "eta: " << _eta_lo << ' ' << _eta_hi << ' ' << _N_eta << '\n';
    
    // Headers
    PolarizationMap_out << std::setw(7) << "i_xi"
			<< std::setw(7) << "i_eta"
			<< std::setw(15) << "I"
			<< std::setw(15) << "Q"
			<< std::setw(15) << "U"
			<< std::setw(15) << "V"
			<< std::setw(15) << "tau"
			<< std::setw(15) << "D\n";

    for (int i_eta=0; i_eta<_N_eta; ++i_eta){
      for (int i_xi=0; i_xi<_N_xi; ++i_xi){
	PolarizationMap_out << std::setw(7) << i_xi
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

#ifdef VRT2_USE_MPI_MAP
void PolarizationMap::collect(int rank)
{

#ifdef MPI_BUFF_LIMITS
  // Assumes that this process is involved iff (index-_rank)%_size == 0
  MPI_Barrier(_pmap_communicator);
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
		  MPI_Send(&_I[index],1,MPI_DOUBLE,rank,20,_pmap_communicator);
		  MPI_Send(&_Q[index],1,MPI_DOUBLE,rank,21,_pmap_communicator);
		  MPI_Send(&_U[index],1,MPI_DOUBLE,rank,22,_pmap_communicator);
		  MPI_Send(&_V[index],1,MPI_DOUBLE,rank,23,_pmap_communicator);
		  MPI_Send(&_tau[index],1,MPI_DOUBLE,rank,24,_pmap_communicator);
		  MPI_Send(&_D[index],1,MPI_DOUBLE,rank,25,_pmap_communicator);
		}
		else if (_rank == rank) { // If I'm the root, then recieve
		  MPI_Recv(&_I[index],1,MPI_DOUBLE,sender_rank,20,_pmap_communicator);
		  MPI_Recv(&_Q[index],1,MPI_DOUBLE,sender_rank,21,_pmap_communicator);
		  MPI_Recv(&_U[index],1,MPI_DOUBLE,sender_rank,22,_pmap_communicator);
		  MPI_Recv(&_V[index],1,MPI_DOUBLE,sender_rank,23,_pmap_communicator);
		  MPI_Recv(&_tau[index],1,MPI_DOUBLE,sender_rank,24,_pmap_communicator);
		  MPI_Recv(&_D[index],1,MPI_DOUBLE,sender_rank,25,_pmap_communicator);
		}
      }
    }
#else
  double *send = new double[_I.size()];
  double *recv = new double[_I.size()];

  // I
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _I[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _I.size(), MPI_DOUBLE, MPI_SUM,_pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _I[i + _N_xi*j] = recv[i + _N_xi*j];
  
  // Q
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _Q[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _N_xi*_N_eta, MPI_DOUBLE, MPI_SUM,_pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _Q[i + _N_xi*j] = recv[i + _N_xi*j];

  // U
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _U[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _N_xi*_N_eta, MPI_DOUBLE, MPI_SUM,_pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _U[i + _N_xi*j] = recv[i + _N_xi*j];

  // V
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _V[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _N_xi*_N_eta, MPI_DOUBLE, MPI_SUM,_pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _V[i + _N_xi*j] = recv[i + _N_xi*j];

  // tau
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _tau[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _N_xi*_N_eta, MPI_DOUBLE, MPI_SUM,_pmap_communicator);
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      _tau[i + _N_xi*j] = recv[i + _N_xi*j];

  // D
  for (int i=0; i<_N_xi; ++i)
    for (int j=0; j<_N_eta; ++j)
      send[i + _N_xi*j] = _D[i + _N_xi*j];
  MPI_Allreduce(&send[0], &recv[0], _N_xi*_N_eta, MPI_DOUBLE, MPI_SUM,_pmap_communicator);
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

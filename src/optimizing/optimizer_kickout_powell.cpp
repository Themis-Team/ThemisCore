/*!
  \file optimizer_kickout_powell.cpp
  \author Avery Broderick
  \brief Implementation file for the optimizer_kickout_powell class
*/


#include "optimizer_kickout_powell.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>


namespace Themis{

  optimizer_kickout_powell::optimizer_kickout_powell(int seed)
    : _num_likelihood(1), _rng(seed), _prior_edge_beta(1.0), _ko_ll_red_fac(10.0), _ko_itermax(20), _ko_rounds(20)
  {
  }

  optimizer_kickout_powell::~optimizer_kickout_powell()
  {
  }

  void optimizer_kickout_powell::set_cpu_distribution(int num_likelihood)
  {
    _num_likelihood = num_likelihood;
  }

  void optimizer_kickout_powell::set_kickout_parameters(double kickout_loglikelihood_reduction_factor, size_t kickout_itermax, size_t kickout_rounds)
  {
    _ko_ll_red_fac = kickout_loglikelihood_reduction_factor;
    _ko_itermax = kickout_itermax;
    _ko_rounds = kickout_rounds;
  }


  std::vector<double> optimizer_kickout_powell::run_optimizer(likelihood& L, int dof_estimate, std::vector< std::vector<double> > start_parameter_values, std::string optimizer_results_filename, size_t number_of_restarts, double tolerance)
  {
    // Set pointer to likelihood object
    _Lptr = &L;

    // Get the number of parameters for the model
    _ndim = L.priors().size();

    // Set the number of degrees of freedom (approximately)
    _dof_estimate = dof_estimate;
   
    // Set the number of instances
    size_t number_of_instances = start_parameter_values.size();
    
    // Create a set of communicators for the likelihoods and Powells
    //  Communicator, ranks, communicator construction
    MPI_Comm W_COMM, L_COMM;
    MPI_Status Stat[number_of_instances];
    int rank, color, hue, W_size, W_rank, L_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //   Likelihoods
    color = rank / _num_likelihood;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &L_COMM);
    MPI_Comm_rank(L_COMM, &L_rank);
    //   Cross-Powell communication (necessary only among rank 0)
    hue = rank%_num_likelihood;
    MPI_Comm_split(MPI_COMM_WORLD, hue, rank, &W_COMM);
    MPI_Comm_rank(W_COMM, &W_rank);
    MPI_Comm_size(W_COMM, &W_size);
    
    // Output my current information
    std::cout << "optimizer_kickout_powell.run_optimizer : rank=" << rank << " L_rank=" << L_rank << " W_rank=" << W_rank << std::endl;


    // Set the likelihood commmunicator 
    _Lptr->set_mpi_communicator(L_COMM);

    // Open the optimizer summary output file
    std::ofstream optout;
    if (W_rank==0 && L_rank==0)
      optout.open(optimizer_results_filename);

    // Loop through and save best values
    //   Make some space for saving best values
    std::vector<double> Lbest_list;
    std::vector< std::vector<double> > pbest_list;
    double* buff = new double[_ndim+1];
    for (size_t i=0; i<number_of_instances; ++i)
    {
      if (i%W_size==size_t(W_rank)) // If this process is tasked with this instance
      {
	//std::cout << "W_rank=" << W_rank << " started optimization run" << std::endl;


	// Start with kickouts
	_prior_edge_beta = 0.0;
	std::vector<double> p_kicks;
	double L_kicks=0;
	for (size_t r=0; r<_ko_rounds; ++r)
	{
	  double Lopt = get_optimal_point(start_parameter_values[i],tolerance,_ko_itermax);

	  if (r==0 || Lopt>L_kicks)
	  {
	    L_kicks = Lopt;
	    p_kicks = start_parameter_values[i];
	    //std::cout << "Rank " << rank << " KICK KEEP: " << Lopt << std::endl;
	  }

	  //std::cout << "Rank " << rank << " KICKOUT CHECK: " << Lopt << " " << -10*_dof_estimate << std::endl;
	  if (Lopt<-0.5*_ko_ll_red_fac*_dof_estimate) // Then not a good location, try again!
	  {
	    start_parameter_values[i] = generate_start_point(L);
	    for (size_t k=0; k<_ndim; ++k)
	      buff[k] = start_parameter_values[i][k];
	    MPI_Bcast(&buff[0],_ndim,MPI_DOUBLE,0,L_COMM);
	    for (size_t k=0; k<_ndim; ++k)
	      start_parameter_values[i][k] = buff[k];	    
	  }
	}
	// Get the best of the lot and continue
	start_parameter_values[i] = p_kicks;
	//std::cout << "Rank " << rank << " best KICK: " << L_kicks << std::endl;

	

	/* // Repeat with the requested number of restarts
	for (size_t r=1; r<number_of_restarts; ++r)
	{
	  _prior_edge_beta = double(r)/double(number_of_restarts);
	  get_optimal_point(start_parameter_values[i],tolerance,2*_ko_itermax);
	}
	*/

	// Save the likelihood of the best
	_prior_edge_beta = std::max(0.9999,1.0-0.5/double(number_of_restarts));
	Lbest_list.push_back( get_optimal_point(start_parameter_values[i],tolerance,2*_ko_itermax) );

	// Save the position of the best
	pbest_list.push_back( start_parameter_values[i] );

	//std::cout << "W_rank=" << W_rank << " finished optimization run" << std::endl;
      }
    }

    //std::cerr << "FOO " << W_rank << " " << L_rank << '\n';

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Loop through and communicate best values to the master process to output
    for (size_t i=0; i<number_of_instances; ++i)
    {
      if (i%W_size==size_t(W_rank)) // If this process is tasked with this instance
      {
	// Send the best to the master
	buff[0] = Lbest_list[Lbest_list.size()-1];
	for (size_t j=0; j<_ndim; ++j)
	  buff[j+1] = pbest_list[Lbest_list.size()-1][j];
	// (only send if this is not the master)
	//std::cerr << "optimizer_kickout_powell::run_optimizer : foo1 " << W_rank << " " << L_rank << " " << i << '\n';
	/*std::cout << " okp::ro " << W_rank << " : ";
	for (size_t j=0; j<=_ndim; ++j)
	  std::cout << std::setw(15) << buff[j];
	  std::cout << std::endl;*/
	if (L_rank==0 && W_rank!=0)
	{
	  MPI_Send(&buff[0],int(_ndim)+1,MPI_DOUBLE,0,50+int(W_rank),W_COMM);
	}
	//std::cerr << "optimizer_kickout_powell::run_optimizer : foo2 " << W_rank << " " << L_rank << " " << i << '\n';
      }
            
      // If this is the master, output to the optimizer summary file
      if (W_rank==0 && L_rank==0) 
      {
	// (only receive if sent from something other than the master)
	//std::cerr << "optimizer_kickout_powell::run_optimizer : bar1 " << W_rank << " " << L_rank << " " << i << '\n';
	if (i%W_size!=0)
	{
	  MPI_Recv(&buff[0],int(_ndim)+1,MPI_DOUBLE,int(i%W_size),50+int(i%W_size),W_COMM, &Stat[i]);
	}
	//std::cerr << "optimizer_kickout_powell::run_optimizer : bar2 " << W_rank << " " << L_rank << " " << i << '\n';
	
	// Write output to the summary file
	for (size_t j=0; j<_ndim+1; ++j)
	  optout << std::setw(15) << buff[j];
	optout << std::endl;
      }
    }

    //std::cerr << "BAR " << W_rank << " " << L_rank << '\n';

    
    MPI_Barrier(MPI_COMM_WORLD);

    // Find the best point in two steps:
    // 1. Find the best local point
    double Lbest = Lbest_list[0];
    std::vector<double> pbest = pbest_list[0];
    for (size_t i=1; i<Lbest_list.size(); ++i)
      if (Lbest_list[i]>Lbest)
      {
	Lbest = Lbest_list[i];
	pbest = pbest_list[i];
      }

    // 2. Now find the best global point from among the best local pointst
    int ibest;
    double* Lbest_array = new double[W_size];
    if (L_rank==0)
    {
      // Gather the best local likelihoods into an array indexed by W_rank
      MPI_Gather(&Lbest,1,MPI_DOUBLE,Lbest_array,1,MPI_DOUBLE,0,W_COMM);
      
      // Have the master do all of the selection
      if (W_rank==0) 
      {
	ibest = 0;
	Lbest = Lbest_array[0];
	for (size_t i=1; i<size_t(W_size); ++i)
	  if (Lbest_array[i]>Lbest)
	  {
	    Lbest = Lbest_array[i];
	    ibest = int(i);
	  }
      }
      
      // Once the W_rank of the best fit is known (ibest), now broadcast it to all of the processes in W_COMM with L_rank=0
      MPI_Bcast(&ibest,1,MPI_INT,0,W_COMM);
      
      // Now fill the buffer with the best point and broadcast to all processes in W_COMM with L_rank=0
      for (size_t j=0; j<_ndim; ++j)
	buff[j] = pbest[j];
      MPI_Bcast(&buff[0],_ndim,MPI_DOUBLE,ibest,W_COMM);
    }

    // Broadcast the best point to every process.
    // Assumes rank=0 is in W_rank=0.  This should be true given the specification of the MPI_Comm_split function.
    MPI_Bcast(&buff[0],_ndim,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (size_t j=0; j<_ndim; ++j)
      pbest[j] = buff[j];

    // Output best point in the output summary file
    if (W_rank==0 && L_rank==0) 
    {
      optout << std::endl;
      optout << std::setw(15) << Lbest;
      for (size_t j=0; j<_ndim; ++j)
	optout << std::setw(15) << buff[j];
      optout << std::endl;
    }

    // Clean up the memory allocation
    delete[] Lbest_array;
    delete[] buff;


    std::cout << "Rank (" << rank << ", " << W_rank << ", " << L_rank << ") optimizer_kickout_powell::run_optimizer finished at Lbest\n";
    
    // Return best point
    return pbest;
  }

  std::vector<double> optimizer_kickout_powell::run_optimizer(likelihood& L, int dof_estimate, std::vector<double> start_parameter_values, std::string optimizer_results_filename, size_t number_of_instances, size_t number_of_restarts, double tolerance)
  {
    // Get the number of parameters for the model
    _ndim = L.priors().size();

    // If the number_of_instances=0, set to be the maximum number permitted by the currect number of processes.
    if (number_of_instances==0)
    {
      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      number_of_instances = size/_num_likelihood;
    }

    // Create a vector of vectors and push back a bunch of start positions
    std::vector< std::vector<double> > start_plist;
    start_plist.push_back(start_parameter_values); // Add the one parameter vector passed
    for (size_t j=1; j<number_of_instances; ++j)
      start_plist.push_back(generate_start_point(L));

    // Run the optimizer
    return ( run_optimizer(L,dof_estimate,start_plist,optimizer_results_filename,number_of_restarts,tolerance) );
  }

  std::vector<double> optimizer_kickout_powell::run_optimizer(likelihood& L, int dof_estimate, std::string optimizer_results_filename, size_t number_of_instances, size_t number_of_restarts, double tolerance)
  {
    // Get the number of parameters for the model
    _ndim = L.priors().size();

    // If the number_of_instances=0, set to be the maximum number permitted by the currect number of processes.
    if (number_of_instances==0)
    {
      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      number_of_instances = size/_num_likelihood;
    }

    // Create a vector of vectors and push back a bunch of start positions
    std::vector< std::vector<double> > start_plist;
    for (size_t j=0; j<number_of_instances; ++j)
      start_plist.push_back(generate_start_point(L));

    // Run the optimizer
    return ( run_optimizer(L,dof_estimate,start_plist,optimizer_results_filename,number_of_restarts,tolerance) );
  }


  std::vector<double> optimizer_kickout_powell::generate_start_point(likelihood& L)
  {
    std::vector<double> p(_ndim);
    
    for (size_t k=0; k<_ndim; ++k)
    {
      double pmin = L.priors()[k]->lower_bound();
      double pmax = L.priors()[k]->upper_bound();

      if (pmin==-std::numeric_limits<double>::infinity() && pmax==std::numeric_limits<double>::infinity())
	p[k] = std::tan(0.5*M_PI*(2.0*_rng.rand()-1.0));
      else if (pmin==-std::numeric_limits<double>::infinity())
	p[k] = pmax + std::log(_rng.rand());
      else if (pmax==std::numeric_limits<double>::infinity())
	p[k] = pmin - std::log(_rng.rand());
      else
	p[k] = (pmax-pmin)*_rng.rand() + pmin;
    }

    return p;
  }

  double optimizer_kickout_powell::get_optimal_point(std::vector<double>& pstart, double tolerance, int itermax)
  {  
    double *p = new double[_ndim+1]; // Unit-offset parameter vector
    double **xi = new double*[_ndim+1]; // Unit-offset local 
    for (size_t j=1; j<_ndim+1; ++j)
    {
      // Initialize the xi matrix to the identity (with unit offset)
      xi[j] = new double[_ndim+1];
      for (size_t k=1; k<_ndim+1; ++k)
	xi[j][k] = (j==k ? 1.0 : 0);

      // Initialize the first point to the point (with unit offset)
      p[j] = pstart[j-1];
    }

    // Minimize via Powell
    int iter; // Current not used
    double fret; // Return minimized function value (-L)
    powell(p,xi,tolerance,iter,fret,itermax);

    std::cerr << "CONGRATS! optimizer_kickout_powell.get_optimal_point finished in " << iter << " evaluations to " << fret << "\n";

    // Save best point
    for (size_t j=0; j<_ndim; ++j)
      pstart[j] = p[j+1];

    // Cleanup memory allocations
    for (size_t j=1; j<_ndim+1; ++j)
      delete[] xi[j];
    delete[] xi;
    delete[] p;

    // Return best likelihood
    return (-fret);
  }



  double optimizer_kickout_powell::func(double p[])
  {
    std::vector<double> pvec(_ndim);
    for (size_t j=0; j<_ndim; ++j)
      pvec[j] = p[j+1];
    return ( tempered_func(pvec,_prior_edge_beta) );
  }

  double optimizer_kickout_powell::tempered_func(std::vector<double> p, double prior_edge_beta)
  {
    double tempering_mask=0.0;
    
    for (size_t j=0; j<_ndim; ++j)
    {
      double pmin=_Lptr->priors()[j]->lower_bound();
      double pmax=_Lptr->priors()[j]->upper_bound();
      double dp,db=1.0-prior_edge_beta;
      if (pmin>-std::numeric_limits<double>::infinity() && pmax<std::numeric_limits<double>::infinity())
      {
	dp = (p[j]-pmin)/(pmax-pmin) * 10.0;
	tempering_mask += std::log( std::max(0.0, dp<db ? 1.0 - (dp-db)*(dp-db)/(db*db) : 1.0 ) );
	dp = (pmax-p[j])/(pmax-pmin) * 10.0;
	tempering_mask += std::log( std::max(0.0, dp<db ? 1.0 - (dp-db)*(dp-db)/(db*db) : 1.0 ) );
      }
      else if (pmin==-std::numeric_limits<double>::infinity() && pmax<std::numeric_limits<double>::infinity())
      {
	dp = (pmax-p[j]);
	tempering_mask += std::log( std::max(0.0, dp<db ? 1.0 - (dp-db)*(dp-db)/(db*db) : 1.0 ) );	
      }
      else if (pmin>-std::numeric_limits<double>::infinity() && pmax==std::numeric_limits<double>::infinity())
      {
	dp = (p[j]-pmin);
	tempering_mask += std::log( std::max(0.0, dp<db ? 1.0 - (dp-db)*(dp-db)/(db*db) : 1.0 ) );
      }
      //otherwise both are infinite and the tempering_mask is unchanged.
    }
    
    tempering_mask *= 2*_dof_estimate;

    return ( -(_Lptr->operator()(p)+tempering_mask) );
  }

  double optimizer_kickout_powell::f1dim(double x)
  {
    int j;
    double f;
    double *xt = new double[_ndim+1];
    for (j=1;j<=int(_ndim);j++) 
      xt[j]=_pcom[j]+x*_xicom[j];
    f=func(xt);
    delete[] xt;
    return f;
  }
  
#define ITMAX 200
  
  void optimizer_kickout_powell::powell(double p[], double **xi, double ftol, int& iter, double& fret, int itermax)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int n = int(_ndim);

    if (itermax==0)
      itermax=ITMAX+1;


    int i,ibig,j;
    double del,fp,fptt,t;
    
    double *pt = new double[n+1];
    double *ptt = new double[n+1];
    double *xit = new double[n+1];
    
    _pcom = new double[n+1];
    _xicom = new double[n+1];
    
    fret=func(p);
    for (j=1;j<=n;j++) pt[j]=p[j];
    for (iter=1;;++iter) {

      //std::cout << "Rank " << rank << " Powell step " << iter << " with beta " << _prior_edge_beta << " and f " << fret << std::endl;

      fp=fret;
      ibig=0;
      del=0.0;
      for (i=1;i<=n;i++) {
	for (j=1;j<=n;j++) xit[j]=xi[j][i];
	fptt=fret;
	linmin(p,xit,fret);
	if (std::fabs(fptt-fret) > del) {
	  del=std::fabs(fptt-fret);
	  ibig=i;
	}
      }
      if (2.0*std::fabs(fp-fret) <= ftol*(std::fabs(fp)+std::fabs(fret)) || iter==itermax) {
	delete[] pt;
	delete[] ptt;
	delete[] xit;
	delete[] _pcom;
	delete[] _xicom;
	return;
      }
      if (iter == ITMAX) {
	std::cerr << "WARNING! In optimizer_kickout_powell::powell ITMAX=" << ITMAX << " exceeded\n";
	delete[] pt;
	delete[] ptt;
	delete[] xit;
	delete[] _pcom;
	delete[] _xicom;
	return;
      }
      for (j=1;j<=n;j++) {
	ptt[j]=2.0*p[j]-pt[j];
	xit[j]=p[j]-pt[j];
	pt[j]=p[j];
      }
      fptt=func(ptt);
      if (fptt < fp) {
	t=2.0*(fp-2.0*fret+fptt)*(fp-fret-del)*(fp-fret-del) - del*(fp-fptt)*(fp-fptt);
	if (t < 0.0) {
	  linmin(p,xit,fret);
	  for (j=1;j<=n;j++) {
	    xi[j][ibig]=xi[j][n];
	    xi[j][n]=xit[j];
	  }
	}
      }
    }
  }
#undef ITMAX
  
#define TOL 2.0e-4
  void optimizer_kickout_powell::linmin(double p[], double xi[], double& fret)
  {
    int n = int(_ndim);
    int j;
    double xx,xmin,fx,fb,fa,bx,ax;

    for (j=1;j<=n;j++) {
      _pcom[j]=p[j];
      _xicom[j]=xi[j];
    }
    ax=0.0;
    xx=1.0;
    mnbrak(&ax,&xx,&bx,&fa,&fx,&fb);
    fret=brent(ax,xx,bx,TOL,xmin);
    for (j=1;j<=n;j++) {
      xi[j] *= xmin;
      p[j] += xi[j];
    }
  }
#undef TOL


#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SIGN(a,b) ((b) >= 0.0 ? std::fabs(a) : -std::fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

  double optimizer_kickout_powell::brent(double ax, double bx, double cx, double tol, double& xmin)
  {
    int iter;
    double a,b,d=0.0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
    double e=0.0;

    a=(ax < cx ? ax : cx);
    b=(ax > cx ? ax : cx);
    x=w=v=bx;
    fw=fv=fx=f1dim(x);
    for (iter=1;iter<=ITMAX;iter++) {
      xm=0.5*(a+b);
      tol2=2.0*(tol1=tol*std::fabs(x)+ZEPS);
      if (std::fabs(x-xm) <= (tol2-0.5*(b-a))) {
	xmin=x;
	return fx;
      }
      if (std::fabs(e) > tol1) {
	r=(x-w)*(fx-fv);
	q=(x-v)*(fx-fw);
	p=(x-v)*q-(x-w)*r;
	q=2.0*(q-r);
	if (q > 0.0) 
	  p = -p;
        q=std::fabs(q);
	etemp=e;
	e=d;
        if (std::fabs(p) >= std::fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
	  d=CGOLD*(e=(x >= xm ? a-x : b-x));
	else {
	  d=p/q;
	  u=x+d;
	  if (u-a < tol2 || b-u < tol2)
	    d=SIGN(tol1,xm-x);
	}
      } else {
	d=CGOLD*(e=(x >= xm ? a-x : b-x));
      }
      u=(std::fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
      fu=f1dim(u);
      if (fu <= fx) {
	if (u >= x) 
	  a=x; 
	else 
	  b=x;
        SHFT(v,w,x,u);
        SHFT(fv,fw,fx,fu);
      } else {
	if (u < x) 
	  a=u; 
	else 
	  b=u;
	if (fu <= fw || w == x) {
	  v=w;
	  w=u;
	  fv=fw;
	  fw=fu;
	} else if (fu <= fv || v == x || v == w) {
	  v=u;
	  fv=fu;
	}
      }
    }
    std::cerr << "WARNING! In optimizer_kickout_powell::brent too many interations\n";
    xmin=x;
    return fx;
  }
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT


#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

  void optimizer_kickout_powell::mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc)
  {
    double ulim,u,r,q,fu,dum;

    *fa=f1dim(*ax);
    *fb=f1dim(*bx);
    if (*fb > *fa) {
      SHFT(dum,*ax,*bx,dum);
      SHFT(dum,*fb,*fa,dum);
    }
    *cx=(*bx)+GOLD*(*bx-*ax);
    *fc=f1dim(*cx);
    while (*fb > *fc) {
      r=(*bx-*ax)*(*fb-*fc);
      q=(*bx-*cx)*(*fb-*fa);
      u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
	(2.0*SIGN(std::max(std::fabs(q-r),TINY),q-r));
      ulim=(*bx)+GLIMIT*(*cx-*bx);
      if ((*bx-u)*(u-*cx) > 0.0) {
	fu=f1dim(u);
	if (fu < *fc) {
	  *ax=(*bx);
	  *bx=u;
	  *fa=(*fb);
	  *fb=fu;
	  return;
	} else if (fu > *fb) {
	  *cx=u;
	  *fc=fu;
	  return;
	}
	u=(*cx)+GOLD*(*cx-*bx);
	fu=f1dim(u);
      } else if ((*cx-u)*(u-ulim) > 0.0) {
	fu=f1dim(u);
	if (fu < *fc) {
          SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx));
          SHFT(*fb,*fc,fu,f1dim(u));
        }
      } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
	u=ulim;
	fu=f1dim(u);
      } else {
	u=(*cx)+GOLD*(*cx-*bx);
	fu=f1dim(u);
      }
      SHFT(*ax,*bx,*cx,u);
      SHFT(*fa,*fb,*fc,fu);
    }
  }
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT

}

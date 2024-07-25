/*!
  \file optimizer_simplex.cpp
  \author Avery Broderick
  \brief Implementation file for the optimizer_simplex class
*/


#include "optimizer_simplex.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>


namespace Themis{

  optimizer_simplex::optimizer_simplex(int seed)
    : _num_likelihood(1), _rng(seed)
  {
  }

  optimizer_simplex::~optimizer_simplex()
  {
  }

  void optimizer_simplex::set_cpu_distribution(int num_likelihood)
  {
    _num_likelihood = num_likelihood;
  }


  std::vector<double> optimizer_simplex::run_optimizer(likelihood& L, std::vector< std::vector<double> > start_parameter_values, std::string optimizer_results_filename, size_t number_of_restarts, size_t maximum_iterations, double tolerance)
  {
    // Set pointer to likelihood object
    _Lptr = &L;

    // Get the number of parameters for the model
    _ndim = L.priors().size();
   
    // Set the number of instances
    size_t number_of_instances = start_parameter_values.size();
    
    // Create a set of communicators for the likelihoods and simplexes
    //  Communicator, ranks, communicator construction
    MPI_Comm W_COMM, L_COMM;
    MPI_Status Stat;
    int rank, color, hue, W_size, W_rank, L_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //   Likelihoods
    color = rank / _num_likelihood;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &L_COMM);
    MPI_Comm_rank(L_COMM, &L_rank);
    //   Cross-simplex communication (necessary only among rank 0)
    hue = rank%_num_likelihood;
    MPI_Comm_split(MPI_COMM_WORLD, hue, rank, &W_COMM);
    MPI_Comm_rank(W_COMM, &W_rank);
    MPI_Comm_size(W_COMM, &W_size);
    
    // Set the likelihood commmunicator 
    _Lptr->set_mpi_communicator(L_COMM);

    // Open the optimizer summary output file
    std::ofstream optout;
    if (W_rank==0 && L_rank==0)
      optout.open(optimizer_results_filename);

    // Make some space for saving best values
    std::vector<double> Lbest_list;
    std::vector< std::vector<double> > pbest_list;
    double* buff = new double[_ndim+1];
    for (size_t i=0; i<number_of_instances; ++i)
    {
      if (i%W_size==size_t(W_rank)) // If this process is tasked with this instance
      {
	// Repeat with the requested number of restarts
	for (size_t r=0; r<number_of_restarts; ++r)
	  get_optimal_point(start_parameter_values[i],tolerance,maximum_iterations);

	// Save the likelihood of the best
	Lbest_list.push_back( get_optimal_point(start_parameter_values[i],tolerance,maximum_iterations) );

	// Save the position of the best
	pbest_list.push_back( start_parameter_values[i] );

	// Send the best to the master
	buff[0] = Lbest_list[Lbest_list.size()-1];
	for (size_t j=0; j<_ndim; ++j)
	  buff[j+1] = pbest_list[Lbest_list.size()-1][j];
	// (only send if this is not the master)
	if (L_rank==0 && W_rank!=0)
	  MPI_Send(buff,_ndim+1,MPI_DOUBLE,0,10,W_COMM);
      }

      // If this is the master, output to the optimizer summary file
      if (W_rank==0 && L_rank==0) 
      {
	// (only receive if sent from something other than the master)
	if (i%W_size!=0)
	  MPI_Recv(buff,_ndim+1,MPI_DOUBLE,i%W_size,10,W_COMM, &Stat);

	// Write output to the summary file
	for (size_t j=0; j<_ndim+1; ++j)
	  optout << std::setw(15) << buff[j];
	optout << std::endl;
      }
    }

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

    // Return best point
    return pbest;
  }

  std::vector<double> optimizer_simplex::run_optimizer(likelihood& L, std::vector<double> start_parameter_values, std::string optimizer_results_filename, size_t number_of_instances, size_t number_of_restarts, size_t maximum_iterations, double tolerance)
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
    return ( run_optimizer(L,start_plist,optimizer_results_filename,number_of_restarts,maximum_iterations,tolerance) );
  }

  std::vector<double> optimizer_simplex::run_optimizer(likelihood& L, std::string optimizer_results_filename, size_t number_of_instances, size_t number_of_restarts, size_t maximum_iterations, double tolerance)
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
    return ( run_optimizer(L,start_plist,optimizer_results_filename,number_of_restarts,maximum_iterations,tolerance) );
  }


  std::vector<double> optimizer_simplex::generate_start_point(likelihood& L)
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

  double optimizer_simplex::get_optimal_point(std::vector<double>& pstart, double tolerance, size_t maximum_iterations)
  {  
    // Allocate and initialize arrays of the parameters and -likelihood values
    double **p = new double*[_ndim+1];
    double *y = new double[_ndim+1];
    std::vector<double> ptmp(_ndim);
    for (size_t j=0; j<_ndim+1; ++j)
    {
      p[j] = new double[_ndim];
	  
      do {

	for (size_t k=0; k<_ndim; ++k)
	{
	  double p0 = pstart[k];
	  double pmin = _Lptr->priors()[k]->lower_bound();
	  double pmax = _Lptr->priors()[k]->upper_bound();
	  
	  if (pmin==-std::numeric_limits<double>::infinity())
	    pmin = p0-1.0;
	  if (pmax==std::numeric_limits<double>::infinity())
	    pmax = p0+1.0;
	  
	  if (pmin>=p0 || pmax<=p0) 
	  {
	    std::cerr << "ERROR! Start position for p[" << k << "] is outside of or against prior bounds\n";
	    std::exit(1);
	  }

	  double dp = 1.0e-4*std::min(std::fabs(pmax-p0),std::fabs(pmin-p0));
	  p[j][k] = p0 + dp*(2.0*_rng.rand()-1.0);
	  
	  ptmp[k] = p[j][k];
	}
	
	y[j] = -_Lptr->operator()(ptmp);
	
      } while (y[j]==-std::numeric_limits<double>::infinity());
    }

    // Minimize via amoeba for simplexes in the list to be used
    amoeba(p,y,tolerance,maximum_iterations);
    
    // Find the best fit among those on the simplex
    size_t ibest = 0;
    double Lbest = -y[0];
    for (size_t j=1; j<_ndim+1; ++j)
      if (-y[j]>Lbest) 
      {
	Lbest = y[j];
	ibest = j;
      }
    for (size_t k=0; k<_ndim; ++k)
      pstart[k] = p[ibest][k];

    // Cleanup memory allocations
    for (size_t j=0; j<_ndim+1; ++j)
      delete[] p[j];
    delete[] p;
    delete[] y;

    return Lbest;
  }





#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

  void optimizer_simplex::amoeba(double **p, double y[], double ftol, int NMAX)
  {
    int i,ihi,ilo,inhi,j,mpts=_ndim+1;
    double rtol,sum,swap,ysave,ytry;
    int nfunk=0;
    
    std::vector<double> psum(_ndim);
    for (j=0;j<int(_ndim);j++) 
    {
      for (sum=0.0,i=0; i<mpts; i++) 
	sum += p[i][j];
      psum[j]=sum;
    }

    for (;;) 
    {
      ilo=0;
      ihi = y[0]>y[1] ? (inhi=1,0) : (inhi=0,1);
      for (i=0; i<mpts; i++)
      {
	if (y[i] <= y[ilo]) 
	  ilo=i;
	if (y[i] > y[ihi])
	{
	  inhi=ihi;
	  ihi=i;
	} 
	else if (y[i] > y[inhi] && i != ihi) 
	  inhi=i;
      }
      rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo]));
      if (rtol < ftol) 
      {
	SWAP(y[0],y[ilo]);
	for (i=0; i<int(_ndim); i++) 
	  SWAP(p[0][i],p[ilo][i]);
	break;
      }
      if (nfunk >= NMAX)
      {
	SWAP(y[0],y[ilo]);
	for (i=0; i<int(_ndim); i++) 
	  SWAP(p[0][i],p[ilo][i]);
	std::cerr << "WARNING! In optimizer_simplex::ameoba NMAX exceeded\n";
	break;
      }
      nfunk += 2;

      ytry=amotry(p,y,psum,ihi,-1.0);
      if (ytry <= y[ilo])
	ytry=amotry(p,y,psum,ihi,2.0);
      else if (ytry >= y[inhi]) {
	ysave=y[ihi];
	ytry=amotry(p,y,psum,ihi,0.5);
	if (ytry >= ysave) 
	{
	  for (i=0; i<mpts; i++)
	  {
	    if (i != ilo)
	    {
	      for (j=0; j<int(_ndim); j++)
	      {
		p[i][j] = psum[j] = 0.5*(p[i][j]+p[ilo][j]);
	      }
	      y[i]=-_Lptr->operator()(psum);
	    }
	  }
	  nfunk += int(_ndim);

	  for (j=0; j<int(_ndim); j++) 
	  {
	    for (sum=0.0,i=0; i<mpts; i++) 
	      sum += p[i][j];
	    psum[j]=sum;
	  }
	}
      } 
      else 
	--nfunk;
    }
  }
#undef SWAP

  double optimizer_simplex::amotry(double **p, double y[], std::vector<double>& psum, int ihi, double fac)
  {
    int j;
    double fac1,fac2,ytry;
    std::vector<double> ptry(_ndim);
    
    fac1 = (1.0-fac)/double(_ndim);
    fac2 = fac1-fac;
    
    for (j=0; j<int(_ndim); j++) 
      ptry[j] = psum[j]*fac1-p[ihi][j]*fac2;
    ytry = -_Lptr->operator()(ptry);
    
    if (ytry < y[ihi]) 
    {
      y[ihi]=ytry;
      for (j=0;j<int(_ndim);j++) 
      {
	psum[j] += ptry[j]-p[ihi][j];
	p[ihi][j]=ptry[j];
      }
    }

    return ytry;
  }

}

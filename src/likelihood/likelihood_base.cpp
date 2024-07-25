/*!
  \file likelihood_base.cpp
  \author 
  \date  April, 2017
  \brief Implementation file for the Base Likelihood class
*/

#include "likelihood_base.h"  
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace Themis {

  //int data_set_index_foo = 0;
  
  likelihood_base::likelihood_base() 
    : _logLikelihood(0.0), _xlast(0,0), _comm(MPI_COMM_WORLD), _N_model(1), _size(1)
  {
    initialize_mpi();

    /*
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::stringstream fname;
    fname << "proc_" << std::setw(3) << std::setfill('0') << rank << "_" << data_set_index_foo << ".err";
    _procerr.open(fname.str());
    Themis::data_set_index_foo += 1;
    */


    _grad_local_buff = new double[_size];
    _grad_global_buff = new double[_size];
  }
  
  likelihood_base::~likelihood_base() 
  {
    delete[] _grad_local_buff;
    delete[] _grad_global_buff;    
  }
  
  double likelihood_base::operator() (std::vector<double>& x) 
  { 
    if (x!=_xlast)
    {
      _xlast=x;
      _logLikelihood=0.0;
    }
    return ( _logLikelihood ); 
  }

  
  
  std::vector<double> likelihood_base::gradient(std::vector<double>& x, prior& Pr)
  {
    // Default computation is a centered finite difference
    std::vector<double> y=x;
    double h;

    if (_size!=x.size()) {
      delete[] _grad_local_buff;
      delete[] _grad_global_buff;
      _size = x.size();
      _local_size = _size/_L_size;
      if (_local_size*_L_size<int(_size))
	_local_size += 1;
      _global_size = _local_size*_L_size;
      _grad_local_buff = new double[_global_size];
      _grad_global_buff = new double[_global_size];
    }
    memset( _grad_local_buff, 0.0, _global_size*sizeof(double));
    memset( _grad_global_buff, 0.0, _global_size*sizeof(double));
    
    int j;
    for (size_t i=0; i<x.size(); ++i)
    {
      if ( i/_local_size == size_t(_L_rank) )
      {
	j = i%_local_size;
	
	// Obtain adaptive stepsize
	h = step_size(std::fabs(Pr.upper_bound(i)-Pr.lower_bound(i)));
	
	// Forward step
	y[i] = x[i]+h;
	if (std::isfinite(Pr(y)))
	  _grad_local_buff[j] += this->operator()(y);
	else
	  _grad_local_buff[j] = -std::numeric_limits<double>::infinity();
	
	// Backward step
	y[i] = x[i]-h;
	if (std::isfinite(Pr(y)))
	  _grad_local_buff[j] -= this->operator()(y);
	else
	  _grad_local_buff[j] = std::numeric_limits<double>::infinity();

	_grad_local_buff[j] /= (2.0*h);
	
	// Return and complete
	y[i] = x[i];
      }
    }

    // Distribute gradient evaluation information
    MPI_Allgather(_grad_local_buff,_local_size,MPI_DOUBLE,_grad_global_buff,_local_size,MPI_DOUBLE,_Lcomm);

    // Repackage for return
    std::vector<double> grad(x.size());
    for (size_t i=0; i<x.size(); ++i)
      grad[i] = _grad_global_buff[i];

    return grad;
  }

  
#if 0  // Parallelized and using Allreduce  
  std::vector<double> likelihood_base::gradient(std::vector<double>& x, prior& Pr)
  {

    // Default computation is a centered finite difference
    std::vector<double> y=x;
    double h;

    if (_size!=x.size()) {
      delete[] _grad_local_buff;
      delete[] _grad_global_buff;
      _size = x.size();
      _grad_local_buff = new double[_size];
      _grad_global_buff = new double[_size];
    }
    memset( _grad_local_buff, 0.0, x.size()*sizeof(double));
    memset( _grad_global_buff, 0.0, x.size()*sizeof(double));
    

    for (size_t i=0; i<x.size(); ++i)
    {
      
      if ((2*int(i)+0)%_L_size==_L_rank)
      {
	// Obtain adaptive stepsize
	h = step_size(std::fabs(Pr.upper_bound(i)-Pr.lower_bound(i)));
	
	// Forward step
	y[i] = x[i]+h;
	if (std::isfinite(Pr(y)))
	  _grad_local_buff[i] += this->operator()(y) / (2.0*h);
	else
	  _grad_local_buff[i] = -std::numeric_limits<double>::infinity();
	
	// Return and complete
	y[i] = x[i];
      }
	
      if ((2*int(i)+1)%_L_size==_L_rank)
      {
	// Obtain adaptive stepsize
	h = step_size(std::fabs(Pr.upper_bound(i)-Pr.lower_bound(i)));

	// Backward step
	y[i] = x[i]-h;
	if (std::isfinite(Pr(y)))
	  _grad_local_buff[i] -= this->operator()(y) / (2.0*h);
	else
	  _grad_local_buff[i] = std::numeric_limits<double>::infinity();

	// Return and complete
	y[i] = x[i];
      }
    }
    
    // Distribute gradient evaluation information
    MPI_Allreduce(_grad_local_buff,_grad_global_buff,x.size(),MPI_DOUBLE,MPI_SUM,_Lcomm);

    // Repackage for return
    std::vector<double> grad(x.size());
    for (size_t i=0; i<x.size(); ++i)
      grad[i] = _grad_global_buff[i];

    return grad;
  }
#endif  

  
  std::vector<double> likelihood_base::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
    int comm_size, comm_rank;
    MPI_Comm_size(_Lcomm, &comm_size);
    MPI_Comm_rank(_Lcomm, &comm_rank);

    /*
    // Distribute x
    double *xbuff = new double[x.size()];
    for (size_t i=0; i<x.size(); ++i)
      xbuff[i] = x[i];
    MPI_Bcast(xbuff,x.size(),MPI_DOUBLE,0,_Lcomm);
    for (size_t i=0; i<x.size(); ++i)
      x[i] = xbuff[i];
    delete[] xbuff;
    */
    
    // Default computation is a centered finite difference
    std::vector<double> grad(x.size(),0.0);
    std::vector<double> y=x;
    double h;
    for (size_t i=0; i<x.size(); ++i)
    {
      // Obtain adaptive stepsize
      h = step_size(std::fabs(Pr.upper_bound(i)-Pr.lower_bound(i)));
	
      // Forward step
      y[i] = x[i]+h;
      if (std::isfinite(Pr(y)))
	grad[i] = this->operator()(y);
      else
	grad[i] = -std::numeric_limits<double>::infinity();
      
      // Backward step
      y[i] = x[i]-h;
      if (std::isfinite(Pr(y)))
	grad[i] -= this->operator()(y);
      else
	grad[i] = std::numeric_limits<double>::infinity();
      
      // Return and complete
      y[i] = x[i];
      grad[i] /= (2.0*h);
    }

    // _procerr << "Grad"
    // 	     << std::setw(5) << wrank
    // 	     << std::setw(5) << comm_rank
    // 	     << std::setw(15) << this->operator()(x);
    // _procerr << " | ";
    // for (size_t i=0; i<x.size(); ++i)
    //   _procerr << std::setw(15) << x[i];
    // _procerr << " | ";
    // for (size_t i=0; i<x.size(); ++i)
    //   _procerr << std::setw(15) << grad[i];
    // _procerr << '\n';
    // _procerr.flush();

    
    return grad;
  }
  
  double likelihood_base::chi_squared(std::vector<double>& x) 
  { 
    return ( -2.0*operator()(x) );
  }

  void likelihood_base::set_mpi_communicator(MPI_Comm comm)
  {
    _comm = comm;
    initialize_mpi();
  }

  void likelihood_base::set_cpu_distribution(int num_model)
  {    
    _N_model = num_model;
  }

  void likelihood_base::initialize_mpi()
  {
    // Start size from current comm
    int size, rank;
    MPI_Comm_size(_comm,&size);
    MPI_Comm_rank(_comm,&rank);

    if (size%_N_model != 0 )
    {
      std::cerr << "Must use all processors!  Likelihod/model CPU distribution does not add up.\n";
      std::exit(1);
    }

    int num_likelihood = size/_N_model;

    int color, flavour;
    int M_rank, M_size;
    if (rank == 0)
      std::cout << "Setting likelihood cpu distributions and splitting MPI processes\n";
    //color communicates between the samplers of the same temp.
    color = rank/(size/num_likelihood);
    MPI_Comm_split(_comm, color, rank, &_Mcomm);
    MPI_Comm_rank(_Mcomm, &M_rank);
    MPI_Comm_size(_Mcomm, &M_size);
 
    //flavour communicates between the different temperatures.
    flavour = rank % (size/num_likelihood);
    MPI_Comm_split(_comm, flavour, rank, &_Lcomm);
    MPI_Comm_rank(_Lcomm, &_L_rank);
    MPI_Comm_size(_Lcomm, &_L_size);
    
    //Reset the seed to ensure that the likelihoods get the same random seed so they get the same position.
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
    for(int k =-1; k< size; k++)
    {
      if (k<0 && rank==0)
	std::cout << "Likelihood breakdown:\nW C  M  L" << std::endl;
      if (rank == k)
	std::cout << wrank << " " << rank << "  " << M_rank << "  " << _L_rank << std::endl;
      MPI_Barrier(_comm);
    }
  }
  
  
  void likelihood_base::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    if (rank==0)
      std::cerr << "WARNING: Output not defined for this likelihood yet.\n"
		<< "         Please talk too the developer post haste!\n\n";
  }
  
  void likelihood_base::output_model_data_comparison(std::ostream& out)
  {
    output(out);
  }
  
  void likelihood_base::output_model_data_comparison(std::string filename)
  {
    int rank;
    //MPI_Comm_rank(_comm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ofstream out;
    if (rank==0)
      out.open(filename.c_str());

    output(out);

    if (rank==0)
      out.close();
  }
      


  
};


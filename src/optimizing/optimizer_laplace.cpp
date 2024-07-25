#include "optimizer_laplace.h"
#include "transform_base.h"
#include "transform_logit.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>


namespace Themis
{
optimizer_laplace::optimizer_laplace(likelihood& L, 
                                     std::vector<std::string> var_names,
                                     size_t dimension)
  :_L(&L), _epsilon(1e-6), _max_iterations(100), _max_linesearch(200), _scale(1), _num_likelihood(1), _start_points(0, std::vector<double>(0,0.0)), _nln(&L, dimension) 
{
  _var_names.resize(0);
  
  if ( var_names.size() != dimension )
  {
    std::cout << "Warning var_names is has the wrong dimension, filling it with the default names.\n";
    for ( size_t i = 0; i < dimension; ++i )
    {
      _var_names.push_back("p"+std::to_string(i));
    }
  }
  else
  {
    _var_names = var_names;
  }
  _dimension = dimension;

  //Now assign the parameter bounds
  _lower = Eigen::VectorXd::Constant(dimension,0.0);
  _upper = Eigen::VectorXd::Constant(dimension,0.0);
  for ( size_t i = 0; i < dimension; ++i )
  {
    double upper = L.priors()[i]->upper_bound();
    double lower = L.priors()[i]->lower_bound();
    if ( (std::isfinite(lower)) && std::isfinite(upper) ){
      _transforms.push_back(new transform_logit(lower, upper));
      //_transforms.push_back(new transform_none());
      _upper(i) = 5;
      _lower(i) = -5;
    }else if ( std::isfinite(lower) ){
      _transforms.push_back(new transform_none());
      _lower(i) = lower + 1e-4;
      _upper(i) = std::numeric_limits<double>::infinity();
    }else if ( std::isfinite(upper) ){
      _transforms.push_back(new transform_none());
      _upper(i) = upper - 1e-4;
      _lower(i) = -std::numeric_limits<double>::infinity();
    }else{
      _transforms.push_back(new transform_none());
      _lower(i) = -std::numeric_limits<double>::infinity();
      _upper(i) = std::numeric_limits<double>::infinity();
    }
  }
  _nln.set_tranforms(_transforms);
}


void optimizer_laplace::set_scale(double scale)
{
  _scale = scale;
}

void optimizer_laplace::_nlogprob::set_scale(double scale)
{
  __scale = scale;
}

double optimizer_laplace::_nlogprob::loglikelihood_trans(const Eigen::VectorXd& x)
{
  std::vector<double> params(this->__dimension, 0.0);
  double jacobian = 0.0;
  for ( size_t i = 0; i < this->__dimension; ++i ){
    params[i] = __transforms[i]->inverse(x(i));
    jacobian += std::log(std::fabs(__transforms[i]->inverse_jacobian(x(i))+1e-50));
  }
  return (this->__L->operator()(params) + jacobian);
}


double optimizer_laplace::_nlogprob::loglikelihood_trans(const std::vector<double>& x)
{
  std::vector<double> params(this->__dimension, 0.0);
  double jacobian = 0.0;
  for ( size_t i = 0; i < this->__dimension; ++i ){
    params[i] = __transforms[i]->inverse(x[i]);
    jacobian += std::log(std::fabs(__transforms[i]->inverse_jacobian(x[i])+1e-50));
  }
  return (this->__L->operator()(params) + jacobian);
}

double optimizer_laplace::_nlogprob::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
{
  grad.resize(this->__dimension);
  std::vector<double> params(this->__dimension, 0.0);
  double jacobian = 0.0;
  for ( size_t i = 0; i < this->__dimension; ++i ){
    params[i] = __transforms[i]->inverse(x[i]);
    jacobian += std::log(std::fabs(__transforms[i]->inverse_jacobian(x[i])+1e-50));
  }
  double l = this->loglikelihood_trans(x)/__scale;
  std::vector<double> gradient = this->__L->gradient(params);
  for ( size_t i = 0; i < this->__dimension; ++i )
  { 
    double jac = __transforms[i]->inverse_jacobian(x(i));
    double dljac =  (std::expm1(-x(i)))/(std::exp(-x(i))+1);
    grad[i] = (-gradient[i]*jac - dljac)/__scale;
  }
  if ( !std::isfinite(l) || std::isnan(l) )
  {
    for ( size_t i = 0; i < this->__dimension; ++i )
      grad[i] = 0.0;
    return std::numeric_limits<double>::infinity();
  }

  return -l;
}

void optimizer_laplace::set_cpu_distribution(int num_likelihood)
{
  _num_likelihood = num_likelihood;
}



void optimizer_laplace::set_start_points(std::vector<std::vector<double> > start_points)
{
  _start_points = start_points;
}

int optimizer_laplace::parallel_optimizer(std::vector<double>& parameters, double& max, size_t number_of_instances, int seed, std::string optname)
{
  
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
  
  Ran2RNG rng(seed+rank); 
    
  // Output my current information
  std::cout << "optimizer_laplace.parallel_optimizer : rank=" << rank << " L_rank=" << L_rank << " W_rank=" << W_rank << std::endl;


  // Set the likelihood commmunicator 
  _L->set_mpi_communicator(L_COMM);
 

  // Open the optimizer summary output file
  std::ofstream optout;
  if (W_rank==0 && L_rank==0)
    optout.open(optname);

  // Loop through and save best values
  //   Make some space for saving best values
  std::vector<double> Lbest_list;
  std::vector< std::vector<double> > pbest_list;
  double* buff = new double[_dimension+1];
  int counter = 0;
  int finish = number_of_instances/W_size;
  for (size_t i=0; i<number_of_instances; ++i)
  {
    std::vector<double> pstart(_dimension, 0.0);
    if (i%W_size==size_t(W_rank)) // If this process is tasked with this instance
    {
      clock_t start_t = clock();
      //Generate start point an distribute is across the likelihood
      //std::cout << "W_rank=" << W_rank << " started optimization run" << std::endl;
      if ( i > _start_points.size()-1)
        pstart = generate_start_point(rng);
      else
        pstart = _start_points[i];
      for (size_t k=0; k<_dimension; ++k)
	  buff[k] = pstart[k];
      MPI_Bcast(&buff[0],_dimension,MPI_DOUBLE,0,L_COMM);
      for (size_t k=0; k<_dimension; ++k)
	  pstart[k] = buff[k];	    
      

      double lopt = 0.0;
      this->run_optimizer(pstart, lopt);
      pbest_list.push_back(pstart);
      Lbest_list.push_back(lopt);
      
    
      clock_t end_t = clock();
      //Output some QOL stuff
      counter++;
      if (L_rank == 0){
        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        std::stringstream message;
        message << "Iteration on worker rank "+std::to_string(W_rank)+": ";
        message << std::setw(it_print_width) << counter+1 << " / " << finish;
        message << " [" << std::setw(3)
                << static_cast<int>(((100.0*(counter))/finish)) << "%] ";
        message << "Last step took " << (end_t-start_t)/CLOCKS_PER_SEC << " s.";
        std::cout << message.str() << std::endl;
      }
    }
  }
  std::cerr << "Done optimizing on  " << W_rank << '\n';

  MPI_Barrier(MPI_COMM_WORLD);

   
  // Loop through and communicate best values to the master process to output
  for (size_t i=0; i<number_of_instances; ++i){
    if (i%W_size==size_t(W_rank)) // If this process is tasked with this instance
    {
      // Send the best to the master
      buff[0] = Lbest_list[Lbest_list.size()-1];
      for (size_t j=0; j<_dimension; ++j)
        buff[j+1] = pbest_list[Lbest_list.size()-1][j];
      if (L_rank==0 && W_rank!=0){
	MPI_Send(&buff[0],int(_dimension)+1,MPI_DOUBLE,0,50+int(W_rank),W_COMM);
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
	MPI_Recv(&buff[0],int(_dimension)+1,MPI_DOUBLE,int(i%W_size),50+int(i%W_size),W_COMM, &Stat[i]);
      }
      //std::cerr << "optimizer_kickout_powell::run_optimizer : bar2 " << W_rank << " " << L_rank << " " << i << '\n';
	
      // Write output to the summary file
      for (size_t j=0; j<_dimension+1; ++j)
	optout << std::setw(15) << buff[j];
      optout << std::endl;
    }
  }


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
    for (size_t j=0; j<_dimension; ++j)
      buff[j] = pbest[j];
    
    MPI_Bcast(&buff[0],_dimension,MPI_DOUBLE,ibest,W_COMM);
  }

  // Broadcast the best point to every process.
    // Assumes rank=0 is in W_rank=0.  This should be true given the specification of the MPI_Comm_split function.
  MPI_Bcast(&buff[0],_dimension,MPI_DOUBLE,0,MPI_COMM_WORLD);
  for (size_t j=0; j<_dimension; ++j)
    parameters[j] = buff[j];
  max = _L->operator()(parameters);

  // Output best point in the output summary file
  if (W_rank==0 && L_rank==0) 
  {
    optout << std::endl;
    optout << std::setw(15) << Lbest;
    for (size_t j=0; j<_dimension; ++j)
      optout << std::setw(15) << buff[j];
    optout << std::endl;
  }

  // Clean up the memory allocation
  delete[] Lbest_array;
  delete[] buff;

  std::cout << "Rank (" << rank << ", " << W_rank << ", " << L_rank << ") optimizer_laplace::parallel_optimizer finished at Lbest\n";

  return number_of_instances;

}

int optimizer_laplace::run_optimizer(std::vector<double>& parameters, double& max)
{
  Eigen::VectorXd x = Eigen::VectorXd::Constant(_dimension, 0.0);
  for ( size_t i = 0; i < _dimension; ++i )
    x(i) = parameters[i];
  int nitr = run_optimizer(x, max);
  for ( size_t i = 0; i < _dimension; ++i )
    parameters[i] = x(i);
  return nitr;
}


std::vector<double> optimizer_laplace::generate_start_point(Ran2RNG& rng)
{
  std::vector<double> p(_dimension);
    
  for (size_t k=0; k<_dimension; ++k)
  {
    double pmin = _L->priors()[k]->lower_bound();
    double pmax = _L->priors()[k]->upper_bound();

    if (pmin==-std::numeric_limits<double>::infinity() && pmax==std::numeric_limits<double>::infinity())
      p[k] = std::tan(0.5*M_PI*(2.0*rng.rand()-1.0));
    else if (pmin==-std::numeric_limits<double>::infinity())
      p[k] = pmax + std::log(rng.rand());
    else if (pmax==std::numeric_limits<double>::infinity())
      p[k] = pmin - std::log(rng.rand());
    else
      p[k] = (pmax-pmin)*rng.rand() + pmin;
  }

  return p;
}


void optimizer_laplace::set_parameters(double epsilon, size_t max_iterations, size_t max_linesearch)
{
  _epsilon = epsilon;
  _max_iterations = max_iterations;
  _max_linesearch = max_linesearch;
}

int optimizer_laplace::run_optimizer(Eigen::VectorXd& parameters, double& max)
{
  Eigen::VectorXd x = Eigen::VectorXd::Constant(_dimension, 0.0);
  LBFGSpp::LBFGSBParam<double> param;// New parameter class
  param.epsilon = _epsilon;
  param.max_iterations = _max_iterations;
  param.max_linesearch = _max_linesearch;
  LBFGSpp::LBFGSBSolver<double> solver(param);
  
  
  //Convert to the unbounded params
  max = -1.0*max;
  std::vector<double> p(_dimension, 0.0);
  for ( size_t i = 0; i < _dimension; ++i ){
    x(i) = _transforms[i]->forward(parameters[i]);
    p[i] = parameters[i]; 
  }
  //double scale = std::fabs(_L->operator()(p)/100.0);
  _nln.set_scale(_scale);
  //Run solver
  int nitr = 0;
  try{
    nitr = solver.minimize(_nln, x, max, _lower, _upper);
  }catch(const std::exception& e){
    std::cout << "Stopped early so using last value\n";
  }
  
  //Convert back
  for ( size_t i = 0; i < _dimension; ++i ){
    parameters[i] = _transforms[i]->inverse(x(i));
    p[i] = parameters[i];
  }
  max = _L->operator()(p);
  return nitr;
}

void optimizer_laplace::find_precision(const Eigen::VectorXd& params, Eigen::MatrixXd& precision, std::string outname)
{
  precision.resize(_dimension, _dimension);
  std::vector<double> x0(_dimension, 0.0), y0(_dimension, 0.0);
  
  for ( size_t i = 0; i < _dimension; ++i ){
    x0[i] = _transforms[i]->forward(params(i));
    y0[i] = _transforms[i]->forward(params(i));
  }
  
  _nln.set_scale(1.0);
  double l = _nln.loglikelihood_trans(x0);
  //Alright lets find the derivatives using finite difference
  for (size_t i = 0; i < _dimension; ++i){
    for (size_t j = 0; j <= i; j++){
      double hx = step_size(std::fabs(x0[i])); 
      double hy = step_size(std::fabs(x0[j]));
      //std::cout << "hx: " << hx << std::endl;
      if ( i == j ){
        //finite difference
        x0[i] += hx;
        double lx = _nln.loglikelihood_trans(x0);
        y0[i] -= hy;
        double ly = _nln.loglikelihood_trans(y0);
        precision(i,j) = -(lx - 2*l + ly)/(hx*hy);
        //Move back to the map
        x0[i] -= hx;
        y0[i] += hy;
         
      }else{
        //do f(x+h,y+h)
        x0[i] += hx;
        x0[j] += hy;
        double lpp = _nln.loglikelihood_trans(x0);
        x0[i] -= hx;
        x0[j] -= hy;
        //do f(x-h,y-h)
        x0[i] -= hx;
        x0[j] -= hy;
        double lmm = _nln.loglikelihood_trans(x0);
        x0[i] += hx;
        x0[j] += hy;
        //do f(x+h,y-k)
        x0[i] += hx;
        x0[j] -= hy;
        double lpm = _nln.loglikelihood_trans(x0);
        x0[i] -= hx;
        x0[j] += hy;
        //do f(x-h,y+k)
        x0[i] -= hx;
        x0[j] += hy;
        double lmp = _nln.loglikelihood_trans(x0);
        x0[i] += hx;
        x0[j] -= hy;
        precision(i,j) = -(lpp - lpm - lmp + lmm)/(4*hx*hy);
        precision(j,i) = precision(i,j);
      }
    }
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if ( outname != "" ){
    if (rank == 0){
      std::ofstream out(outname.c_str());
     out.precision(12);
      out << "# MAP\n";
      for ( size_t i = 0; i < _dimension; ++i )
        out << std::setw(25) << _transforms[i]->forward(params(i));
      out << std::endl;
      out << "# Precision\n";
      out << precision << std::endl;
    }
  }
}


} //end Themis

/*!
  \file likelihood.cpp
  \author 
  \date  April, 2017
  \brief Implementation file for the Likelihood class
  \details
*/


#include "likelihood.h"
#include "random_number_generator.h"

#include <iostream>
#include <fstream>
#include <iomanip>

namespace Themis
{
  likelihood::likelihood(std::vector<prior_base*> P, std::vector<transform_base*> T, std::vector<likelihood_base*> L, std::vector<double>& W)
    : _P(P), _T(T), _L(L), _W(W), _use_finite_difference_gradients(false), _comm(MPI_COMM_WORLD)
  {
    if (T.size() > 0)
      _is_transform_provided = true;
    else
      _is_transform_provided = false;
  }

  likelihood::likelihood(std::vector<prior_base*> P, std::vector<likelihood_base*> L, std::vector<double>& W)
    : _P(P), _L(L), _W(W), _is_transform_provided(false), _use_finite_difference_gradients(false), _comm(MPI_COMM_WORLD)
  {
  }
  
  void likelihood::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    for (size_t i=0; i<_L.size(); ++i)
      _L[i]->set_mpi_communicator(comm);
  }

  
  double likelihood::operator() (std::vector<double>& x)
  {
    double logL = 0.0;
    prior Pr(_P);

    _X.resize(x.size());
    
    //Apply the inverse parameter transformation
    if(_is_transform_provided)
    {
      for(size_t i = 0 ; i < _T.size(); ++i)
      {
	_X[i] = _T[i] -> inverse(x[i]);
      }
    }
    else
      _X = x;

    //Add up the logPrior
    logL += Pr(_X); 


    // Only if it is not -infinity (or +infinity), add up the weighted likelihood functions
    if (std::isfinite(logL))
    {
      for(size_t i = 0 ; i < _W.size(); ++i)
      {
        logL += _W[i] * _L[i]->operator()(_X);
      }
    }
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //std::cout << "rank " << rank << "  logL = " << logL << std::endl;


    //return -logL;
    if( std::isnan(logL))
    {
      std::cerr << "Likelihood.cpp: Error. Likelihood is taking NaN value." << std::endl;
      for (size_t i=0; i<_X.size(); ++i)
	std::cerr << std::setw(15) << _X[i];
      std::cerr << std::endl;
      return -std::numeric_limits< double >::infinity();
    }
    else
      return logL;
  }
  
  std::vector<double> likelihood::gradient(std::vector<double>& x)
  {
    _X.resize(x.size());

    // Transforms are all 1D, so the Jacobian is necessarily diagonal
    std::vector<double> jacobian(x.size(),1.0);
    if(_is_transform_provided)
    {
      for (size_t i=0; i<_T.size(); ++i)
      {
	_X[i] = _T[i] -> inverse(x[i]);
	jacobian[i] = _T[i] -> inverse_jacobian(x[i]);
      }
    }
    else
      _X = x;
    
    // Prior contributions
    prior Pr(_P);
    std::vector<double> grad = Pr.gradient(_X);

    // Short-circuit on infs
    for (size_t i=0; i<x.size(); ++i)
      if ( ! std::isfinite(grad[i]) )
	return grad;

    if (_use_finite_difference_gradients)
    {
      // Use finite differences across likelihods
      std::vector<double> grad_sub(x.size(),0.0);
      std::vector<double> y=x;
      double h;
      for (size_t i=0; i<x.size(); ++i)
      {
	// Obtain adaptive stepsize
	h = step_size(std::fabs(Pr.upper_bound(i)-Pr.lower_bound(i)));
	
	// Forward step
	y[i] = x[i]+h;
	if (std::isfinite(Pr(y)))
	  grad_sub[i] = this->operator()(y);
	else
	  grad_sub[i] = -std::numeric_limits<double>::infinity();
        
        //std::cerr << "Forward: "  << i << std::setw(15) << y[i] << grad_sub[i] << std::endl;
      
	// Backward step
	y[i] = x[i]-h;
	if (std::isfinite(Pr(y)))
	  grad_sub[i] -= this->operator()(y);
	else
	  grad_sub[i] = std::numeric_limits<double>::infinity();
        //std::cerr << "Back: "  << i << std::setw(15) << y[i] << grad_sub[i] << std::endl;
      
	// Return and complete
	y[i] = x[i];
	grad_sub[i] /= (2.0*h);
	grad[i] += grad_sub[i];
        //std::cerr << "Grad: "  << i << std::setw(15) << y[i] << grad_sub[i] << std::endl;
      }
    }
    else
    {
      // Likelihood contributions: loop over likelihoods and sum the gradients
      std::vector<double> grad_sub;
      for(size_t i=0 ; i<_W.size(); ++i)
      {
	grad_sub = _L[i]->gradient(_X,Pr);
	for (size_t j=0; j<x.size(); ++j)
	  grad[j] += _W[i] * grad_sub[j];
      }
    }
      
    // Multiply by the transform Jacobian
    for (size_t j=0; j<x.size(); ++j)
      grad[j] *= jacobian[j];
    

    // Return
    return grad;
  }
  
  double likelihood::chi_squared(std::vector<double>&x)
  {
    double chi2 = 0.0;
    
    prior Pr(_P);

    _X.resize(x.size());
    
    //Apply the inverse parameter transformation
    if(_is_transform_provided)
    {
      for(size_t i = 0 ; i < _T.size(); ++i)
      {
	_X[i] = _T[i] -> inverse(x[i]);
      }
    }
    else
      _X = x;

    
    //Add up the chi-squared values
    if (std::isfinite(Pr(_X)))
      for(size_t i = 0 ; i < _W.size(); ++i)
      {
	chi2 +=  _L[i]->chi_squared(_X);
      }      
    else
      chi2=std::numeric_limits<double>::infinity();
    
    return chi2;
  }


  void likelihood::write_prior_chain(std::string chainfile, size_t nsamples, int seed)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    // Open chain file and write top details
    std::ofstream outchain;
    double lx, ly;
    std::vector<double> x(_P.size(),0.0), y(_P.size(),0.0);

    if (rank==0)
    {
      outchain.open(chainfile);
      outchain << "#chainfmt afss\n#";
      for (size_t j=0; j<x.size(); ++j)
	outchain << std::setw(15) << "p"+std::to_string(j);
      outchain << std::endl;
    }

    Themis::Ran2RNG rng(seed);
    size_t i=0;
    for (size_t k=0; i<nsamples && k<100*nsamples; k++ )
    {
      for (size_t j=0; j<x.size(); ++j)
      {
	y[j] = (_P[j]->upper_bound()-_P[j]->lower_bound())*rng.rand()+_P[j]->lower_bound();
	if ( std::exp(_P[j]->operator()(y[j])-_P[j]->operator()(x[j])) >= rng.rand() )
	  x[j] = y[j];
      }
      
      
      if (rank==0)
      {
	if (prior_check(x)>-std::numeric_limits< double >::infinity())
	{
	  for (size_t j=0; j<x.size(); ++j)
	    outchain << std::setw(15) << x[j];
	  outchain << std::endl;
	  i++;
	}
      }
    }
    if (rank==0)
    {
      outchain.close();
      if (i<nsamples)
	std::cerr << "WARNING: Quit sampling prior before " << nsamples << " reached.\n";
    }
  }
  
  double likelihood::prior_check(std::vector<double>& x)
  {
    double logL = 0.0;
    prior Pr(_P);

    _X.resize(x.size());
    
    //Apply the inverse parameter transformation
    if(_is_transform_provided)
    {
      for(size_t i = 0 ; i < _T.size(); ++i)
      {
	_X[i] = _T[i] -> inverse(x[i]);
      }
    }
    else
      _X = x;

    //Add up the logPrior
    logL += Pr(_X); 


    // Only if it is not -infinity (or +infinity), add up the weighted likelihood functions
    if (std::isfinite(logL))
    {
      for(size_t i = 0 ; i < _W.size(); ++i)
      {
        logL += 0.0; // _W[i] * _L[i]->operator()(_X);
      }
    }
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //std::cout << "rank " << rank << "  logL = " << logL << std::endl;


    //return -logL;
    if( std::isnan(logL))
    {
      std::cerr << "Likelihood.cpp: Error. Likelihood is taking NaN value." << std::endl;
      for (size_t i=0; i<_X.size(); ++i)
	std::cerr << std::setw(15) << _X[i];
      std::cerr << std::endl;
      return -std::numeric_limits< double >::infinity();
    }
    else
      return logL;
  }
  

  void likelihood::forward_transform(std::vector<double>&x)
  {
    if(x.size() != _T.size())
    {
      std::cout << "Error: likelihood::likelihood.h:" << std::endl;
      std::cout << "Parameter vector size not equal to the number of transforms!" << std::endl;
    }
    
    for(size_t i = 0 ; i < _T.size(); ++i)
    {
      x[i] = _T[i] -> forward(x[i]);
    }
  }

  bool likelihood::transform_state()
  {
    return _is_transform_provided;
  }

  void likelihood::use_finite_difference_gradients()
  {
    _use_finite_difference_gradients = true;
  }

  void likelihood::use_intrinsic_likelihood_gradients()
  {
    _use_finite_difference_gradients = false;
  }
  
  std::vector<prior_base*> likelihood::priors()
  {
    return _P;
  }
  std::vector<transform_base*> likelihood::transforms()
  {
    return _T;
  }
  std::vector<likelihood_base*> likelihood::likelihoods()
  {
    return _L;
  }
  std::vector<double>& likelihood::weights()
  {
    return _W;
  }


  void likelihood::output_1d_slice(std::string fname, std::vector<double> p1, std::vector<double> p2, double xmin, double xmax, size_t Nx)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    std::vector<double> p(p1.size());
    double x, lval;


    std::ofstream out;
    if (rank==0)
    {
      out.open(fname);
      out << '#' << std::setw(14) << "x"
	  << std::setw(15) << "log(L)"
	  << std::setw(15) << "p[k]"
	  << std::endl
	  << "#\n";
    }

    for (size_t i=0; i<Nx; ++i)
    {
      x = xmin + (xmax-xmin)*double(i)/(double(Nx)-1.0);
      for (size_t k=0; k<p.size(); ++k)
	p[k] = p1[k] + (p2[k]-p1[k])*x;

      lval = operator()(p);
      
      if (rank==0)
      {
	out << std::setw(15) << x
	    << std::setw(15) << lval;
	for (size_t k=0; k<p.size(); ++k)
	  out << std::setw(15) << p[k];
	out << '\n';
      }
    }
    
    if (rank==0)
      out.close();
  }

  void likelihood::output_2d_slice(std::string fname, std::vector<double> p1, std::vector<double> p2, std::vector<double> p3, double xmin, double xmax, size_t Nx, double ymin, double ymax, size_t Ny)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    std::vector<double> p(p1.size()), dp2(p1.size()), dp3(p1.size());
    double x, y, lval;

    double dp2ddp3=0;
    double dp2norm=0;
    for (size_t k=0; k<p.size(); ++k)
    {
      dp2[k] = p2[k]-p1[k];
      dp3[k] = p3[k]-p1[k];

      dp2ddp3 += dp2[k]*dp3[k];
      dp2norm += dp2[k]*dp2[k];
    }
    dp2norm = std::sqrt(dp2norm);
    for (size_t k=0; k<p.size(); ++k)
    {
      dp3[k] = dp3[k];// - dp2ddp3*dp2[k]/dp2norm;
    }

    std::ofstream out;
    if (rank==0)
    {
      out.open(fname);
      out << '#' 
	  << std::setw(14) << "x"
	  << std::setw(15) << "y"
	  << std::setw(15) << "log(L)"
	  << std::setw(15) << "p[k]"
	  << std::endl
	  << "#\n";
    }

    for (size_t i=0; i<Nx; ++i)
    {
      x = xmin + (xmax-xmin)*double(i)/(double(Nx)-1.0);

      for (size_t j=0; j<Ny; ++j)
      {
	y = ymin + (ymax-ymin)*double(j)/(double(Ny)-1.0);

	for (size_t k=0; k<p.size(); ++k)
	  p[k] = p1[k] + dp2[k]*x + dp3[k]*y;

	lval = operator()(p);
      
	if (rank==0)
        {
	  out << std::setw(15) << x
	      << std::setw(15) << y
	      << std::setw(15) << lval;
	  for (size_t k=0; k<p.size(); ++k)
	    out << std::setw(15) << p[k];
	  out << '\n';
	}
      }
    }
    
    if (rank==0)
      out.close();
  }


};

/*! 
  \file validation/rosenbrock_mcmc_sampling.cpp
  \author Roman Gold & Paul Tiede & Mansour Karami
  \date Feb 2020
  
  \internal
  \validation Testing the DEO differential evolution tempering scheme
  \endinternal
  
  \brief Samples a 2D Rosenbrock function 

  \details Testing the ability of the sampler to sample likelihood distributions with long tails and unisotropic gradients.
  The natural logarithm of the likelihood is  given by:

  \f$\log{(L(\mathbf{x}))} = -100*(y-x^2)^2)+(1-x)^2)\f$

  Using the output chain file the marginalized distributions are calculated and plotted:

  \image html sampler-rosenbrock-triangle.png "Marginalized posterior probabilty distribution"

*/


#include "likelihood.h"
#include "likelihood_2dnormal.h"
#include "optimizer_laplace.h"
#include <mpi.h>
#include <memory> 
#include <string>
#include <Eigen/Core>
#include <LBFGSB.h>

int main(int argc, char* argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  
  // Dynamically allocate a prior for each variable and add it to the container
  // Here we are using a flat prior for all the parameters and confine them 
  // to the interval [-2.5*pi, 2.5*pi]
  int dim = 2;
  std::vector<double> m(dim,0.0);
  m[0] = 1.0;
  double corr = 0.1;
  std::vector<double> cov(dim, 0.0);
  cov[0] = 4.0;
  cov[1] = 9.0;
  

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<Themis::transform_base*> T;
  for ( int i = 0; i < dim;  ++i ){
    P.push_back(new Themis::prior_linear(-100, 100));
    T.push_back(new Themis::transform_none());
  }
  // P.push_back(new Themis::prior_linear(-8, 8));
  // P.push_back(new Themis::prior_linear(-8, 8));
  // P.push_back(new Themis::prior_linear(-8, 8));
  



  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  // L.push_back(new Themis::likelihood_eggbox(5.0));
  L.push_back(new Themis::likelihood_2dnormal(m,cov,corr));


  // Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  // vector to hold the name of variables, if the names are provided it would  
  // be added as the header to the chain file
  std::vector<std::string> var_names;



  //Make a laplace optimizer
  std::cout << "Creating optimizer\n";
  Themis::optimizer_laplace optim(L_obj, var_names, dim);
  
  
  Eigen::VectorXd start = Eigen::VectorXd::Constant(dim, 5.0);
  std::cout << "Starting optmizer\n";
  double lmap = L_obj(m);
  int nitr = optim.run_optimizer(start, lmap);
  std::cout << "lmap = " << lmap << std::endl;
  std::cout << "LBFGSB ran for " << nitr << " iterations\n";
  std::cout << "Maximum at \n" << start << std::endl;
  std::cout << "map: " << lmap << std::endl;

  //Or if you want to run a bunch of optimizations starting at random points call
  std::vector<double> parameters(dim, 0.0);
  int number_instances = 10; //number of times to run the optimizer from a random location
  int seed = 23;
  optim.parallel_optimizer(parameters, lmap, number_instances, seed, "");


  std::cout << "Finding precision matrix\n";
  Eigen::MatrixXd precision;
  optim.find_precision(start, precision, "laplace.txt");
  
  

  // Finalize MPI
  MPI_Finalize();
  return 0;
  
}




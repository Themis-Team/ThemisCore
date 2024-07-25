/*!  
  \file sampler_grid_search.h
  \author Mansour Karami
  \date July 2017
  
  \brief Header file for Grid Search class.
  \details The grid search sampler evaluates the chi squared on a grid in the model parameter space. 
  The positions on the grid in each parameter are defind by a minimum value, a maximum value and the number of 
  smaples in that particular direction. The sampler outputs a file that contains the values of the parameters 
  on the gird together with their corresponding chi squared values. 
  
*/


#ifndef THEMIS_SAMPLER_GRID_SEARCH_H
#define THEMIS_SAMPLER_GRID_SEARCH_H

#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "../likelihood/likelihood.h"


namespace Themis{

  /*! 
    \class sampler_grid_search
    \brief Implements a Grid Search Sampler.
      
    \details Runs a grid search sampler.
  */
  class sampler_grid_search
  {
    public:
      
      /*!
        \brief Grid Search Sampler class constructor. 
        */
      sampler_grid_search();
      ~sampler_grid_search();

      /*!
        /brief Runs the grid search sampler.
        
        \param _L An object of class likelihood.
        \param range_min Vector of doubles holding the minimun value for each parameter.
        \param range_max Vector of doubles holding the maximun value for each parameter.
        \param num_samples Vector of integers holding the number of divisions for each parameter.
        \param output_file A string holding the name of the output file. The output has a column for each parameter and the last column is the chi squared value associated to the parameters in that row.
        \param output_precision Sets the output precision -- the number of significant digits used to represent a number in the sampler output files. The defaul precision is 6.
        */
      void run_sampler(likelihood _L, std::vector<double>& range_min, std::vector<double>& range_max, std::vector<int>& num_samples, std::string output_file, int output_precision = 6);

      /*!
        \brief Function to set the distribution of processors in different layers of parallelization.
        
        \param num_batches Integer value. Number of regions of parameter space handled by different cpu groups.
        \param num_likelihood Integer value. Number of cpus allocated to each region of parameter space. These
        processes evaluate every single likelihood in parallel.  
      */
      void set_cpu_distribution(int num_batches, int num_likelihood);
  
    private:

      int BNum;
      int LNum;
      bool default_cpu_distribution = true;;
  
  };
};

#endif

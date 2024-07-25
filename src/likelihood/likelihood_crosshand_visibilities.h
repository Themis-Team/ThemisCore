/*! 
  \file likelihood_crosshand_visibilities.h
  \author Avery E Broderick
  \date  March, 2020
  \brief Header file for the crosshand visibilities likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_CROSSHAND_VISIBILITIES_H_
#define THEMIS_LIKELIHOOD_CROSSHAND_VISIBILITIES_H_
#include <vector>
#include <complex>
#include "likelihood_base.h"
#include "data_crosshand_visibilities.h"
#include "model_crosshand_visibilities.h"
#include "uncertainty_crosshand_visibilities.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a visibility amplitude likelihood object
    
    \details This class takes a visibility data object and
    a visibility model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    visibilities have Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality

    \todo
  */
  class likelihood_crosshand_visibilities : public likelihood_base
  {
    public:
      likelihood_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model);
      likelihood_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty);

      ~likelihood_crosshand_visibilities() {};
      
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_crosshand_visibilities& _data;
      model_crosshand_visibilities& _model;
      uncertainty_crosshand_visibilities _local_uncertainty; // Default if none is passed
      uncertainty_crosshand_visibilities& _uncertainty;
  };
};

#endif 

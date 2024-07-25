/*
    \file model_image_sed_fitted_riaf.cpp
    \author Hung-Yi Pu
    \date  Nov, 2018
    \brief test model_image_score clas by generating 10 images of different parameters
    \details  !!!!!!!!!
              [1]before compile, changle #define save_image (1) in model_image_score.cpp
	      [2]after compling, generate all 10 images by using $mpirun -np 10 exec 
	      !!!!!!!!!
*/
	
#include<mpi.h>
#include<ctime>
#include <vector>
#include "model_image_score.h"
#include <err.h>
#define Ntotal 10
#define save_image (1)

//    Parameter list:\n
//      - parameters[0] ...  V_toal
//	- parameters[1] ... (M/R)/ (M0/R0)
//      - parameters[2] ...  Postion Angle

//par1: V_total
double par1_list[Ntotal] = {
1.,
0.6,
0.7,
0.8,
0.9,
1.,
1.,
1.,
1.,
1.
};

//par2: M_ratio
double par2_list[Ntotal] = {
1.,
1.,
1.2,
1.5,
0.5,
1.,
0.3,
2.3,
1.05,
4.
};



int main(int argc, char* argv[])
{
using namespace std;
 
  //====for mpi: every core will go through the follwoing process
  //int numprocs;
  int rank;
  srand (time(NULL));
  int initialized, finalized;

  MPI_Initialized(&initialized);
  if (!initialized)
    MPI_Init(NULL, NULL);


	MPI_Comm_rank(MPI_COMM_WORLD, &rank);      /* get current process id */



  //===useful constants
  double M_sun =1.99e+33;//g
  double D_pc  =3.086e18;//cm  
  
  //==image information	
  //!!!!note that the image file is specified in model_image_score.cpp!!!
  int    Nray  =160;          // resolution of the image
  double frequency = 230.e+9; // GHz
  double fov = 160.;          // total size of the image (uas)
  double M = 6.2e+9*M_sun;    //black hole mass used for post-processing
  double D = 16.9e+6*D_pc;    //distance used for post-processing

  
   //assign vectors
   std::vector<std::vector<double> > I;
   std::vector<std::vector<double> > alpha;
   std::vector<std::vector<double> > beta;
   std::vector<double> parameters;
   double par_V=par1_list[rank];
   double par_M=par2_list[rank];

   parameters.push_back(par_V);
   parameters.push_back(par_M);
	
   //run job
   Themis::model_image_score image(Nray, M, D, fov, frequency);  
   printf("Nray=%d  M=%e D=%e  fov=%f rank=%d V_ratio=%f M_ratio=%f \n", Nray,M,D,fov,rank, par_V, par_M);
   //image.test_call();
   image.generate_image(parameters, I, alpha, beta);
	
       
              
  MPI_Finalized(&finalized);
  if (!finalized)
    MPI_Finalize();
    
 

 return 0;
}

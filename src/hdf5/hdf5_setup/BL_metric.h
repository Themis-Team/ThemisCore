//==============
//  BL metric 
//==============
#include "iharm_def.h"
#ifndef BL_METRIC_H
#define BL_METRIC_H

static double g11(double *y){
       double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;
       double sigma = r*r+a*a*cos(theta)*cos(theta);

       return sigma/delta;      
       }
       
static double g22(double *y){
       double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;

       return delta;      
       }

static double g33(double *y){
        double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;
       double sigma = r*r+a*a*cos(theta)*cos(theta);
//       double AA=(r*r+a*a)*(r*r+a*a)+delta*a*a*sin(theta)*sin(theta);
  //     return AA*sin(theta)*sin(theta)/sigma;            
	double AA=r*r+a*a*(1.+2.*r*sin(theta)*sin(theta)/sigma);
 	return AA*sin(theta)*sin(theta); 
      }
       
       
       
static double g00(double *y){
        double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;
       double sigma = r*r+a*a*cos(theta)*cos(theta);

  //     return -(delta-a*a*sin(theta)*sin(theta))/sigma;            

      return -(1.-2.*r/sigma);
       }

static double g03(double *y){
        double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;
       double sigma = r*r+a*a*cos(theta)*cos(theta);

       return -2.*a*r*sin(theta)*sin(theta)/sigma;            
       }

static double g33_con(double *y){
        double r=y[0];
       double theta=y[1];
       double delta = r*r-2.*r+a*a;
       double sigma = r*r+a*a*cos(theta)*cos(theta);
 
       return (sigma-2.*r)/(delta*sigma*sin(theta)*sin(theta));    

   }
   
#endif
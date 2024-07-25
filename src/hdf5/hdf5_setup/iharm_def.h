
#ifndef IHARM_DEF_H_
#define IHARM_DEF_H_

//==================
//=== set up data numbers before compile! use h5ls to check number of prims!
#define NN1 288
#define NN2 128
#define NN3 128
#define N_readfrom 8     // rho, uu, V1, V2, V3, B1, B2, B3, KTOT, KEL
//===================

#define N_readto   8    // only want rho, uu, V1, V2, V3, B1, B2, B3
#define dRHO     0
#define dUU      1
#define dU1      2
#define dU2      3
#define dU3      4
#define dB1      5
#define dB2      6
#define dB3      7
#define pLOOP    for(int j=0;j<N_readto;j++)
#define fourLOOP    for(int j=0; j<4; j++)
#define DLOOP  for(j=0;j<4;j++) for(k=0;k<4;k++)
#define DLOOPA for(j=0;j<4;j++)
#define SLOOPA for(j=1;j<4;j++)
#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) 
//==trick: finding order when 1D-lized data
#define data_order(a,b,c,d) ((int)a*NN2*NN3*N_readto+(int)b*NN3*N_readto+(int)c*N_readto+(int)d)

#define cen_off 0.5



char harmfname[]="../hdf5_data/dump_00000900.h5"; //Charles's iharm output

#endif
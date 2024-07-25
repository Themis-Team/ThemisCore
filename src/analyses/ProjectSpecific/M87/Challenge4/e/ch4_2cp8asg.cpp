//!!! fitting challenge4 data by crescent model prepared by Alex

#include "model_image_crescent.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_sum.h"
//#include "model_ensemble_averaged_scattered_image.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "utils.h"

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated in rank: " << world_rank << std::endl;


  // Read in visibility amplitude data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge4/ch4_VA_im5.d"));

  // Read in closure phases data
  Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/Challenge4/ch4_CP_im5.d"));
  
  // Choose the model to compare
  //  1 Crescent
  Themis::model_image_crescent crescent, crescent2;
  //  3 Gaussians
  Themis::model_image_asymmetric_gaussian asg1, asg2, asg3, asg4, asg5, asg6, asg7, asg8;
  //  Combined
  Themis::model_image_sum intrinsic_image;
  intrinsic_image.add_model_image(crescent);
  intrinsic_image.add_model_image(asg1);
  intrinsic_image.add_model_image(asg2);
  intrinsic_image.add_model_image(asg3);
  intrinsic_image.add_model_image(asg4);
  intrinsic_image.add_model_image(asg5);
  intrinsic_image.add_model_image(crescent2);
  intrinsic_image.add_model_image(asg6);
  intrinsic_image.add_model_image(asg7);
  intrinsic_image.add_model_image(asg8);
  
  // Use analytical Visibilities
  crescent.use_analytical_visibilities();
  crescent2.use_analytical_visibilities();
  asg1.use_analytical_visibilities();
  asg2.use_analytical_visibilities();
  asg3.use_analytical_visibilities();
  asg4.use_analytical_visibilities();
  asg5.use_analytical_visibilities();
  asg6.use_analytical_visibilities();
  asg7.use_analytical_visibilities();
  asg8.use_analytical_visibilities();


  
  // Container of base prior class pointers with their means and ranges
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;


  double uas2rad = 1e-6/3600. * M_PI/180.;

  // Crescent params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(3.4982414);
  ranges.push_back(0.01);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
  means.push_back(1.0602383e-10);
  ranges.push_back(1e-4*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.10132669);
  ranges.push_back(1e-4);
  //   tau
  P.push_back(new Themis::prior_linear(0.01,0.99));
  means.push_back(0.79737272);
  ranges.push_back(1e-4);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(-2.0610537);
  ranges.push_back(1e-4);
  //   x offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);

  // Gaussian component params
  for (size_t i=0; i<5; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
    //   Size
    P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
    means.push_back(10*uas2rad);
    ranges.push_back(4*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.2);
    ranges.push_back(0.1);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(22*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(22*uas2rad);
  }

  // Crescent2 params
  //   Total Flux V00
  P.push_back(new Themis::prior_linear(0.0,7.0));
  means.push_back(0.5);
  ranges.push_back(0.01);
  //   Outer size R
  P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
  means.push_back(1.0602383e-10);
  ranges.push_back(1e-4*uas2rad);
  //   psi
  P.push_back(new Themis::prior_linear(0.0001,0.9999));
  means.push_back(0.10132669);
  ranges.push_back(1e-4);
  //   tau
  P.push_back(new Themis::prior_linear(0.01,0.99));
  means.push_back(0.79737272);
  ranges.push_back(1e-4);
  //   Position angle
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(-2.0610537);
  ranges.push_back(1e-4);
  //   x offset
  P.push_back(new Themis::prior_linear(-1.0e2*uas2rad,1.0e2*uas2rad));
  means.push_back(0.0);
  ranges.push_back(40*uas2rad);
  //   y offset
  P.push_back(new Themis::prior_linear(-1.0e2*uas2rad,1.0e2*uas2rad));
  means.push_back(0.0);
  ranges.push_back(40*uas2rad);


  // Gaussian component params
  for (size_t i=0; i<3; ++i)
  {
    //   Total Flux V00
    P.push_back(new Themis::prior_linear(0.0,10.0));
    means.push_back(1e-1);
    ranges.push_back(1e-4);
    //   Size
    P.push_back(new Themis::prior_linear(0.0,50*uas2rad));
    means.push_back(10*uas2rad);
    ranges.push_back(4*uas2rad);
    //   Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.2);
    ranges.push_back(0.1);
    //   phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    //   x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(22*uas2rad);
    //   y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
    ranges.push_back(22*uas2rad);
  }


  means[0] = 1.9585735;
  means[1] = 1.2202246e-10;
  means[2] = 0.34763943;
  means[3] = 0.79149591;
  means[4] = -1.4564152;
  means[5] = -3.1809189e-18;
  means[6] = -3.3699943e-18;
  means[7] = 0.82549669;
  means[8] = 9.3940746e-11;
  means[9] = 0.71026369;
  means[10] = 2.0293777;
  means[11] = -1.6266677e-11;
  means[12] = 1.1405847e-10;
  means[13] = 0.2013497;
  means[14] = 2.1859332e-11;
  means[15] = 0.96271647;
  means[16] = 1.6642183;
  means[17] = -6.1969416e-11;
  means[18] = -1.8479018e-10;
  means[19] = 0.12896146;
  means[20] = 3.8598155e-11;
  means[21] = 0.57288564;
  means[22] = 1.689556;
  means[23] = 1.627308e-10;
  means[24] = 1.8684018e-10;
  means[25] = 0.27949084;
  means[26] = 4.647899e-11;
  means[27] = 0.86613154;
  means[28] = 3.1005495;
  means[29] = -1.999447e-10;
  means[30] = -2.0369721e-11;
  means[31] = 0.10571384;
  means[32] = 3.931911e-11;
  means[33] = 0.68285323;
  means[34] = 2.8580728;
  means[35] = 2.0337677e-10;
  means[36] = -1.6669975e-10;

  /*
  // 2nd crescent
  means[37] = 0.5;
  means[38] = 1.0091313e-10;
  means[39] = 0.0060642193;
  means[40] = 0.9899736;
  means[41] = -1.4106404;
  means[42] = 2.759793e-18;
  means[43] = 1.2591185e-18;
  */



  means[0] = 0.68475306;
  means[1] = 1.2906063e-10;
  means[2] = 0.3670335;
  means[3] = 0.22638964;
  means[4] = -1.4270928;
  means[5] = -8.5161322e-19;
  means[6] = -4.7221609e-18;
  means[7] = 0.87464789;
  means[8] = 8.9502816e-11;
  means[9] = 0.71241854;
  means[10] = 1.9760008;
  means[11] = -1.4662161e-11;
  means[12] = 1.1869557e-10;
  means[13] = 0.23852794;
  means[14] = 3.9288021e-11;
  means[15] = 0.91701156;
  means[16] = 1.6953932;
  means[17] = -6.4995516e-11;
  means[18] = -1.8118807e-10;
  means[19] = 0.093848138;
  means[20] = 2.3196439e-11;
  means[21] = 0.82229819;
  means[22] = 1.702032;
  means[23] = 1.7179249e-10;
  means[24] = 1.9280577e-10;
  means[25] = 0.27241725;
  means[26] = 4.3861259e-11;
  means[27] = 0.84046953;
  means[28] = 3.1240173;
  means[29] = -1.9844634e-10;
  means[30] = -2.5313585e-11;
  means[31] = 0.10309835;
  means[32] = 3.3549726e-11;
  means[33] = 0.80930844;
  means[34] = 2.8750421;
  means[35] = 2.0190869e-10;
  means[36] = -1.5243517e-10;
  means[37] = 1.2326906;
  means[38] = 1.0589247e-10;
  means[39] = 0.098895596;
  means[40] = 0.98343926;
  means[41] = -2.9345729;
  means[42] = -1.4896917e-18;
  means[43] = 2.7423401e-18;
  
  // Again
  means[0] = 0.62969325;
  means[1] = 1.5424986e-10;
  means[2] = 0.38342265;
  means[3] = 0.010160929;
  means[4] = -1.4676783;
  means[5] = 3.2525591e-19;
  means[6] = -2.8200829e-18;
  means[7] = 0.80067856;
  means[8] = 9.194783e-11;
  means[9] = 0.7732671;
  means[10] = 1.9878874;
  means[11] = -1.271801e-11;
  means[12] = 1.2769599e-10;
  means[13] = 0.21002739;
  means[14] = 4.4114806e-11;
  means[15] = 0.82051622;
  means[16] = 1.6770519;
  means[17] = -9.5969824e-11;
  means[18] = -1.6998486e-10;
  means[19] = 0.071728488;
  means[20] = 1.544919e-11;
  means[21] = 0.91788226;
  means[22] = 1.8776508;
  means[23] = 1.7568197e-10;
  means[24] = 1.9591347e-10;
  means[25] = 0.23810063;
  means[26] = 2.7106551e-11;
  means[27] = 0.95831051;
  means[28] = 3.0851224;
  means[29] = -2.0408479e-10;
  means[30] = -3.4872811e-11;
  means[31] = 0.081826083;
  means[32] = 2.1287665e-11;
  means[33] = 0.91967612;
  means[34] = 2.9882627;
  means[35] = 2.0513809e-10;
  means[36] = -1.6447504e-10;
  means[37] = 1.4679264;
  means[38] = 9.7940302e-11;
  means[39] = 0.018073663;
  means[40] = 0.98757621;
  means[41] = -2.8740315;
  means[42] = -6.2365315e-19;
  means[43] = 1.1144529e-18;



  means[0] = 0.61308398;
  means[1] = 1.5370384e-10;
  means[2] = 0.38605357;
  means[3] = 0.015423705;
  means[4] = -1.4721042;
  means[5] = -1.2007365e-18;
  means[6] = -1.099555e-19;
  means[7] = 0.79444575;
  means[8] = 9.2188377e-11;
  means[9] = 0.76191888;
  means[10] = 1.987051;
  means[11] = -1.2887372e-11;
  means[12] = 1.2695808e-10;
  means[13] = 0.2087506;
  means[14] = 4.4367845e-11;
  means[15] = 0.81515966;
  means[16] = 1.6703685;
  means[17] = -9.5392077e-11;
  means[18] = -1.6980164e-10;
  means[19] = 0.071704295;
  means[20] = 1.5678551e-11;
  means[21] = 0.91295147;
  means[22] = 1.8801917;
  means[23] = 1.7586925e-10;
  means[24] = 1.9533537e-10;
  means[25] = 0.22976297;
  means[26] = 2.7530384e-11;
  means[27] = 0.95646224;
  means[28] = 3.088906;
  means[29] = -2.0411037e-10;
  means[30] = -3.4092254e-11;
  means[31] = 0.077994391;
  means[32] = 1.9942134e-11;
  means[33] = 0.92838419;
  means[34] = 2.9700267;
  means[35] = 2.0540022e-10;
  means[36] = -1.6357386e-10;
  means[37] = 1.4707186;
  means[38] = 9.8048666e-11;
  means[39] = 0.016088046;
  means[40] = 0.98697197;
  means[41] = -2.8449529;
  means[42] = 8.2141109e-19;
  means[43] = -1.9498083e-18;
  means[44] = 0.016233624;
  means[45] = 6.4232161e-11;
  means[46] = 0.23580729;
  means[47] = 2.8977128;
  means[48] = 1.7271768e-10;
  means[49] = -2.2037069e-10;
  means[50] = 0.0008569187;
  means[51] = 1.704724e-10;
  means[52] = 0.83439994;
  means[53] = 1.0254021;
  means[54] = 1.2058368e-10;
  means[55] = 1.0644899e-10;
  means[56] = 0.016860567;
  means[57] = 1.7112244e-11;
  means[58] = 0.98267475;
  means[59] = 0.18527604;
  means[60] = 2.2220029e-10;
  means[61] = 7.8581913e-11;
  

  means[0] = 0.5833088;
  means[1] = 1.5343369e-10;
  means[2] = 0.41733112;
  means[3] = 0.015048365;
  means[4] = -1.4802918;
  means[5] = 7.3741458e-19;
  means[6] = 7.1255489e-19;
  means[7] = 0.72658781;
  means[8] = 8.8589076e-11;
  means[9] = 0.72673423;
  means[10] = 2.0031057;
  means[11] = -6.3268642e-12;
  means[12] = 1.1683427e-10;
  means[13] = 0.24658658;
  means[14] = 6.3627492e-11;
  means[15] = 0.68064747;
  means[16] = 1.8324141;
  means[17] = -9.7139507e-11;
  means[18] = -1.7946116e-10;
  means[19] = 0.070944124;
  means[20] = 2.7131182e-11;
  means[21] = 0.67432411;
  means[22] = 1.7731716;
  means[23] = 1.6778339e-10;
  means[24] = 1.9050179e-10;
  means[25] = 0.18988536;
  means[26] = 1.838571e-11;
  means[27] = 0.97632345;
  means[28] = 3.0795348;
  means[29] = -2.0311646e-10;
  means[30] = -9.2396097e-12;
  means[31] = 0.10056112;
  means[32] = 2.2152627e-11;
  means[33] = 0.90855094;
  means[34] = 2.9736689;
  means[35] = 2.043141e-10;
  means[36] = -1.7153839e-10;
  means[37] = 1.4945496;
  means[38] = 1.0350338e-10;
  means[39] = 0.097435131;
  means[40] = 0.98968964;
  means[41] = -2.777682;
  means[42] = 5.3546707e-19;
  means[43] = -1.041073e-18;
  means[44] = 0.031853761;
  means[45] = 3.7046545e-11;
  means[46] = 0.033692837;
  means[47] = 3.1165344;
  means[48] = 1.0506722e-10;
  means[49] = -2.043388e-10;
  means[50] = 0.001464727;
  means[51] = 5.2886033e-11;
  means[52] = 0.22594255;
  means[53] = 2.2717316;
  means[54] = -1.8631311e-11;
  means[55] = 3.3756082e-11;
  means[56] = 0.054189433;
  means[57] = 1.6769336e-11;
  means[58] = 0.96420218;
  means[59] = 0.025755619;
  means[60] = 2.1652983e-10;
  means[61] = 1.211438e-10;



  means[0] = 0.5977354;
  means[1] = 1.4970939e-10;
  means[2] = 0.38631998;
  means[3] = 0.013237549;
  means[4] = -1.5944158;
  means[5] = 3.3752907e-18;
  means[6] = -4.0956271e-18;
  means[7] = 0.7544415;
  means[8] = 8.6198903e-11;
  means[9] = 0.69600218;
  means[10] = 2.0218948;
  means[11] = -7.7929028e-12;
  means[12] = 1.1037876e-10;
  means[13] = 0.23928542;
  means[14] = 6.3394294e-11;
  means[15] = 0.55814357;
  means[16] = 1.8057609;
  means[17] = -1.0458558e-10;
  means[18] = -1.7745913e-10;
  means[19] = 0.080600498;
  means[20] = 3.1954032e-11;
  means[21] = 0.51693086;
  means[22] = 1.8082348;
  means[23] = 1.6108065e-10;
  means[24] = 1.8826772e-10;
  means[25] = 0.19790132;
  means[26] = 1.7449472e-11;
  means[27] = 0.98056399;
  means[28] = 3.0774872;
  means[29] = -2.0446082e-10;
  means[30] = -5.5496854e-12;
  means[31] = 0.091755467;
  means[32] = 2.5718256e-11;
  means[33] = 0.839248;
  means[34] = 2.9757374;
  means[35] = 2.0391653e-10;
  means[36] = -1.8209029e-10;
  means[37] = 1.4390858;
  means[38] = 1.0454703e-10;
  means[39] = 0.12391802;
  means[40] = 0.96959556;
  means[41] = -0.1989517;
  means[42] = -2.7528341e-18;
  means[43] = -9.4023084e-19;
  means[44] = 0.03646862;
  means[45] = 3.6603562e-11;
  means[46] = 0.39400708;
  means[47] = 2.1198587;
  means[48] = 1.0115757e-10;
  means[49] = -2.2886484e-10;
  means[50] = 0.0066692697;
  means[51] = 1.5802315e-11;
  means[52] = 0.091838101;
  means[53] = 0.53046674;
  means[54] = -5.0858752e-11;
  means[55] = 6.5847237e-11;
  means[56] = 0.055898688;
  means[57] = 1.7699061e-11;
  means[58] = 0.95025707;
  means[59] = 0.056635149;
  means[60] = 2.1812955e-10;
  means[61] = 1.2769985e-10;

  
  
  for (size_t i=0; i<5; ++i)
  {
    ranges[7+i*6+0] = 1e-5;
    ranges[7+i*6+1] = 1e-4*uas2rad;
    ranges[7+i*6+2] = 1e-5;
    ranges[7+i*6+3] = 1e-5;
    ranges[7+i*6+4] = 1e-4*uas2rad;
    ranges[7+i*6+5] = 1e-4*uas2rad;
  }

  
  for (size_t i=0; i<3; ++i)
  {
    ranges[44+i*6+0] = 1e-5;
    ranges[44+i*6+1] = 1e-4*uas2rad;
    ranges[44+i*6+2] = 1e-5;
    ranges[44+i*6+3] = 1e-5;
    ranges[44+i*6+4] = 1e-4*uas2rad;
    ranges[44+i*6+5] = 1e-4*uas2rad;
  }
  
  
  // vector to hold the name of variables
  //std::vector<std::string> var_names = {"$V_{0}$", "$R$", "$\\psi$", "$\\tau$", "$\\xi$"};
  std::vector<std::string> var_names;// = {"$V_{0}$", "$R$", "$\\psi$", "$\\tau$", "$\\xi$"};
  

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_visibility_amplitude(VM,intrinsic_image));
  //L.push_back(new Themis::likelihood_visibility_amplitude(VM,image));

  //Closure Phases
  L.push_back(new Themis::likelihood_closure_phase(CP,intrinsic_image));
  
  
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);
  
  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Output residual data
  L_obj(means);
  L[0]->output_model_data_comparison("VA_residuals.d");
  L[1]->output_model_data_comparison("CP_residuals.d");
  
  // Create a sampler object
  //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  
  // Generate a chain
  int Number_of_chains = 200;
  int Number_of_temperatures = 8;
  int Number_of_procs_per_lklhd = 1;
  int Number_of_steps = 100000; 
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  //bool restart_flag = true;
  int out_precision = 8;
  int verbosity = 0;


  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"Crescent.ckpt");
  
  // Run the Sampler                            
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain-Crescent.dat", "Lklhd-Crescent.dat", "Chi2-Crescent.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}

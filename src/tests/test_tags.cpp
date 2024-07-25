#include "model_image.h"
#include "model_image_adaptive_splined_raster.h"
#include "model_image_astroray.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_crescent.h"
#include "model_image_gaussian.h"
#include "model_image_general_riaf.h"
#include "model_image_general_riaf_shearing_spot.h"
#include "model_image_grtrans.h"
#include "model_image_ipole.h"
#include "model_image_multi_asymmetric_gaussian.h"
#include "model_image_multigaussian.h"
#include "model_image_orbiting_spot.h"
#include "model_image_polynomial_variable.h"
#include "model_image_raptor.h"
#include "model_image_raster.h"
#include "model_image_refractive_scattering.h"
#include "model_image_riaf.h"
#include "model_image_ring.h"
#include "model_image_score.h"
#include "model_image_sed_fitted_riaf.h"
#include "model_image_sed_fitted_riaf_intensity.h"
#include "model_image_sed_fitted_riaf_johannsen.h"
#include "model_image_shearing_spot.h"
#include "model_image_shearing_spot_johannsen.h"
#include "model_image_slashed_ring.h"
#include "model_image_smooth.h"
#include "model_image_splined_raster.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_upsample.h"
#include "model_image_vrt2_pmap.h"
#include "model_image_xsring.h"
#include "model_image_xsringauss.h"

#include "mpi.h"

using namespace Themis;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  
  {
    model_image_adaptive_splined_raster m(3,7,-0.42);
    m.write_model_tag_file("mt02.tag");
  }

  {
    model_image_astroray m;
    m.write_model_tag_file("mt03.tag");
  }

  {
    model_image_asymmetric_gaussian m;
    m.write_model_tag_file("mt04.tag");
  }

  {
    model_image_crescent m;
    m.write_model_tag_file("mt05.tag");
  }

  {
    model_image_gaussian m;
    m.write_model_tag_file("mt06.tag");
  }

  {
    model_image_general_riaf m;
    m.write_model_tag_file("mt07.tag");
  }

  {
    model_image_general_riaf_shearing_spot m(0,10);
    m.write_model_tag_file("mt08.tag");
  }

  {
    model_image_grtrans m;
    m.write_model_tag_file("mt09.tag");
  }

  {
    model_image_ipole m;
    m.write_model_tag_file("mt10.tag");
  }

  {
    model_image_multi_asymmetric_gaussian m(4);
    m.write_model_tag_file("mt11.tag");
  }

  {
    model_image_multigaussian m(7);
    m.write_model_tag_file("mt12.tag");
  }

  {
    model_image_orbiting_spot m(-2,42.1234);
    m.write_model_tag_file("mt13.tag");
  }

  {
    model_image_symmetric_gaussian g;
    model_image_polynomial_variable m(g,1);
    m.write_model_tag_file("mt14.tag");
  }

  {
    model_image_raptor m;
    m.write_model_tag_file("mt15.tag");
  }

  {
    model_image_raster m(-1e-10,1e-10,5,-0.5e-10,1.5e-10,3);
    m.write_model_tag_file("mt16.tag");
  }

  {
    model_image_crescent c;
    model_image_refractive_scattering m(c,13,0);
    m.write_model_tag_file("mt17.tag");
  }

  {
    model_image_riaf m;
    m.write_model_tag_file("mt18.tag");
  }

  {
    model_image_ring m;
    m.write_model_tag_file("mt19.tag");
  }

  {
    model_image_score m("testsim.dat","testREADME.txt");
    m.write_model_tag_file("mt20.tag");
  }

  {
    model_image_sed_fitted_riaf m;
    m.write_model_tag_file("mt21.tag");
  }

  {
    model_image_sed_fitted_riaf_intensity m;
    m.write_model_tag_file("mt22.tag");
  }

  {
    model_image_sed_fitted_riaf_johannsen m;
    m.write_model_tag_file("mt23.tag");
  }

  {
    model_image_shearing_spot m(0,10);
    m.write_model_tag_file("mt24.tag");
  }

  {
    model_image_shearing_spot_johannsen m(0,10);
    m.write_model_tag_file("mt25.tag");
  }

  {
    model_image_slashed_ring m;
    m.write_model_tag_file("mt26.tag");
  }

  {
    model_image_crescent c;
    model_image_smooth m(c);
    m.write_model_tag_file("mt27.tag");
  }

  {
    model_image_splined_raster m(-1.5e-10,0.5e-10,3,-0.5e-10,0.5e-10,4);
    m.write_model_tag_file("mt28.tag");
  }

  {
    model_image_gaussian g1;
    model_image_gaussian g2;
    model_image_sum m;
    m.add_model_image(g1);
    m.add_model_image(g2);
    m.write_model_tag_file("mt29.tag");
  }

  {
    model_image_symmetric_gaussian m;
    m.write_model_tag_file("mt30.tag");
  }

  {
    model_image_gaussian g;
    model_image_upsample m(g,16);
    m.write_model_tag_file("mt31.tag");
  }

  {
    model_image_vrt2_pmap m;
    m.write_model_tag_file("mt32.tag");
  }

  {
    model_image_xsring m;
    m.write_model_tag_file("mt33.tag");
  }

  {
    model_image_xsringauss m;
    m.write_model_tag_file("mt34.tag");
  }

  
  
  
  

  MPI_Finalize();

  return 0;
}

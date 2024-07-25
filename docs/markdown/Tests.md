Themis Examples and Tests {#Tests}
============

\brief A list of Themis tests to run and Themis examples to follow.

## Examples

To help get started, we have collected a set of well commented examples, illustrating the various steps in setting up a comparison between models and data.  These include importing data, defining and generating model images, running an MCMC sampler to explore the likelihood surface, and performing explicit parameter estimates.  These examples can be compiled and run by executing "make examples" in Themis/src.  Follow the links below for more information about each example.

- Reading in data: reading_data.cpp
- Making an RIAF image: generate_riaf_image.cpp
- How to run a sampler: \link examples/eggbox_mcmc_sampling.cpp eggbox_mcmc_sampling.cpp \endlink
- Gaussian with scattering: scattered_gaussian_fitting.cpp
- RIAF with multiple kinds of data: riaf_model_fitting.cpp
- Making a shearing spot movie: generate_movie_shearing_spot.cpp


## Tests

As Themis is developed we also create various tests, intended to assess different elements of Themis.  

**Automatic Easy Tests**

The first are a subset of rapid component tests used to ensure that key Themis components continue to operate as expected.  These are automatically run periodically at intervals that depend on their importance and difficulty.

- Test the data types: tests/datatypes.cpp
- Test the numerical visibility amplitude & closure phase generation: tests/gaussian_blob.cpp
- Test gaussian model with analytical visibility amplitude data: tests/gaussian_image_comparison.cpp
- Test gaussian model with numerical visibility amplitude data: tests/gaussian_image_comparison_numerical.cpp
- Tets the Symmetric Gaussian model with visibility amplitude data: tests/symmetric_gaussian_comparison.cpp
- Test the Asymmetric Gaussian model with visibility amplitude data: tests/asymmetric_gaussian_image_comparison.cpp
- Test the sampler: tests/base_mcmc_sampling.cpp


**Validation tests**

The second are a set of validation tests used to ensure that Themis reproduces previously published results.  These are integration tests -- where the component tests verify parts of Themis, these validation tests ensure that there are no unforseen problems when they are run together.  These necessarily take considerably longer to run and are therefore collected primarily as part of the historical record of Themis development.  Importantly, all modifications to Themis must avoid changing the operation of these tests, permitting revalidation as required.

- Test the sampler: \link validation/eggbox_mcmc_sampling.cpp validation/eggbox_mcmc_sampling \endlink
- Test the Symmetric Gaussian model with visibility amplitude data: \link validation/symmetric_gaussian_comparison.cpp validation/symmetric_gaussian_comparison \endlink
- Test the Asymmetric Gaussian model with visibility amplitude data: \link validation/asymmetric_gaussian_image_comparison.cpp validation/asymmetric_gaussian_image_comparison \endlink
- Test the Geometric Crescent model with visibility amplitude data: \link validation/crescent_image_comparison.cpp validation/crescent_image_comparison \endlink
- Test the RIAF model with visibility amplitude data: \link validation/riaf_visibility_amplitude.cpp validation/riaf_visibility_amplitude \endlink
- Test the RIAF model with visibility amplitude and closure phase data: \link validation/riaf_va_cp.cpp validation/riaf_va_cp \endlink

/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <iostream>
#include <memory>

#include <BayesFilters/BootstrapCorrection.h>
#include <BayesFilters/DrawParticles.h>
#include <BayesFilters/GaussianLikelihood.h>
#include <BayesFilters/SimulatedLinearSensor.h>
#include <BayesFilters/SimulatedStateModel.h>
#include <BayesFilters/Resampling.h>
#include <BayesFilters/SIS.h>
#include <BayesFilters/NonLinearScalarModel.h>
#include <BayesFilters/utils.h>
#include <Eigen/Dense>

using namespace bfl;
using namespace Eigen;


class SISSimulation : public SIS
{
public:
    SISSimulation
    (
        unsigned int num_particle,
        std::size_t state_size,
        unsigned int simulation_steps,
        std::unique_ptr<ParticleSetInitialization> initialization,
        std::unique_ptr<PFPrediction> prediction,
        std::unique_ptr<PFCorrection> correction,
        std::unique_ptr<Resampling> resampling
    ) noexcept :
        SIS(num_particle, state_size, std::move(initialization), std::move(prediction), std::move(correction), std::move(resampling)),
        simulation_steps_(simulation_steps)
    { }

protected:
    bool runCondition() override
    {
        if (getFilteringStep() < simulation_steps_)
            return true;
        else
            return false;
    }

private:
    unsigned int simulation_steps_;
};


class ParticlesInitialization : public ParticleSetInitialization
{
public:
    ParticlesInitialization(const double& initial_state)
    {
        initial_state_.resize(1);
        initial_state_(0) = initial_state;
    }

    ~ParticlesInitialization()
    { };

protected:
    bool initialize(ParticleSet& particles) override
    {
        // ParticleSet.components contains the number of particles
        for (std::size_t i = 0; i < particles.components; i++)
            particles.state() = initial_state_;

        return true;
    }

    VectorXd initial_state_;
};


int main()
{
    std::cout << "Running a SIS particle filter on a simulated target." << std::endl;
    std::cout << "Data is logged in the test folder with prefix testSIS." << std::endl;

    std::size_t state_size = 1;

    /* A set of parameters needed to run a SIS particle filter in a simulated environment. */
    unsigned int num_particle = 50;
    VectorXd initial_state(state_size, state_size);
    initial_state << 0.1;
    unsigned int simulation_time = 100;

    /* Step 1 - Initialization */
    /* Initialize initialization class. */
    std::unique_ptr<ParticleSetInitialization> particles_initialization = utils::make_unique<ParticlesInitialization>(initial_state(0));


    /* Step 2 - Prediction */
    /* Step 2.1 - Define the state model */
    /* Initialize model taken from example 15.1 in book Optimal State Estimation Kalman, H-infinity, and Nonlinear Approaches. */
    double process_variance = 1.0;

    std::unique_ptr<StateModel> wna = utils::make_unique<NonLinearScalarModel>(process_variance);

    /* Step 2.2 - Define the prediction step */
    /* Initialize the particle filter prediction step and pass the ownership of the state model. */
    std::unique_ptr<PFPrediction> pf_prediction = utils::make_unique<DrawParticles>();
    pf_prediction->setStateModel(std::move(wna));


    /* Step 3 - Correction */
    /* Step 3.1 - Define where the measurement are originated from (either simulated or from a real process) */
    /* Initialize simulaterd target model with a white noise acceleration. */
    std::unique_ptr<StateModel> target_model = utils::make_unique<NonLinearScalarModel>(process_variance);
    std::unique_ptr<SimulatedStateModel> simulated_state_model = utils::make_unique<SimulatedStateModel>(std::move(target_model), initial_state, simulation_time);
    simulated_state_model->enable_log(".", "testSIS");

    /* Step 3.2 - Initialize a measurement model (a linear sensor reading the scalar state x). */
    double meas_sigma = 1.0;
    std::unique_ptr<MeasurementModel> simulated_linear_sensor = utils::make_unique<SimulatedLinearSensor>(std::move(simulated_state_model), meas_sigma, 0.0);
    simulated_linear_sensor->enable_log(".", "testSIS");


    /* Step 3.3 - Define the likelihood model */
    /* Initialize the the exponential likelihood, a PFCorrection decoration of the particle filter correction step. */
    std::unique_ptr<LikelihoodModel> exp_likelihood = utils::make_unique<GaussianLikelihood>();

    /* Step 3.4 - Define the correction step */
    /* Initialize the particle filter correction step and pass the ownership of the measurement model. */
    std::unique_ptr<PFCorrection> pf_correction = utils::make_unique<BoostrapCorrection>();
    pf_correction->setLikelihoodModel(std::move(exp_likelihood));
    pf_correction->setMeasurementModel(std::move(simulated_linear_sensor));


    /* Step 4 - Resampling */
    /* Initialize a resampling algorithm */
    std::unique_ptr<Resampling> resampling = utils::make_unique<Resampling>();


    /* Step 5 - Assemble the particle filter */
    std::cout << "Constructing SIS particle filter..." << std::flush;
    SISSimulation sis_pf(num_particle, state_size, simulation_time, std::move(particles_initialization), std::move(pf_prediction), std::move(pf_correction), std::move(resampling));
    sis_pf.enable_log(".", "testSIS");
    std::cout << "done!" << std::endl;


    /* Step 6 - Prepare the filter to be run */
    std::cout << "Booting SIS particle filter..." << std::flush;
    sis_pf.boot();
    std::cout << "completed!" << std::endl;


    /* Step 7 - Run the filter and wait until it is closed */
    /* Note that since this is a simulation, the filter will end upon simulation termination */
    std::cout << "Running SIS particle filter..." << std::flush;
    sis_pf.run();
    std::cout << "waiting..." << std::flush;
    if (!sis_pf.wait())
        return EXIT_FAILURE;
    std::cout << "completed!" << std::endl;


    return EXIT_SUCCESS;
}

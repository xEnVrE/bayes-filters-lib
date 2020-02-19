/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/SimulatedLinearSensor.h>

using namespace bfl;
using namespace Eigen;


SimulatedLinearSensor::SimulatedLinearSensor
(
    std::unique_ptr<SimulatedStateModel> simulated_state_model,
    const double sigma_x,
    const double sigma_y,
    const unsigned int seed
) :
    LinearModel(sigma_x, sigma_y, seed),
    simulated_state_model_(std::move(simulated_state_model))
{
    /* Since a LinearSensor is intended as a sensor that measure
       the state directly (i.e. no linear combination of the states),
       then it is possible to extract the output size from the
       state description of the simulated state model. */
    VectorDescription state_description = simulated_state_model_->getStateModel().getStateDescription();
    std::size_t state_linear_size = state_description.linear_size;
    std::size_t state_circular_size = state_description.circular_size;

    std::size_t linear_size = 0;
    std::size_t circular_size = 0;

    for (std::size_t i = 0; i < H_.rows(); i++)
    {
        Eigen::MatrixXf::Index state_index;
        H_.row(i).array().abs().maxCoeff(&state_index);
        if (state_index < state_linear_size)
            linear_size++;
        else
            circular_size++;
    }

    input_state_description_ = simulated_state_model_->getStateModel().getStateDescription();
    measurement_description_ = VectorDescription(linear_size, circular_size);
}


SimulatedLinearSensor::SimulatedLinearSensor
(
    std::unique_ptr<SimulatedStateModel> simulated_state_model,
    const double sigma_x,
    const double sigma_y
) :
    SimulatedLinearSensor(std::move(simulated_state_model), sigma_x, sigma_y, 1)
{ }


SimulatedLinearSensor::SimulatedLinearSensor(std::unique_ptr<SimulatedStateModel> simulated_state_model) :
    SimulatedLinearSensor(std::move(simulated_state_model), 10.0, 10.0, 1)
{ }


SimulatedLinearSensor::~SimulatedLinearSensor() noexcept
{ }


bool SimulatedLinearSensor::freeze(const Data& data)
{
    if (!simulated_state_model_->bufferData())
        return false;

    measurement_ = H_ * any::any_cast<MatrixXd>(simulated_state_model_->getData());

    MatrixXd noise;
    std::tie(std::ignore, noise) = getNoiseSample(measurement_.cols());

    measurement_ += noise;

    log();

    return true;
}


std::pair<bool, Data> SimulatedLinearSensor::measure(const Data& data) const
{
    return std::make_pair(true, measurement_);
}


VectorDescription SimulatedLinearSensor::getInputDescription() const
{
    return input_description_;
}


VectorDescription SimulatedLinearSensor::getMeasurementDescription() const
{
    return measurement_description_;
}


void SimulatedLinearSensor::log()
{
    logger(measurement_.transpose());
}

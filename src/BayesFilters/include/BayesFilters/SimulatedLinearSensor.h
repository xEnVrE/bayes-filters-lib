/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef SIMULATEDLINEARSENSOR_H
#define SIMULATEDLINEARSENSOR_H

#include <BayesFilters/LinearModel.h>
#include <BayesFilters/SimulatedStateModel.h>

#include <memory>

namespace bfl {
    class SimulatedLinearSensor;
}


class bfl::SimulatedLinearSensor : public LinearModel
{
public:
    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model, const double sigma_x, const double sigma_y, const unsigned int seed);

    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model, const double sigma_x, const double sigma_y);

    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model);

    virtual ~SimulatedLinearSensor() noexcept;

    bool freeze(const Data& data = Data()) override;

    std::pair<bool, bfl::Data> measure(const Data& data = Data()) const override;

    VectorDescription getInputDescription() const override;

    VectorDescription getMeasurementDescription() const override;

protected:
    std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model_;

    Eigen::MatrixXd measurement_;

    VectorDescription input_description_;

    VectorDescription measurement_description_;

    void log() override;
};

#endif /* SIMULATEDLINEARSENSOR_H */

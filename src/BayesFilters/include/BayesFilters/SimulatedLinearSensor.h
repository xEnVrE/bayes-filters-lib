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
    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model, const float sigma_x, const float sigma_y, const unsigned int seed);

    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model, const float sigma_x, const float sigma_y);

    SimulatedLinearSensor(std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model);

    virtual ~SimulatedLinearSensor() noexcept;

    bool bufferAgentData() const override;

    std::pair<bool, bfl::Data> getAgentMeasurements() const override;

protected:
    std::unique_ptr<bfl::SimulatedStateModel> simulated_state_model_;
};

#endif /* SIMULATEDLINEARSENSOR_H */
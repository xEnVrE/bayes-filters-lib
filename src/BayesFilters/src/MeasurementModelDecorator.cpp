/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/MeasurementModelDecorator.h>

using namespace bfl;
using namespace Eigen;


MeasurementModelDecorator::MeasurementModelDecorator(std::unique_ptr<MeasurementModel> observation_model) noexcept :
    measurement_model(std::move(observation_model)) { }


MeasurementModelDecorator::MeasurementModelDecorator(MeasurementModelDecorator&& observation_model) noexcept :
    measurement_model(std::move(observation_model.measurement_model)) { }


MeasurementModelDecorator::~MeasurementModelDecorator() noexcept { }


MeasurementModelDecorator& MeasurementModelDecorator::operator=(MeasurementModelDecorator&& observation_model) noexcept
{
    measurement_model = std::move(observation_model.measurement_model);

    return *this;
}


std::pair<bool, Data> MeasurementModelDecorator::measure(const Data& data) const
{
    return measurement_model->measure();
}


std::pair<bool, Data> MeasurementModelDecorator::predictedMeasure(const Ref<const MatrixXd>& cur_states)
{
    return measurement_model->predictedMeasure(cur_states);
}


std::pair<bool, Data> MeasurementModelDecorator::innovation(const Data& predicted_measurements, const Data& measurements) const
{
    return measurement_model->innovation(predicted_measurements, measurements);
}


std::pair<bool, MatrixXd> MeasurementModelDecorator::getNoiseCovarianceMatrix() const
{
    return measurement_model->getNoiseCovarianceMatrix();
}


bool MeasurementModelDecorator::setProperty(const std::string& property)
{
    return measurement_model->setProperty(property);
}


bool MeasurementModelDecorator::freeze(const Data& data)
{
    return measurement_model->freeze();
}


VectorDescription MeasurementModelDecorator::getInputDescription() const
{
    return measurement_model->getInputDescription();
}


VectorDescription MeasurementModelDecorator::getMeasurementDescription() const
{
    return measurement_model->getMeasurementDescription();
}

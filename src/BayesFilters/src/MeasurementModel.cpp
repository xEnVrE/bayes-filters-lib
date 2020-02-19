/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/MeasurementModel.h>

#include <iostream>

using namespace bfl;
using namespace Eigen;


MeasurementModel::~MeasurementModel() noexcept
{ }


std::pair<bool, MatrixXd> MeasurementModel::getNoiseCovarianceMatrix() const
{
    return std::make_pair(false, MatrixXd::Zero(1, 1));
}


bool MeasurementModel::setProperty(const std::string& property)
{
    static_cast<void>(property);
    return false;
}


VectorDescription MeasurementModel::getInputDescription() const
{
    /* Not all measurement model have a vector-valued input. */
    throw std::runtime_error("ERROR::MEASUREMENTMODEL::GETINPUTSTATEDESCRIPTION\nERROR:\n\tMethod not implemented.");
}


VectorDescription MeasurementModel::getMeasurementDescription() const
{
    /* Not all measurement model have a vector-valued output. */
    throw std::runtime_error("ERROR::MEASUREMENTMODEL::GETMEASUREMENTDESCRIPTION\nERROR:\n\tMethod not implemented.");
}

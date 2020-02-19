/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/StateModel.h>

using namespace bfl;
using namespace Eigen;


Eigen::MatrixXd StateModel::getJacobian()
{
    throw std::runtime_error("ERROR::STATEMODEL::GETJACOBIAN\nERROR:\n\tMethod not implemented.");
}


Eigen::VectorXd StateModel::getTransitionProbability(const Ref<const MatrixXd>& prev_states, const Ref<const MatrixXd>& cur_states)
{
    throw std::runtime_error("ERROR::STATEMODEL::TRANSITIONPROBABILITY\nERROR:\n\tMethod not implemented.");
}


Eigen::MatrixXd StateModel::getNoiseCovarianceMatrix() const
{
    throw std::runtime_error("ERROR::STATEMODEL::GETNOISECOVARIANCEMATRIX\nERROR:\n\tMethod not implemented.");
}


Eigen::MatrixXd StateModel::getNoiseSample(const std::size_t num)
{
    throw std::runtime_error("ERROR::STATEMODEL::GETNOISESAMPLE\nERROR:\n\tMethod not implemented.");
}


VectorDescription StateModel::getInputDescription() const
{
    /* By default, we assume that the input to the discrete time state model is the state at the previous time step.
       Hence, the description of the input is the same as the state description. */
    return getStateDescription();
}

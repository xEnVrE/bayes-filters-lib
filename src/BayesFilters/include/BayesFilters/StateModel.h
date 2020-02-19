/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef STATEMODEL_H
#define STATEMODEL_H

#include <BayesFilters/VectorDescription.h>

#include <Eigen/Dense>

namespace bfl {
    class StateModel;
}


class bfl::StateModel
{
public:
    virtual ~StateModel() noexcept { };

    virtual void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> prop_states) = 0;

    virtual void motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> mot_states) = 0;

    virtual Eigen::MatrixXd getJacobian();

    virtual Eigen::VectorXd getTransitionProbability(const Eigen::Ref<const Eigen::MatrixXd>& prev_states, const Eigen::Ref<const Eigen::MatrixXd>& cur_states);

    virtual Eigen::MatrixXd getNoiseCovarianceMatrix() const;

    virtual Eigen::MatrixXd getNoiseSample(const std::size_t num);

    virtual bool setProperty(const std::string& property) = 0;

    virtual VectorDescription getInputDescription() const;

    virtual VectorDescription getStateDescription() const = 0;
};

#endif /* STATEMODEL_H */

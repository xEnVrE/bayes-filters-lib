/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BayesFilters/AdditiveStateModel.h>

#include <Eigen/Dense>

using namespace bfl;
using namespace Eigen;


void AdditiveStateModel::motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> mot_states)
{
    propagate(cur_states, mot_states);

    mot_states += getNoiseSample(mot_states.cols());
}


VectorDescription AdditiveStateModel::getInputDescription() const
{
    VectorDescription input_description = getStateDescription();
    input_description.noise_components = getNoiseCovarianceMatrix().rows();
    input_description.noise_size = input_description.noise_components;
    input_description.total_size += input_description.noise_size;

    return input_description;
}

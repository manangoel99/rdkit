//
//  Copyright (C) 2020 Manan Goel
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#include "AtomicContrib.h"
#include <ForceField/ForceField.h>
#include <RDGeneral/Invariant.h>
#include <RDGeneral/utils.h>
#include <Numerics/EigenSerializer/EigenSerializer.h>
#include <Eigen/Dense>
#include <GraphMol/Descriptors/AtomicEnvironmentVector.h>
#include <fstream>
using namespace Eigen;

namespace ForceFields {
namespace ANI {
ANIAtomContrib::ANIAtomContrib(ForceField *owner, int atomType,
                               unsigned int atomIdx, VectorXi &speciesVec,
                               unsigned int numAtoms, unsigned int numLayers,
                               unsigned int ensembleSize,
                               std::string modelType) {
  PRECONDITION(owner, "Bad Owner")
  PRECONDITION(atomType == 0 || atomType == 1 || atomType == 2 || atomType == 3,
               "Atom Type not Supported");
  PRECONDITION(modelType == "ANI-1x" || modelType == "ANI-1ccx",
               "Model Not currently supported")
  PRECONDITION(ensembleSize > 0,
               "There must be at least 1 model in the ensemble");
  URANGE_CHECK(atomIdx, numAtoms);
  dp_forceField = owner;
  this->d_atomType = atomType;
  this->d_atomIdx = atomIdx;
  this->d_speciesVec = speciesVec;
  this->d_numAtoms = numAtoms;
  this->d_ensembleSize = ensembleSize;
  this->d_modelType = modelType;

  if (this->d_atomEncoding.find(this->d_atomType) !=
      this->d_atomEncoding.end()) {
    auto atomicSymbol = this->d_atomEncoding[this->d_atomType];
    for (unsigned int modelNum = 0; modelNum < ensembleSize; modelNum++) {
      std::vector<MatrixXd> currModelWeights;
      std::vector<MatrixXd> currModelBiases;
      Utils::loadFromBin(&currModelWeights, &currModelBiases, modelNum,
                         atomicSymbol, this->d_modelType);
      this->d_weights.push_back(currModelWeights);
      this->d_biases.push_back(currModelBiases);
    }
    Utils::loadSelfEnergy(&(this->d_selfEnergy), atomicSymbol,
                          this->d_modelType);
  } else {
    this->d_selfEnergy = 0;
  }

  // Different values for means of the gaussian symmetry functions
  std::string path = getenv("RDBASE");
  std::string paramFilePath =
      path + "/Code/ForceField/ANI/Params/" + modelType + "/AEVParams/";

  // Weights for the radial symmetry functions
  ArrayXd ShfR;
  RDNumeric::EigenSerializer::deserialize(ShfR, paramFilePath + "ShfR.bin");
  // Variance terms for the gaussian symmetry functions
  ArrayXd EtaR;
  RDNumeric::EigenSerializer::deserialize(EtaR, paramFilePath + "EtaR.bin");

  // Weights for the angular symmetry functions
  ArrayXd ShfZ;
  RDNumeric::EigenSerializer::deserialize(ShfZ, paramFilePath + "ShfZ.bin");
  ArrayXd ShfA;
  RDNumeric::EigenSerializer::deserialize(ShfA, paramFilePath + "ShfA.bin");
  // distance wise shifts in the distance term of the angular symmetry function

  ArrayXd zeta;
  RDNumeric::EigenSerializer::deserialize(zeta, paramFilePath + "zeta.bin");
  ArrayXd etaA;
  RDNumeric::EigenSerializer::deserialize(etaA, paramFilePath + "etaA.bin");

  this->d_aevParams.insert(std::make_pair("ShfR", ShfR));
  this->d_aevParams.insert(std::make_pair("EtaR", EtaR));
  this->d_aevParams.insert(std::make_pair("ShfZ", ShfZ));
  this->d_aevParams.insert(std::make_pair("ShfA", ShfA));
  this->d_aevParams.insert(std::make_pair("zeta", zeta));
  this->d_aevParams.insert(std::make_pair("etaA", etaA));
}

double ANIAtomContrib::forwardProp(ArrayXXd &aev) const {
  if (this->d_atomType == -1) {
    return 0;
  }

  if (aev.cols() != 1) {
    aev.transposeInPlace();
  }

  MatrixXd aevMat = aev.matrix();

  std::vector<double> energies;
  energies.reserve(this->d_weights.size());
  for (unsigned int modelNo = 0; modelNo < this->d_weights.size(); modelNo++) {
    auto temp = aevMat;
    for (unsigned int layer = 0; layer < this->d_weights[modelNo].size();
         layer++) {
      temp = ((this->d_weights[modelNo][layer] * temp) +
              this->d_biases[modelNo][layer])
                 .eval();
      if (layer < this->d_weights[modelNo].size() - 1) {
        Utils::CELU(temp, 0.1);
      }
    }
    energies.push_back(temp.coeff(0, 0));
  }
  return std::accumulate(energies.begin(), energies.end(), 0.0) /
         energies.size();
}

double ANIAtomContrib::getEnergy(double *pos) const {
  auto aev = RDKit::Descriptors::ANI::AtomicEnvironmentVector(
      pos, this->d_speciesVec, this->d_numAtoms, &(this->d_aevParams));
  ArrayXXd row = aev.row(this->d_atomIdx);
  return this->ANIAtomContrib::forwardProp(row) + this->d_selfEnergy;
}

double ANIAtomContrib::getEnergy(Eigen::ArrayXXd &aev) const {
  ArrayXXd row = aev.row(this->d_atomIdx);
  return this->ANIAtomContrib::forwardProp(row) + this->d_selfEnergy;
}

void ANIAtomContrib::getGrad(double *pos, double *grad) const {
  auto aev = RDKit::Descriptors::ANI::AtomicEnvironmentVector(
      pos, this->d_speciesVec, this->d_numAtoms, &(this->d_aevParams));

  MatrixXd row = aev.row(this->d_atomIdx).matrix();
  std::vector<MatrixXd> hiddenStates;
  std::vector<MatrixXd> grads;
  for (unsigned int modelNo = 0; modelNo < this->d_weights.size(); modelNo++) {
    auto temp = row;
    for (unsigned int layer = 0; layer < this->d_weights[modelNo].size();
         layer++) {
      temp = ((this->d_weights[modelNo][layer] * temp) +
              this->d_biases[modelNo][layer])
                 .eval();
      hiddenStates.push_back(temp);
      if (layer < this->d_weights[modelNo].size() - 1) {
        Utils::CELU(temp, 0.1);
      }
    }
    MatrixXd gradient = MatrixXd::Identity(this->d_weights[modelNo][0].cols(),
                                           this->d_weights[modelNo][0].cols());
    for (unsigned int i = 0; i < this->d_weights[modelNo].size() - 1; i++) {
      Utils::CELUGrad(hiddenStates[i], 0.1);
      auto k = hiddenStates[i].asDiagonal() * this->d_weights[modelNo][i];
      gradient = (k * gradient).eval();
    }
    gradient = this->d_weights[modelNo][this->d_weights[modelNo].size() - 1] *
               gradient;
    grads.push_back(gradient);
    hiddenStates.clear();
  }
  MatrixXd final_grad = MatrixXd::Zero(row.rows(), row.cols());
  for (auto i : grads) {
    final_grad += i;
  }
  final_grad = final_grad / this->d_ensembleSize;

  // AEV derivative w.r.t position
  MatrixXd radialPart(4, 16);
  unsigned int col = 0;
  for (int i = 0; i < 64; i++) {
    radialPart(i / 16, col) = row(i, 0);
    col++;
    if (i % 16 == 0) {
      col = 0;
    }
  }
  auto numAtoms = this->d_numAtoms;
  auto species = this->d_speciesVec;
  ArrayXXd coordinates(numAtoms, 3);
  for (unsigned int i = 0; i < numAtoms; i++) {
    coordinates.row(i) << pos[3 * i], pos[3 * i + 1], pos[3 * i + 2];
  }
  // Fetch pairs of atoms which are neigbours which lie within the cutoff
  // distance 5.2 Angstroms. The constant was obtained by authors of torchANI
  ArrayXi atomIndex12;
  RDKit::Descriptors::ANI::NeighborPairs(&coordinates, &species, 5.2, numAtoms,
                                         &atomIndex12);
  ArrayXXd selectedCoordinates(atomIndex12.rows(), 3);
  RDKit::Descriptors::ANI::IndexSelect(&selectedCoordinates, &coordinates,
                                       atomIndex12, 0);

  // Vectors between pairs of atoms that lie in each other's neighborhoods
  unsigned int numPairs = selectedCoordinates.rows() / 2;
  ArrayXXd vec(numPairs, 3);
  for (unsigned int i = 0; i < numPairs; i++) {
    vec.row(i) =
        selectedCoordinates.row(i) - selectedCoordinates.row(i + numPairs);
  }

  ArrayXXd distances = vec.matrix().rowwise().norm().array();

  ArrayXXi species12(2, numPairs);
  ArrayXXi species12Flipped(2, numPairs);
  ArrayXXi atomIndex12Unflattened(2, numPairs);
  for (unsigned int i = 0; i < numPairs; i++) {
    species12(0, i) = species(atomIndex12(i));
    species12(1, i) = species(atomIndex12(i + numPairs));

    species12Flipped(1, i) = species(atomIndex12(i));
    species12Flipped(0, i) = species(atomIndex12(i + numPairs));

    atomIndex12Unflattened(0, i) = atomIndex12(i);
    atomIndex12Unflattened(1, i) = atomIndex12(i + numPairs);
  }
  std::map<int, std::vector<int>> addedMapping;
  for (int i = 4 * this->d_atomIdx; i < 4 * this->d_atomIdx + 4; i++) {
    std::vector<int> k;
    addedMapping.insert(std::make_pair(i, k));
  }
  auto index12 = (atomIndex12Unflattened * 4 + species12Flipped).transpose();
  for (auto idxCol = 0; idxCol < index12.cols(); idxCol++) {
    for (auto i = 0; i < index12.rows(); i++) {
      for (auto v = addedMapping.begin(); v != addedMapping.end(); v++) {
        if (index12(i, idxCol) == v->first) {
          addedMapping[v->first].push_back(i);
        }
      }
    }
  }
  std::vector<ArrayXXd> derivatives;
  Utils::RadialTerms_d(5.2, derivatives, addedMapping, selectedCoordinates,
                       &(this->d_aevParams), distances, atomIndex12,
                       this->d_atomIdx);
  ArrayXXd RadialJacobian = ArrayXXd::Zero(384, 3);
  unsigned int idx = 0;
  for (auto i : derivatives) {
    for (int j = 0; j < i.rows(); j++) {
      RadialJacobian.row(idx) << i.row(j);
      idx++;
    }
    // std::cout << i << std::endl;
    // std::cout << "***************" << std::endl;
  }
  // std::cout << RadialJacobian.rows() << " " << RadialJacobian.cols()
            // << std::endl;
  // std::cout << final_grad.rows() << " " << final_grad.cols() << std::endl;
  // std::cout << final_grad.matrix() * RadialJacobian.matrix() << std::endl;
  // std::cout << Jacobian << std::endl;
  // std::cout << Jacobian.colwise().sum() << std::endl;
  // std::cout << "=======================" << std::endl;
}

namespace Utils {

void RadialTerms_d(double cutoff, std::vector<ArrayXXd> &derivatives,
                   std::map<int, std::vector<int>> &addedMapping,
                   ArrayXXd &selectedCoordinates,
                   const std::map<std::string, Eigen::ArrayXXd> *params,
                   ArrayXXd &distances, ArrayXi &atomIndex12,
                   unsigned int atomIdx) {
  ArrayXd EtaR = params->find("EtaR")->second;
  ArrayXd ShfR = params->find("ShfR")->second;
  auto numPairs = selectedCoordinates.rows() / 2;
  for (auto i = addedMapping.begin(); i != addedMapping.end(); i++) {
    auto addedRows = i->second;
    ArrayXXd der = ArrayXXd::Zero(16, 3);
    for (auto v : addedRows) {
      auto idx1 = atomIndex12(v);
      auto idx2 = atomIndex12(v + numPairs);
      auto dist = distances(v);
      auto coord1 = selectedCoordinates.row(v);
      auto coord2 = selectedCoordinates.row(v + numPairs);
      int multi = 1;
      if (atomIdx == idx1) {
        multi = 1;
      }
      if (atomIdx == idx2) {
        multi = -1;
      }
      auto vec = multi * (coord1 - coord2) / dist;
      for (auto etaIdx = 0; etaIdx < EtaR.size(); etaIdx++) {
        ArrayXXd term1 = ((ShfR - dist).pow(2) * EtaR(etaIdx) * -1).exp();
        ArrayXXd term2 =
            (-M_PI / (2 * cutoff) * std::sin((M_PI * dist / cutoff))) +
            EtaR(etaIdx) * (ShfR - dist) *
                (std::cos((M_PI * dist / cutoff)) + 1);
        auto intermediate = 0.25 * term1 * term2;
        for (int k = 0; k < intermediate.size(); ++k) {
          der.row(k) += vec * intermediate(k);
        }
      }
    }
    derivatives.push_back(der);
  }
}

// void RadialTerms_d(double cutoff, ArrayXXd *distances, ArrayXXd
// &RadialTerms_,
//                    const std::map<std::string, Eigen::ArrayXXd> *params) {
// ArrayXd EtaR = params->find("EtaR")->second;
// ArrayXd ShfR = params->find("ShfR")->second;
//   RadialTerms_.resize(distances->rows(), ShfR.size() * EtaR.size());

//   for (auto i = 0; i < distances->rows(); i++) {
//     ArrayXXd calculatedRowVector(1, ShfR.size() * EtaR.size());
//     unsigned int idx = 0;
// for (auto etaIdx = 0; etaIdx < EtaR.size(); etaIdx++) {
//   ArrayXXd term1 =
//       ((ShfR - (*distances)(i)).pow(2) * EtaR(etaIdx) * -1).exp();
//   ArrayXXd term2 =
//       (-M_PI / (2 * cutoff) * std::sin((M_PI * (*distances)(i) /
//       cutoff))) + EtaR(etaIdx) * (ShfR - (*distances)(i)) *
//           (std::cos((M_PI * (*distances)(i) / cutoff)) + 1);
//   auto intermediate = 0.25 * term1 * term2;

//   for (unsigned int j = 0; j < intermediate.size(); j++) {
//     calculatedRowVector(0, idx + j) = intermediate(j);
//   }
//   idx += ShfR.size();
// }
//     RadialTerms_.row(i) = calculatedRowVector;
//   }
// }

void CELU(MatrixXd &input, double alpha) {
  input = input.unaryExpr([&](double val) {
    return std::max(0.0, val) +
           std::min(alpha * (std::exp(val / alpha) - 1), 0.0);
  });
}

void CELUGrad(MatrixXd &input, double alpha) {
  input = input.unaryExpr([&](double val) {
    if (val > 0) {
      return 1.0;
    } else {
      return std::exp(val / alpha);
    }
  });
}

std::vector<std::string> tokenize(const std::string &s) {
  boost::char_separator<char> sep(", \n\r\t");
  boost::tokenizer<boost::char_separator<char>> tok(s, sep);
  std::vector<std::string> tokens;
  std::copy(tok.begin(), tok.end(),
            std::back_inserter<std::vector<std::string>>(tokens));
  return tokens;
}

void loadFromBin(std::vector<MatrixXd> *weights, unsigned int model,
                 std::string weightType, unsigned int layer,
                 std::string atomType, std::string modelType) {
  std::string path = getenv("RDBASE");
  std::string paramFile = path + "/Code/ForceField/ANI/Params/" + modelType +
                          "/model" + std::to_string(model) + "/" + atomType +
                          "_" + std::to_string(layer) + "_" + weightType +
                          ".bin";
  MatrixXf weight;
  RDNumeric::EigenSerializer::deserialize(weight, paramFile);
  weights->push_back(weight.cast<double>());
}

void loadFromBin(std::vector<MatrixXd> *weights, std::vector<MatrixXd> *biases,
                 unsigned int model, std::string atomType,
                 std::string modelType) {
  std::string path = getenv("RDBASE");
  std::string paramFile = path + "/Code/ForceField/ANI/Params/" + modelType +
                          "/model" + std::to_string(model) + ".bin";
  std::vector<MatrixXf> floatWeights, floatBiases;
  RDNumeric::EigenSerializer::deserializeAll(&floatWeights, &floatBiases,
                                             paramFile, atomType);
  for (unsigned int i = 0; i < floatWeights.size(); i++) {
    weights->push_back(floatWeights[i].cast<double>());
    biases->push_back(floatBiases[i].cast<double>());
  }
}

void loadFromCSV(std::vector<MatrixXd> *weights, unsigned int model,
                 std::string weightType, unsigned int layer,
                 std::string atomType, std::string modelType) {
  std::string path = getenv("RDBASE");
  std::string paramFile = path + "/Code/ForceField/ANI/Params/" + modelType +
                          "/model" + std::to_string(model) + "/" + atomType +
                          "_" + std::to_string(layer) + "_" + weightType;

  std::ifstream instrmSF(paramFile.c_str());
  if (!instrmSF.good()) {
    throw ValueErrorException(paramFile + " Model File does not exist");
    return;
  }
  std::string line;
  std::vector<std::string> tokens;
  std::vector<std::vector<double>> weight;
  unsigned int cols = 1;
  while (!instrmSF.eof()) {
    std::getline(instrmSF, line);
    tokens = tokenize(line);
    std::vector<double> row;
    for (auto v : tokens) {
      std::istringstream os(v);
      double d;
      os >> d;
      row.push_back(d);
    }
    if (row.size() > 0) {
      cols = row.size();
      weight.push_back(row);
    }
  }

  MatrixXd param(weight.size(), cols);

  for (unsigned int i = 0; i < weight.size(); i++) {
    for (unsigned int j = 0; j < weight[i].size(); j++) {
      param(i, j) = weight[i][j];
    }
  }
  weights->push_back(param);
}

void loadSelfEnergy(double *energy, std::string atomType,
                    std::string modelType) {
  std::string path = getenv("RDBASE");
  std::string filePath =
      path + "/Code/ForceField/ANI/Params/" + modelType + "/selfEnergies";

  std::ifstream selfEnergyFile(filePath.c_str());
  if (!selfEnergyFile.good()) {
    throw ValueErrorException(filePath + " : File Does Not Exist");
    return;
  }
  std::string line;
  while (!selfEnergyFile.eof()) {
    std::getline(selfEnergyFile, line);
    boost::char_separator<char> sep(" ,=");
    boost::tokenizer<boost::char_separator<char>> tok(line, sep);
    std::vector<std::string> tokens;
    std::copy(tok.begin(), tok.end(),
              std::back_inserter<std::vector<std::string>>(tokens));

    if (tokens[0] == atomType) {
      std::istringstream os(tokens[2]);
      os >> *energy;
      break;
    }
  }
  selfEnergyFile.close();
}

}  // namespace Utils
}  // namespace ANI
}  // namespace ForceFields
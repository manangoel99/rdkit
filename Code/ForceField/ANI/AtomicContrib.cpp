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
  }
  ArrayXi evenCloserIndices((distances.array() <= 3.5).count());
  idx = 0;
  for (auto i = 0; i < distances.size(); i++) {
    if (distances(i) <= 3.5) {
      evenCloserIndices(idx) = i;
      idx++;
    }
  }

  ArrayXXi species12Angular(2, evenCloserIndices.size());
  ArrayXXi atomIndex12Angular(2, evenCloserIndices.size());

  ArrayXXd vecAngular(evenCloserIndices.size(), 3);

  RDKit::Descriptors::ANI::IndexSelect(&species12Angular, &species12,
                                       evenCloserIndices, 1);
  RDKit::Descriptors::ANI::IndexSelect(
      &atomIndex12Angular, &atomIndex12Unflattened, evenCloserIndices, 1);
  RDKit::Descriptors::ANI::IndexSelect(&vecAngular, &vec, evenCloserIndices, 0);

  auto n = evenCloserIndices.size();
  std::pair<std::vector<int>, ArrayXXi> tripletInfo;

  RDKit::Descriptors::ANI::TripleByMolecules(&atomIndex12Angular, &tripletInfo);
  auto pairIndex12 = tripletInfo.second;
  auto centralAtomIndex = tripletInfo.first;
  ArrayXXi sign12(2, pairIndex12.cols());

  // compute mapping between representation of central-other to pair
  for (auto i = 0; i < pairIndex12.rows(); i++) {
    for (auto j = 0; j < pairIndex12.cols(); j++) {
      if (pairIndex12(i, j) < n) {
        sign12(i, j) = 1;
      } else {
        sign12(i, j) = -1;
      }
    }
  }

  n = atomIndex12Angular.cols();

  // pairIndex12 = pairindex12 % n
  auto localIndex = pairIndex12.cast<int>();

  pairIndex12 =
      (localIndex.array() - (localIndex.array() / n).array() * n).array();

  std::map<int, std::pair<ArrayXXd, ArrayXXd>> vecCoordMapping;
  for (int i = 0; i < evenCloserIndices.size(); i++) {
    auto ans = std::make_pair(
        evenCloserIndices(i),
        std::make_pair(
            selectedCoordinates.row(evenCloserIndices(i)),
            selectedCoordinates.row(evenCloserIndices(i) + numPairs)));
  }


  ArrayXi pairIndex12Flattened(2 * pairIndex12.cols());
  idx = 0;
  for (auto i = 0; i < pairIndex12.rows(); i++) {
    for (auto j = 0; j < pairIndex12.cols(); j++) {
      pairIndex12Flattened(idx) = pairIndex12(i, j);
      idx++;
    }
  }

  ArrayXXd vecFlattened(pairIndex12Flattened.size(), 3);
  RDKit::Descriptors::ANI::IndexSelect(&vecFlattened, &vecAngular,
                                       pairIndex12Flattened, 0);
  ArrayXXd vec12(vecFlattened.rows(), 3);
  for (auto i = 0; i < vecFlattened.rows() / 2; i++) {
    vec12.row(i) = vecFlattened.row(i) * sign12(0, i);
  }

  for (auto i = vecFlattened.rows() / 2; i < vecFlattened.rows(); i++) {
    vec12.row(i) = vecFlattened.row(i) * sign12(1, i - vecFlattened.rows() / 2);
  }
  std::vector<ArrayXXd> angularDerivatives;
  Utils::AngularTerms_d(3.5, angularDerivatives, vec12, &(this->d_aevParams));
  std::cout << angularDerivatives.size() << std::endl;
  std::cout << "=============================================" << std::endl;

  ArrayXXi centralAtomIndexArr(centralAtomIndex.size(), 1);

  for (size_t i = 0; i < centralAtomIndex.size(); i++) {
    centralAtomIndexArr.row(i) << centralAtomIndex[i];
  }

  ArrayXXi species12Small1(2, pairIndex12.cols());
  ArrayXXi species12Small2(2, pairIndex12.cols());

  for (auto i = 0; i < pairIndex12.rows(); i++) {
    for (auto j = 0; j < pairIndex12.cols(); j++) {
      species12Small1(i, j) = species12Angular(0, pairIndex12(i, j));
    }
  }

  for (auto i = 0; i < pairIndex12.rows(); i++) {
    for (auto j = 0; j < pairIndex12.cols(); j++) {
      species12Small2(i, j) = species12Angular(1, pairIndex12(i, j));
    }
  }

  ArrayXXi species12_(sign12.rows(), sign12.cols());

  for (auto i = 0; i < sign12.rows(); i++) {
    for (auto j = 0; j < sign12.cols(); j++) {
      if (sign12(i, j) == 1) {
        species12_(i, j) = species12Small2(i, j);
      } else {
        species12_(i, j) = species12Small1(i, j);
      }
    }
  }

  ArrayXXi index(species12_.cols(), 1);
  ArrayXXi triuIndices;
  RDKit::Descriptors::ANI::TriuIndex(4, triuIndices);

  for (auto i = 0; i < species12_.cols(); i++) {
    index.row(i) = triuIndices(species12_(0, i), species12_(1, i));
  }
  // The constant 10 comes from 10 pairs that can be formed
  index = index + (centralAtomIndexArr.array() * 10).array();

  std::vector<ArrayXXd> sumDerivatives;
  sumDerivatives.reserve(10 * numAtoms);


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

void AngularTerms_d(double cutoff, std::vector<ArrayXXd> &derivatives,
                    ArrayXXd &vectors12,
                    const std::map<std::string, ArrayXXd> *params) {
  ArrayXd ShfZ = params->find("ShfZ")->second;
  ArrayXd ShfA = params->find("ShfA")->second;
  ArrayXd zeta = params->find("zeta")->second;
  ArrayXd etaA = params->find("etaA")->second;
  for (int i = 0; i < vectors12.rows() / 2; i++) {
    auto vecij = vectors12.matrix().row(i);
    auto vecik = vectors12.matrix().row(i + vectors12.rows() / 2);

    auto Rij = vecij.norm();
    auto Rik = vecik.norm();

    auto dotProduct = vecij.dot(vecik);

    auto thetaijk = std::acos(0.95 * dotProduct / (Rij * Rik));
    unsigned int idx = 0;
    ArrayXXd der(32, 3);
    for (int ShfZidx = 0; ShfZidx < ShfZ.size(); ShfZidx++) {
      for (int ShfAidx = 0; ShfAidx < ShfA.size(); ShfAidx++) {
        for (int zetaidx = 0; zetaidx < zeta.size(); zetaidx++) {
          for (int etaAidx = 0; etaAidx < etaA.size(); etaAidx++) {
            auto expTerm = std::exp(-etaA(etaAidx) * std::pow((Rij + Rik)/2 - ShfA(ShfAidx), 2));
            auto term1 = 1;
            term1 *= zeta(zetaidx) *
                     std::pow(std::cos(thetaijk - ShfZ(ShfZidx)) + 1,
                              zeta(zetaidx) - 1) *
                     std::sin(thetaijk - ShfZ(ShfZidx));
            auto cutoff_ij = 0.5 * (std::cos(M_PI * Rij / cutoff) + 1);
            auto cutoff_ik = 0.5 * (std::cos(M_PI * Rik / cutoff) + 1);
            term1 *= (cutoff_ij * cutoff_ik) / (Rij * Rik);
            auto vectorij = 0.95 * vecij * (1 - dotProduct / (Rij * Rij));
            auto vectorik = 0.95 * vecik * (1 - dotProduct / (Rik * Rik));
            auto part1 = term1 * (vectorij + vectorik) /
                         std::sqrt(1 - std::pow(std::cos(thetaijk), 2)) * expTerm;
            auto part2 = M_PI * std::pow((std::cos(thetaijk - ShfZ(ShfZidx)) + 1), zeta(zetaidx)) * cutoff_ij * std::sin(M_PI * Rik / cutoff) * vecik * expTerm / (2 * cutoff * Rik);
            auto part3 = M_PI * std::pow((std::cos(thetaijk - ShfZ(ShfZidx)) + 1), zeta(zetaidx)) * cutoff_ik * std::sin(M_PI * Rij / cutoff) * vecij * expTerm / (2 * cutoff * Rij);
            auto part4 = - etaA(etaAidx) * std::pow((std::cos(thetaijk - ShfZ(ShfZidx)) + 1), zeta(zetaidx)) * cutoff_ij * cutoff_ik * ((Rij + Rik) / 2 - ShfA(ShfAidx)) * (- 1 * (vecik / Rik) - (vecij / Rij)) * expTerm;
            der.row(idx) << std::pow(2, 1 - zeta(zetaidx)) * (part1 + part2 + part3 + part4);
            idx++;
          }
        }
      }
    }
    derivatives.push_back(der);
  }
}

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
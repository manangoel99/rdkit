//
//  Copyright (C) 2020 Manan Goel
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#ifndef AtomicEnvironmentVectorRDKIT_H_JUNE2020
#define AtomicEnvironmentVectorRDKIT_H_JUNE2020
#ifdef RDK_HAS_EIGEN3

#ifdef RDK_BUILD_DESCRIPTORS3D
#include <Eigen/Dense>
namespace RDKit {
class ROMol;
namespace Descriptors {
namespace ANI {
const std::string AtomicEnvironmentVectorVersion = "1.0.0";

//-------------------------------------------------------
//! Generates a vector from the molecule containing encoding of each atom
// such that H -> 0, C -> 1, N -> 2, O -> 3 and all other atoms -> -1
/*!
  \param mol A mol object

  \return Vector containing encoding of atoms in the molecule
*/
RDKIT_DESCRIPTORS_EXPORT Eigen::VectorXi GenerateSpeciesVector(
    const ROMol &mol);

RDKIT_DESCRIPTORS_EXPORT Eigen::VectorXi GenerateSpeciesVector(
    const int *atomNums, unsigned int numAtoms);

//! Calculates torchANI style symmetry functions combining both radial and
//! angular terms
/*!
  \param mol      Mol object for which symmetry functions are to be found
  \param confId   Conformer ID for the conformer for which symmetry
  functions are to be found

  \return numAtoms * 384 shaped matrix containing 384 features for every
  atom in the input mol consisting of both radial and angular terms
*/
RDKIT_DESCRIPTORS_EXPORT void AtomicEnvironmentVector(
    Eigen::ArrayXXd &AEV, const ROMol &mol,
    const std::map<std::string, Eigen::ArrayXXd> *params, int confId = -1);

//! Calculates torchANI style symmetry functions combining both radial and
//! angular terms
/*!
  \param pos      Array of positions of atoms
  \param species  Encoding of atom types with index
  \param numAtoms Number of Atoms

  \return numAtoms * 384 shaped matrix containing 384 features for every atom in
  the input mol consisting of both radial and angular terms
*/
RDKIT_DESCRIPTORS_EXPORT void AtomicEnvironmentVector(
    Eigen::ArrayXXd &AEV, double *pos, const Eigen::VectorXi &species,
    unsigned int numAtoms,
    const std::map<std::string, Eigen::ArrayXXd> *params);

}  // namespace ANI
}  // namespace Descriptors
}  // namespace RDKit
#endif
#endif
#endif

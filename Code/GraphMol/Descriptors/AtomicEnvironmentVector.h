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

RDKIT_DESCRIPTORS_EXPORT void TriuIndex(unsigned int numSpecies, Eigen::ArrayXXi &triuIndices);

//! Calculates the value a continuous smoothening function for a distance such
//! that values
// greater than the cutoff give 0
/*!
  \param distances A 2 dimensional array of pairwise distances
  \param cutoff    A double value signifying cutoff distance

  \return 2 dimensional array containing corresponding values computed by cutoff
  function
*/
template <typename Derived>
RDKIT_DESCRIPTORS_EXPORT Eigen::ArrayXXd CosineCutoff(
    Eigen::ArrayBase<Derived> *distances, double cutoff) {
  // Cosine cutoff function assuming all distances are less than the cutoff
  PRECONDITION(cutoff > 0.0, "Cutoff must be greater than zero");
  PRECONDITION(((*distances) <= cutoff).count() == distances->size(),
               "All distances must be less than the cutoff");
  PRECONDITION(distances != nullptr, "Array of distances is NULL");
  return 0.5 * ((*distances) * (M_PI / cutoff)).cos() + 0.5;
}

RDKIT_DESCRIPTORS_EXPORT void TripleByMolecules(
    Eigen::ArrayXXi *atomIndex12Angular,
    std::pair<std::vector<int>, Eigen::ArrayXXi> *tripletInfo);

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

//! Computes pairs of atoms that are neighbors bypassing duplication to make
//! calculation faster
/*!
  \param coordinates  A matrix of size atoms * 3 containing coordinates of each
  atom
  \param species      A vector of size atoms containing mapping from atom
  index to encoding
  \param cutoff       Maximum distance within which 2 atoms
  are considered to be neighbours
  \param atomIndex12  Array in which each column represents pairs of atoms

  \return 2 dimensional array with 2 rows with each column corresponding to a
  pair of atoms which are neighbours
*/
RDKIT_DESCRIPTORS_EXPORT void NeighborPairs(Eigen::ArrayXXd *coordinates,
                                            const Eigen::VectorXi *species,
                                            double cutoff,
                                            unsigned int numAtoms,
                                            Eigen::ArrayXi *atomIndex12);

//! Calculates torchANI style symmetry functions combining both radial and
//! angular terms
/*!
  \param mol      Mol object for which symmetry functions are to be found
  \param confId   Conformer ID for the conformer for which symmetry
  functions are to be found

  \return numAtoms * 384 shaped matrix containing 384 features for every
  atom in the input mol consisting of both radial and angular terms
*/
RDKIT_DESCRIPTORS_EXPORT Eigen::ArrayXXd AtomicEnvironmentVector(
    const ROMol &mol, const std::map<std::string, Eigen::ArrayXXd> *params,
    int confId = -1);

//! Calculates torchANI style symmetry functions combining both radial and
//! angular terms
/*!
  \param pos      Array of positions of atoms
  \param species  Encoding of atom types with index
  \param numAtoms Number of Atoms

  \return numAtoms * 384 shaped matrix containing 384 features for every atom in
  the input mol consisting of both radial and angular terms
*/
RDKIT_DESCRIPTORS_EXPORT Eigen::ArrayXXd AtomicEnvironmentVector(
    double *pos, const Eigen::VectorXi &species, unsigned int numAtoms,
    const std::map<std::string, Eigen::ArrayXXd> *params);

//! Constructs a vector with values of another vector at specified indices along
//! given dimension
/*!
  \param vector1    Matrix in which values are to be stored
  \param vector2    Matrix from which values are to be taken
  \param index      Array which specifies indices of vector2
  \param dim        dimension along which indices are to be picked

  \return Matrix containing values at positions specified by index in vector2
*/
template <typename Derived>
RDKIT_DESCRIPTORS_EXPORT void IndexSelect(Eigen::ArrayBase<Derived> *vector1,
                                          Eigen::ArrayBase<Derived> *vector2,
                                          Eigen::ArrayXi &index,
                                          unsigned int dim) {
  PRECONDITION(vector1 != nullptr && vector2 != nullptr,
               "Input vectors are NULL");
  PRECONDITION(dim == 0 || dim == 1,
               "Only values 0 and 1 are accepted for dim");
  for (auto i = 0; i < index.size(); i++) {
    switch (dim) {
      case 0:
        vector1->row(i) = vector2->row(index(i));
        break;
      case 1:
        vector1->col(i) = vector2->col(index(i));
        break;
      default:
        throw ValueErrorException("Value of dim must be 0 or 1");
    }
  }
}

}  // namespace ANI
}  // namespace Descriptors
}  // namespace RDKit
#endif
#endif
#endif

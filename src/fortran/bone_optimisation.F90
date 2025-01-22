!> \file
!> \author Chris Bradley
!> \brief This is an example program to solve a bone optimisation problem using OpenCMISS calls.
!>
!> Implements the example from
!>
!> See Masaki Otomore, Takayuki Yamada, Kazuhiro Izui, and Shinji Nishiwaki, 2015, "Matlab code for
!> a level set-based topology optimization method using a reaction diffusion equation", Struct.
!> Multidisc. Optim., 51:1159-1172. DOI:10.1007/s00158-014-1190-z
!>
!> \section LICENSE
!>
!> Version: MPL 1.1/GPL 2.0/LGPL 2.1
!>
!> The contents of this file are subject to the Mozilla Public License
!> Version 1.1 (the "License"); you may not use this file except in
!> compliance with the License. You may obtain a copy of the License at
!> http://www.mozilla.org/MPL/
!>
!> Software distributed under the License is distributed on an "AS IS"
!> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
!> License for the specific language governing rights and limitations
!> under the License.
!>
!> The Original Code is OpenCMISS
!>
!> The Initial Developer of the Original Code is University of Auckland,
!> Auckland, New Zealand and University of Oxford, Oxford, United
!> Kingdom. Portions created by the University of Auckland and University
!> of Oxford are Copyright (C) 2007 by the University of Auckland and
!> the University of Oxford. All Rights Reserved.
!>
!> Contributor(s):
!>
!> Alternatively, the contents of this file may be used under the terms of
!> either the GNU General Public License Version 2 or later (the "GPL"), or
!> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
!> in which case the provisions of the GPL or the LGPL are applicable instead
!> of those above. If you wish to allow use of your version of this file only
!> under the terms of either the GPL or the LGPL, and not to allow others to
!> use your version of this file under the terms of the MPL, indicate your
!> decision by deleting the provisions above and replace them with the notice
!> and other provisions required by the GPL or the LGPL. If you do not delete
!> the provisions above, a recipient may use your version of this file under
!> the terms of any one of the MPL, the GPL or the LGPL.
!>

PROGRAM BoneOptimisation

#ifdef WITH_MPI  
#ifdef WITH_F08_MPI
  USE MPI_F08
#elif WITH_F90_MPI 
  USE MPI
#endif
#endif  
  USE OpenCMISS

  IMPLICIT NONE

  !Program parameters

  REAL(OC_RP), PARAMETER :: PI=3.141592653589793238462643383279502884197_OC_RP 
 
  !Geomery
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_DIMENSIONS = 3 !The number of dimensions
  
  REAL(OC_RP), PARAMETER :: LENGTH = 12.50_OC_RP !The length of the domain 
  REAL(OC_RP), PARAMETER :: HEIGHT = 10.00_OC_RP !The height of the domain
  REAL(OC_RP), PARAMETER :: WIDTH = 10.00_OC_RP !The height of the domain
  
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_X_ELEMENTS = 10 !Number of elements along the length of the domain
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_Y_ELEMENTS = 6 !Number of elements along the height of the domain
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_Z_ELEMENTS = 6 !Number of elements along the width of the domain

  !Loading case
  INTEGER(OC_Intg), PARAMETER :: CANTILEVER_LOADING_CASE = 1
  INTEGER(OC_Intg), PARAMETER :: SIMPLY_SUPPORTED_LOADING_CASE = 2
  
  INTEGER(OC_Intg), PARAMETER :: LOADING_CASE = CANTILEVER_LOADING_CASE
  !INTEGER(OC_Intg), PARAMETER :: LOADING_CASE = SIMPLY_SUPPORTED_LOADING_CASE
  
  !Loading parametes
  REAL(OC_RP), PARAMETER :: MAX_FORCE = 0.6666_OC_RP

  !Elasticity parameters
  REAL(OC_RP), PARAMETER :: YOUNGS_MODULUS = 1.0_OC_RP
  REAL(OC_RP), PARAMETER :: YOUNGS_MODULUS_MIN = 0.000001_OC_RP
  REAL(OC_RP), PARAMETER :: POISSONS_RATIO = 0.3
  REAL(OC_RP), PARAMETER :: THICKNESS = 1.0

  !Optimisation prarameters
  REAL(OC_RP), PARAMETER :: DIFFUSION_A_PARAM = 1.0_OC_RP
  REAL(OC_RP), PARAMETER :: DIFFUSION_TAU_PARAM = 0.001_OC_RP

  REAL(OC_RP), PARAMETER :: MAX_VOLUME_RATIO = 0.50_OC_RP
  REAL(OC_RP), PARAMETER :: LEVEL_SET_P_PARAM = 4.0_OC_RP
  REAL(OC_RP), PARAMETER :: LEVEL_SET_D_PARAM = -0.02_OC_RP
  
  INTEGER(OC_Intg), PARAMETER :: N_VOL_ITERATIONS = 100_OC_Intg

  !INTEGER(OC_Intg), PARAMETER :: MAXIMUM_NUMBER_OF_ITERATIONS = 5_OC_Intg
  INTEGER(OC_Intg), PARAMETER :: MAXIMUM_NUMBER_OF_ITERATIONS = 200_OC_Intg

  REAL(OC_RP), PARAMETER :: TIME_START = 0.0_OC_RP
  REAL(OC_RP), PARAMETER :: TIME_STEP = 0.05_OC_RP

  !Integration parameters
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_GAUSS_XI = 3

  !OpenCMISS objects  
  
  TYPE(OC_BasisType) :: basis,elementBasis
  TYPE(OC_BoundaryConditionsType) :: elasticityBoundaryConditions,diffusionBoundaryConditions
  TYPE(OC_ComputationEnvironmentType) :: computationEnvironment
  TYPE(OC_ContextType) :: context
  TYPE(OC_ControlLoopType) :: diffusionControlLoop
  TYPE(OC_CoordinateSystemType) :: coordinateSystem,worldCoordinateSystem
  TYPE(OC_DecomposerType) :: decomposer
  TYPE(OC_DecompositionType) :: decomposition
  TYPE(OC_EquationsType) :: elasticityEquations,diffusionEquations
  TYPE(OC_EquationsSetType) :: diffusionEquationsSet,elasticityEquationsSet
  TYPE(OC_FieldType) :: diffusionDependentField,diffusionEquationsSetField,diffusionMaterialsField,diffusionSourceField, &
    & diffusionStructureField,geometricField,elasticityEquationsSetField,elasticityDependentField,elasticityMaterialsField, &
    & elasticityDerivedField,sedField,structureField,tdField
  TYPE(OC_FieldsType) :: fields
  TYPE(OC_GeneratedMeshType) :: generatedMesh
  TYPE(OC_MeshType) :: mesh
  TYPE(OC_NodesType) :: nodes
  TYPE(OC_ProblemType) :: diffusionProblem,elasticityProblem
  TYPE(OC_RegionType) :: region,worldRegion
  TYPE(OC_SolverType) :: diffusionSolver,diffusionLinearSolver,elasticitySolver,linearSolver
  TYPE(OC_SolverEquationsType) :: diffusionSolverEquations,elasticitySolverEquations
  TYPE(OC_WorkGroupType) :: worldWorkGroup
  
  !Derived parameters
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_X_NODES = NUMBER_OF_X_ELEMENTS+1
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_Y_NODES = NUMBER_OF_Y_ELEMENTS+1
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_Z_NODES = NUMBER_OF_Z_ELEMENTS+1
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_ELEMENTS = NUMBER_OF_X_ELEMENTS*NUMBER_OF_Y_ELEMENTS* &
    & MAX(1,NUMBER_OF_Z_ELEMENTS*(NUMBER_OF_DIMENSIONS-2))
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_NODES = (NUMBER_OF_X_ELEMENTS+1)*(NUMBER_OF_Y_ELEMENTS+1)* &
    & MAX(1,(NUMBER_OF_Z_ELEMENTS+1)*(NUMBER_OF_DIMENSIONS-2))
  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_XI = NUMBER_OF_DIMENSIONS

  INTEGER(OC_Intg), PARAMETER :: NUMBER_OF_DOFS = NUMBER_OF_NODES*NUMBER_OF_DIMENSIONS

  REAL(OC_RP), PARAMETER :: LAME_LAMBDA = POISSONS_RATIO*YOUNGS_MODULUS/ &
    &((1.0_OC_RP-POISSONS_RATIO)*(1.0_OC_RP-2.0_OC_RP*POISSONS_RATIO))
  REAL(OC_RP), PARAMETER :: LAME_MU = YOUNGS_MODULUS/(2.0_OC_RP*(1.0_OC_RP-POISSONS_RATIO))
  REAL(OC_RP), PARAMETER :: LAME_LAMBDA_MIN = POISSONS_RATIO*YOUNGS_MODULUS_MIN/ &
    &((1.0_OC_RP-POISSONS_RATIO)*(1.0_OC_RP-2.0_OC_RP*POISSONS_RATIO))
  REAL(OC_RP), PARAMETER :: LAME_MU_MIN = YOUNGS_MODULUS_MIN/(2.0_OC_RP*(1.0_OC_RP-POISSONS_RATIO))
  
  !Generic OpenCMISS variables
  INTEGER(OC_Intg), PARAMETER :: CONTEXT_USER_NUMBER=1
  INTEGER(OC_Intg), PARAMETER :: COORDINATE_SYSTEM_USER_NUMBER=2
  INTEGER(OC_Intg), PARAMETER :: REGION_USER_NUMBER=3
  INTEGER(OC_Intg), PARAMETER :: BASIS_USER_NUMBER=4
  INTEGER(OC_Intg), PARAMETER :: GENERATED_MESH_USER_NUMBER=5
  INTEGER(OC_Intg), PARAMETER :: MESH_USER_NUMBER=6
  INTEGER(OC_Intg), PARAMETER :: DECOMPOSITION_USER_NUMBER=7
  INTEGER(OC_Intg), PARAMETER :: DECOMPOSER_USER_NUMBER=8
  INTEGER(OC_Intg), PARAMETER :: GEOMETRIC_FIELD_USER_NUMBER=9
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_EQUATIONS_SET_USER_NUMBER=10
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER=11
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_DEPENDENT_FIELD_USER_NUMBER=12
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_MATERIALS_FIELD_USER_NUMBER=13
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_DERIVED_FIELD_USER_NUMBER=14
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_EQUATIONS_SET_USER_NUMBER=15
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_EQUATIONS_SET_FIELD_USER_NUMBER=16
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_DEPENDENT_FIELD_USER_NUMBER=17
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_MATERIALS_FIELD_USER_NUMBER=18
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_SOURCE_FIELD_USER_NUMBER=19
  INTEGER(OC_Intg), PARAMETER :: STRUCTURE_FIELD_USER_NUMBER=20
  INTEGER(OC_Intg), PARAMETER :: SED_FIELD_USER_NUMBER=21
  INTEGER(OC_Intg), PARAMETER :: TD_FIELD_USER_NUMBER=22
  INTEGER(OC_Intg), PARAMETER :: ELASTICITY_PROBLEM_USER_NUMBER=23
  INTEGER(OC_Intg), PARAMETER :: DIFFUSION_PROBLEM_USER_NUMBER=24
  
  INTEGER(OC_Intg) :: numberOfComputationalNodes,computationalNodeNumber
  INTEGER(OC_Intg) :: err,mpiIError

  !Other variables
  INTEGER(OC_Intg) :: componentIdx,elementDomain,elementIdx,elementNumber,iterationIdx,localNodeIdx,nodeDomain,nodeIdx, &
    & nodeNumber,numberOfElementNodes,numberOfLocalElements,numberOfLocalNodes,numberOfSurroundingElements, &
    & numberOfVoigtComponents,surroundingElement,surroundingElementIdx,voigt11Component,voigt12Component,voigt13Component, &
    & voigt22Component,voigt23Component,voigt33Component,xElementIdx,xNodeIdx,xStep,yElementIdx,yNodeIdx,zNodeIdx
  INTEGER(OC_Intg) :: elasticityEquationsSetSpecification(3),elasticityProblemSpecification(3), &
    & diffusionEquationsSetSpecification(3),diffusionProblemSpecification(3)
  INTEGER(OC_Intg) :: decompositionIndex,elasticitySolverEquationsSetIndex,diffusionSolverEquationsSetIndex
  INTEGER(OC_Intg) :: bottomLeftNodeNumber,bottomRightNodeNumber,midNodeNumber,strValue,topLeftNodeNumber,topRightNodeNumber
  INTEGER(OC_Intg), POINTER :: structureValues(:)
  LOGICAL :: onBoundary
  REAL(OC_RP) :: A(6,6),A1,A2,absTDSum,averagePhi,averageTD,bottomLeftPhi,bottomRightPhi,C,C1,C2, &
    & diffusionSource,e(6,1),e11,e22,e33,e12,e13,e23,eTA(1,6),eTAe(1,1),eTKE(1,6),eTKEe(1,1),energy,eT(1,6),G, &
    & initialVolumeRatio,KE(6,6),lambdaValue,maximumG,nodalPhi,objective(MAXIMUM_NUMBER_OF_ITERATIONS),objectiveSum, &
    & rankAbsTDSum,rankObjectiveSum,rankStrSum,rankSEDSum,rankTDSum,strainEnergy,strainEnergyDensity,strSum, &
    & sedSum,time,topLeftPhi,topRightPhi,topologicalDerivative,tdSum,volumeRatio
  REAL(OC_RP), POINTER :: diffusionSourceValues(:),elasticityValues(:),phiValues(:),sedValues(:),strainValues(:), &
    & stressValues(:),strainEnergyValues(:),tdValues(:),tdnValues(:),ymValues(:)
  CHARACTER(LEN=30) :: filename,iterationString

  !Program

  !Intialise OpenCMISS
  CALL OC_Initialise(err) 
  !CALL OC_ErrorHandlingModeSet(OC_ERRORS_TRAP_ERROR,Err)
  CALL OC_ErrorHandlingModeSet(OC_ERRORS_OUTPUT_ERROR,Err)
  !CALL OC_DiagnosticsSetOn(OC_ALL_DIAG_TYPE,[1,2,3,4,5],"Diagnostics",["FiniteElasticity_GrowthTensorCalculate"],err)
  WRITE(filename,'(A,"_",I0,"x",I0,"x",I0,"x",I0,"_",I0)') "BoneOptimisation",NUMBER_OF_x_ELEMENTS,NUMBER_OF_Y_ELEMENTS, &
    & NUMBER_OF_Z_ELEMENTS*(NUMBER_OF_DIMENSIONS-2),LOADING_CASE
  CALL OC_OutputSetOn(filename,err)

  CALL OC_NumberOfVoigtComponentsGet(NUMBER_OF_DIMENSIONS,numberOfVoigtComponents,err)
  CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,1,voigt11Component,err)
  CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,2,voigt12Component,err)
  CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,2,voigt22Component,err)
  IF(NUMBER_OF_DIMENSIONS==3) THEN
    CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,3,voigt13Component,err)
    CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,3,voigt23Component,err)
    CALL OC_TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,3,3,voigt33Component,err)
  ELSE
    voigt13Component=0
    voigt23Component=0
    voigt33Component=0
  ENDIF
   
  !-----------------------------------------------------------------------------------------------------------
  ! CONTEXT
  !-----------------------------------------------------------------------------------------------------------

  !Create a context
  CALL OC_Context_Initialise(context,err)
  CALL OC_Context_Create(CONTEXT_USER_NUMBER,context,err)
  CALL OC_Region_Initialise(worldRegion,err)
  CALL OC_Context_WorldRegionGet(context,worldRegion,err)
  CALL OC_Context_RandomSeedsSet(context,9999,err)
     
  !-----------------------------------------------------------------------------------------------------------
  ! COMPUTATIONAL ENVIRONMENT
  !-----------------------------------------------------------------------------------------------------------

  !Get the computational nodes information
  CALL OC_ComputationEnvironment_Initialise(computationEnvironment,err)
  CALL OC_Context_ComputationEnvironmentGet(context,computationEnvironment,err)
  
  CALL OC_WorkGroup_Initialise(worldWorkGroup,err)
  CALL OC_ComputationEnvironment_WorldWorkGroupGet(computationEnvironment,worldWorkGroup,err)
  CALL OC_WorkGroup_NumberOfGroupNodesGet(worldWorkGroup,numberOfComputationalNodes,err)
  CALL OC_WorkGroup_GroupNodeNumberGet(worldWorkGroup,computationalNodeNumber,err)
    
  !-----------------------------------------------------------------------------------------------------------
  ! COORDINATE SYSTEM
  !-----------------------------------------------------------------------------------------------------------

  !Start the creation of a new RC coordinate system
  CALL OC_CoordinateSystem_Initialise(coordinateSystem,err)
  CALL OC_CoordinateSystem_CreateStart(COORDINATE_SYSTEM_USER_NUMBER,context,coordinateSystem,err)
  !Set the coordinate system to be 2/3D
  CALL OC_CoordinateSystem_DimensionSet(coordinateSystem,NUMBER_OF_DIMENSIONS,err)
  !Finish the creation of the coordinate system
  CALL OC_CoordinateSystem_CreateFinish(coordinateSystem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! REGION
  !-----------------------------------------------------------------------------------------------------------

  !Start the creation of the region
  CALL OC_Region_Initialise(region,err)
  CALL OC_Region_CreateStart(REGION_USER_NUMBER,worldRegion,region,err)
  CALL OC_Region_LabelSet(region,"BoneOptimisation",err)
  !Set the regions coordinate system to the 2/3D RC coordinate system that we have created
  CALL OC_Region_CoordinateSystemSet(region,coordinateSystem,err)
  !Finish the creation of the region
  CALL OC_Region_CreateFinish(region,err)

  !-----------------------------------------------------------------------------------------------------------
  ! BASIS
  !-----------------------------------------------------------------------------------------------------------

  !Start the creation of a trilinear Lagrange basis
  CALL OC_Basis_Initialise(basis,err)
  CALL OC_Basis_CreateStart(BASIS_USER_NUMBER,context,basis,err)
  CALL OC_Basis_TypeSet(basis,OC_BASIS_LAGRANGE_HERMITE_TP_TYPE,err)
  !Set the basis to be a trilinear Lagrange interpolation basis
  CALL OC_Basis_NumberOfXiSet(basis,NUMBER_OF_XI,err)
  IF(NUMBER_OF_XI==2) THEN
    CALL OC_Basis_InterpolationXiSet(basis,[OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION,OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION],err)
    CALL OC_Basis_QuadratureNumberOfGaussXiSet(basis,[NUMBER_OF_GAUSS_XI,NUMBER_OF_GAUSS_XI],err)
  ELSE
    CALL OC_Basis_InterpolationXiSet(basis,[OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION,OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION, &
      & OC_BASIS_LINEAR_LAGRANGE_INTERPOLATION],err)     
    CALL OC_Basis_QuadratureNumberOfGaussXiSet(basis,[NUMBER_OF_GAUSS_XI,NUMBER_OF_GAUSS_XI,NUMBER_OF_GAUSS_XI],err)
  ENDIF
  !Finish the creation of the basis
  CALL OC_Basis_CreateFinish(basis,err)
    
  !-----------------------------------------------------------------------------------------------------------
  ! MESH
  !-----------------------------------------------------------------------------------------------------------

  !Start the creation of a generated mesh in the region
  CALL OC_GeneratedMesh_Initialise(generatedMesh,err)
  CALL OC_GeneratedMesh_CreateStart(GENERATED_MESH_USER_NUMBER,region,generatedMesh,err)
  !Set up a regular x*y*z mesh
  CALL OC_GeneratedMesh_TypeSet(generatedMesh,OC_GENERATED_MESH_REGULAR_MESH_TYPE,err)
  !Set the default basis
  CALL OC_GeneratedMesh_BasisSet(generatedMesh,basis,err)
  !Define the mesh on the region
  IF(NUMBER_OF_DIMENSIONS==2) THEN    
    CALL OC_GeneratedMesh_ExtentSet(generatedMesh,[LENGTH,HEIGHT],err)
    CALL OC_GeneratedMesh_NumberOfElementsSet(generatedMesh,[NUMBER_OF_X_ELEMENTS,NUMBER_OF_Y_ELEMENTS],err)
  ELSE
    CALL OC_GeneratedMesh_ExtentSet(generatedMesh,[LENGTH,HEIGHT,WIDTH],err)
    CALL OC_GeneratedMesh_NumberOfElementsSet(generatedMesh,[NUMBER_OF_X_ELEMENTS,NUMBER_OF_Y_ELEMENTS, &
      & NUMBER_OF_Z_ELEMENTS],err)
  ENDIF
  !Finish the creation of a generated mesh in the region
  CALL OC_Mesh_Initialise(mesh,err)
  CALL OC_GeneratedMesh_CreateFinish(generatedMesh,MESH_USER_NUMBER,mesh,err)

  !-----------------------------------------------------------------------------------------------------------
  ! MESH DECOMPOSITION
  !-----------------------------------------------------------------------------------------------------------
  
  !Create a decomposition
  CALL OC_Decomposition_Initialise(decomposition,err)
  CALL OC_Decomposition_CreateStart(DECOMPOSITION_USER_NUMBER,mesh,decomposition,err)
  !Set the decomposition to be a general decomposition with the specified number of domains
  CALL OC_Decomposition_TypeSet(decomposition,OC_DECOMPOSITION_CALCULATED_TYPE,err)
  !Finish the decomposition
  CALL OC_Decomposition_CreateFinish(decomposition,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DECOMPOSER
  !-----------------------------------------------------------------------------------------------------------

  !Decompose
  CALL OC_Decomposer_Initialise(decomposer,err)
  CALL OC_Decomposer_CreateStart(DECOMPOSER_USER_NUMBER,region,worldWorkGroup,decomposer,err)
  !Add in the decomposition
  CALL OC_Decomposer_DecompositionAdd(decomposer,decomposition,decompositionIndex,err)
  !Finish the decomposer
  CALL OC_Decomposer_CreateFinish(decomposer,err)  

  CALL OC_Decomposition_NumberOfLocalElementsGet(decomposition,numberOfLocalElements,err)
  CALL OC_Decomposition_NumberOfLocalNodesGet(decomposition,1,numberOfLocalNodes,err)  

  !-----------------------------------------------------------------------------------------------------------  
  ! GEOMETRIC FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Start to create a default (geometric) field on the region
  CALL OC_Field_Initialise(geometricField,err)
  CALL OC_Field_CreateStart(GEOMETRIC_FIELD_USER_NUMBER,region,geometricField,err)
  !Set the decomposition to use
  CALL OC_Field_DecompositionSet(geometricField,decomposition,err)
  !Set the field variable label
  CALL OC_Field_VariableLabelSet(geometricField,OC_FIELD_U_VARIABLE_TYPE,"Geometry",err)
  !Set the domain to be used by the field components.
  CALL OC_Field_ComponentMeshComponentSet(geometricField,OC_FIELD_U_VARIABLE_TYPE,1,1,err)
  CALL OC_Field_ComponentMeshComponentSet(geometricField,OC_FIELD_U_VARIABLE_TYPE,2,1,err)
  IF(NUMBER_OF_DIMENSIONS==3) THEN
    CALL OC_Field_ComponentMeshComponentSet(geometricField,OC_FIELD_U_VARIABLE_TYPE,3,1,err)
  ENDIF
  !Set the scaling type
  CALL OC_Field_ScalingTypeSet(geometricField,OC_FIELD_ARITHMETIC_MEAN_SCALING,err)
  !Finish creating the field
  CALL OC_Field_CreateFinish(geometricField,err)
  
  !Update the geometric field parameters
  CALL OC_GeneratedMesh_GeometricParametersCalculate(generatedMesh,geometricField,err)
   
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY EQUATIONS SET
  !-----------------------------------------------------------------------------------------------------------

  !Create the elasticity based finite elasticity equations set
  CALL OC_EquationsSet_Initialise(elasticityEquationsSet,err)
  CALL OC_Field_Initialise(elasticityEquationsSetField,err)
  IF(NUMBER_OF_DIMENSIONS==2) THEN
    elasticityEquationsSetSpecification = [OC_EQUATIONS_SET_ELASTICITY_CLASS,OC_EQUATIONS_SET_LINEAR_ELASTICITY_TYPE, &
      & OC_EQUATIONS_SET_TWO_DIMENSIONAL_PLANE_STRESS_SUBTYPE]
  ELSE
    elasticityEquationsSetSpecification = [OC_EQUATIONS_SET_ELASTICITY_CLASS,OC_EQUATIONS_SET_LINEAR_ELASTICITY_TYPE, &
      & OC_EQUATIONS_SET_THREE_DIMENSIONAL_ISOTROPIC_SUBTYPE]
  ENDIF
  CALL OC_EquationsSet_CreateStart(ELASTICITY_EQUATIONS_SET_USER_NUMBER,region,geometricField, &
    & elasticityEquationsSetSpecification,ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,elasticityEquationsSetField, &
    & elasticityEquationsSet,err)
  !Finish creating the equations set
  CALL OC_EquationsSet_CreateFinish(elasticityEquationsSet,err)

  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY DEPENDENT FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the dependent field on the region
  CALL OC_Field_Initialise(elasticityDependentField,err)  
  !Set up the equations set dependent field    
  CALL OC_EquationsSet_DependentCreateStart(elasticityEquationsSet,ELASTICITY_DEPENDENT_FIELD_USER_NUMBER, &
    & elasticityDependentField,err)
  !Set the field label
  CALL OC_Field_LabelSet(elasticityDependentField,"ElasticityDependent",err)
  !Set the field variable labels
  CALL OC_Field_VariableLabelSet(elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE,"Dependent",err)
  CALL OC_Field_VariableLabelSet(elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE,"Traction",err)
  CALL OC_EquationsSet_DependentCreateFinish(elasticityEquationsSet,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY MATERIALS FIELD
  !-----------------------------------------------------------------------------------------------------------
  
  !Set up the equations set materials field    
  CALL OC_Field_Initialise(elasticityMaterialsField,err) 
  CALL OC_EquationsSet_MaterialsCreateStart(elasticityEquationsSet,ELASTICITY_MATERIALS_FIELD_USER_NUMBER, &
    & elasticityMaterialsField,err)
  !Set the field label
  CALL OC_Field_LabelSet(elasticityMaterialsField,"ElasticityMaterials",err)
  !Set the field variable labels
  CALL OC_Field_VariableLabelSet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,"ElasticityMaterials",err)
  !Set the interpolation types
  CALL OC_Field_ComponentInterpolationSet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,1, &
    & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  IF(NUMBER_OF_DIMENSIONS==2) THEN !Thickness component for plane stress
    CALL OC_Field_ComponentInterpolationSet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,2, &
      & OC_FIELD_CONSTANT_INTERPOLATION,err)
    CALL OC_Field_ComponentInterpolationSet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,3, &
      & OC_FIELD_CONSTANT_INTERPOLATION,err)
  ELSE
    CALL OC_Field_ComponentInterpolationSet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,2, &
      & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  ENDIF
  !Finish the field creation
  CALL OC_EquationsSet_MaterialsCreateFinish(elasticityEquationsSet,err)
  
  !Initialise the material constants
  IF(NUMBER_OF_DIMENSIONS==2) THEN
    CALL OC_Field_ComponentValuesInitialise(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE, &
      & OC_FIELD_VALUES_SET_TYPE,1,YOUNGS_MODULUS,err)
    CALL OC_Field_ComponentValuesInitialise(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE, &
      & OC_FIELD_VALUES_SET_TYPE,2,POISSONS_RATIO,err)
    CALL OC_Field_ComponentValuesInitialise(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE, &
      & OC_FIELD_VALUES_SET_TYPE,3,THICKNESS,err)
  ELSE
    CALL OC_Field_ComponentValuesInitialise(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE, &
      & OC_FIELD_VALUES_SET_TYPE,1,LAME_LAMBDA,err)
    CALL OC_Field_ComponentValuesInitialise(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE, &
      & OC_FIELD_VALUES_SET_TYPE,2,LAME_MU,err)
  ENDIF
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY DERIVED FIELD
  !-----------------------------------------------------------------------------------------------------------
  
  !Set up the equations set derived field    
  CALL OC_Field_Initialise(elasticityDerivedField,err)
  CALL OC_Field_CreateStart(ELASTICITY_DERIVED_FIELD_USER_NUMBER,region,elasticityDerivedField,err)
  !Set the field label
  CALL OC_Field_LabelSet(elasticityDerivedField,"ElasticityDerived",err)
  CALL OC_Field_TypeSet(elasticityDerivedField,OC_FIELD_GENERAL_TYPE,err)
  CALL OC_Field_DecompositionSet(elasticityDerivedField,decomposition,err)
  CALL OC_Field_GeometricFieldSet(elasticityDerivedField,geometricField,err)
  CALL OC_Field_DependentTypeSet(elasticityDerivedField,OC_FIELD_DEPENDENT_TYPE,err)
  CALL OC_Field_NumberOfVariablesSet(elasticityDerivedField,3,err)
  CALL OC_Field_VariableTypesSet(elasticityDerivedField,[OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_V_VARIABLE_TYPE, &
    & OC_FIELD_W_VARIABLE_TYPE],err)
  CALL OC_Field_VariableLabelSet(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,"SmallStrain",err)
  CALL OC_Field_VariableLabelSet(elasticityDerivedField,OC_FIELD_V_VARIABLE_TYPE,"CauchyStress",err)
  CALL OC_Field_VariableLabelSet(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,"ElasticWork",err)
  CALL OC_Field_NumberOfComponentsSet(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,numberOfVoigtComponents,err)
  CALL OC_Field_NumberOfComponentsSet(elasticityDerivedField,OC_FIELD_V_VARIABLE_TYPE,numberOfVoigtComponents,err)
  CALL OC_Field_NumberOfComponentsSet(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,1,err)
  DO componentIdx=1,numberOfVoigtComponents
    CALL OC_Field_ComponentMeshComponentSet(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,componentIdx,1,err)
    CALL OC_Field_ComponentMeshComponentSet(elasticityDerivedField,OC_FIELD_V_VARIABLE_TYPE,componentIdx,1,err)
    CALL OC_Field_ComponentInterpolationSet(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,componentIdx, &
      & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
    CALL OC_Field_ComponentInterpolationSet(elasticityDerivedField,OC_FIELD_V_VARIABLE_TYPE,componentIdx, &
      & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  ENDDO !componentIdx
  CALL OC_Field_ComponentMeshComponentSet(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,1,1,err)
  CALL OC_Field_ComponentInterpolationSet(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,1, &
    & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  CALL OC_Field_CreateFinish(elasticityDerivedField,err)
    
  CALL OC_EquationsSet_DerivedCreateStart(elasticityEquationsSet,ELASTICITY_DERIVED_FIELD_USER_NUMBER, &
    & elasticityDerivedField,err)
  CALL OC_EquationsSet_DerivedVariableSet(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_SMALL_STRAIN, &
    & OC_FIELD_U_VARIABLE_TYPE,err)
  CALL OC_EquationsSet_DerivedVariableSet(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_CAUCHY_STRESS, &
    & OC_FIELD_V_VARIABLE_TYPE,err)
  CALL OC_EquationsSet_DerivedVariableSet(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_ELASTIC_WORK, &
    & OC_FIELD_W_VARIABLE_TYPE,err)
  CALL OC_EquationsSet_DerivedCreateFinish(elasticityEquationsSet,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY EQUATIONS
  !-----------------------------------------------------------------------------------------------------------

  !Set up the equations set equations
  CALL OC_Equations_Initialise(elasticityEquations,err)
  CALL OC_EquationsSet_EquationsCreateStart(elasticityEquationsSet,elasticityEquations,err)
  CALL OC_Equations_SparsityTypeSet(elasticityEquations,OC_EQUATIONS_SPARSE_MATRICES,err)
  CALL OC_Equations_OutputTypeSet(elasticityEquations,OC_EQUATIONS_NO_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(elasticityEquations,OC_EQUATIONS_TIMING_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(elasticityEquations,OC_EQUATIONS_MATRIX_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(elasticityEquations,OC_EQUATIONS_ELEMENT_MATRIX_OUTPUT,err)
  CALL OC_EquationsSet_EquationsCreateFinish(elasticityEquationsSet,err) 
  
  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION EQUATIONS SET
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion equations set
  diffusionEquationsSetSpecification=[OC_EQUATIONS_SET_CLASSICAL_FIELD_CLASS,OC_EQUATIONS_SET_DIFFUSION_EQUATION_TYPE, &
    & OC_EQUATIONS_SET_GENERALISED_DIFFUSION_SUBTYPE]
  
  CALL OC_Field_Initialise(diffusionEquationsSetField,err)
  CALL OC_EquationsSet_Initialise(diffusionEquationsSet,err)
  CALL OC_EquationsSet_CreateStart(DIFFUSION_EQUATIONS_SET_USER_NUMBER,region,geometricField,diffusionEquationsSetSpecification, &
    & DIFFUSION_EQUATIONS_SET_FIELD_USER_NUMBER,diffusionEquationsSetField,diffusionEquationsSet,err)
  CALL OC_EquationsSet_CreateFinish(diffusionEquationsSet,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION DEPENDENT FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion dependent field
  CALL OC_Field_Initialise(diffusionDependentField,err)
  CALL OC_EquationsSet_DependentCreateStart(diffusionEquationsSet,DIFFUSION_DEPENDENT_FIELD_USER_NUMBER, &
    & diffusionDependentField,err)
  CALL OC_Field_LabelSet(diffusionDependentField,"DiffusionDependent",err)
  CALL OC_Field_VariableLabelSet(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,"Phi",err)
  CALL OC_Field_VariableLabelSet(diffusionDependentField,OC_FIELD_DELUDELN_VARIABLE_TYPE,"DelPhiDelN",err)
  CALL OC_EquationsSet_DependentCreateFinish(diffusionEquationsSet,err)

  !Initialise the diffusion Phi to 1.0 
  CALL OC_Field_ComponentValuesInitialise(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1,1.0_OC_RP,err)
  
  NULLIFY(phiValues)
  CALL OC_Field_ParameterSetDataGet(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,phiValues,err)
  
  !CALL PrintArrayNodeRP(phiValues,1,"Phi")
         
  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION MATERIALS FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion materials field
  CALL OC_Field_Initialise(diffusionMaterialsField,err)
  CALL OC_EquationsSet_MaterialsCreateStart(diffusionEquationsSet,DIFFUSION_MATERIALS_FIELD_USER_NUMBER, &
    & diffusionMaterialsField,err)
  CALL OC_Field_LabelSet(diffusionMaterialsField,"DiffusionMaterials",err)
  CALL OC_Field_VariableLabelSet(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,"DiffusionMaterials",err)
  CALL OC_EquationsSet_MaterialsCreateFinish(diffusionEquationsSet,err)
  
  !Initialise the diffusion material parameters, a.del u/del t + div(sigma.grad u) + s = 0
  CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1,DIFFUSION_A_PARAM,err)
  CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1+voigt11Component,-DIFFUSION_TAU_PARAM,err)
  CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1+voigt22Component,-DIFFUSION_TAU_PARAM,err)
  CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1+voigt12Component,0.0_OC_RP,err)
  IF(NUMBER_OF_DIMENSIONS==3) THEN
    CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
      & 1+voigt33Component,-DIFFUSION_TAU_PARAM,err)
    CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
      & 1+voigt13Component,0.0_OC_RP,err)
    CALL OC_Field_ComponentValuesInitialise(diffusionMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
      & 1+voigt23Component,0.0_OC_RP,err)
  ENDIF

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION SOURCE FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion source field
  CALL OC_Field_Initialise(diffusionSourceField,err)
  CALL OC_EquationsSet_SourceCreateStart(diffusionEquationsSet,DIFFUSION_SOURCE_FIELD_USER_NUMBER, &
    & diffusionSourceField,err)
  CALL OC_Field_LabelSet(diffusionSourceField,"DiffusionSource",err)
  CALL OC_Field_VariableLabelSet(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,"DiffusionSource",err)
  !Set the field to be node based
  CALL OC_Field_ComponentInterpolationSet(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,1, &
    & OC_FIELD_NODE_BASED_INTERPOLATION,err)
  CALL OC_EquationsSet_SourceCreateFinish(diffusionEquationsSet,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION EQUATIONS
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion equations
  CALL OC_Equations_Initialise(diffusionEquations,err)
  CALL OC_EquationsSet_EquationsCreateStart(diffusionEquationsSet,diffusionEquations,err)
  CALL OC_Equations_SparsityTypeSet(diffusionEquations,OC_EQUATIONS_SPARSE_MATRICES,err)
  CALL OC_Equations_OutputTypeSet(diffusionEquations,OC_EQUATIONS_NO_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(diffusionEquations,OC_EQUATIONS_TIMING_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(diffusionEquations,OC_EQUATIONS_MATRIX_OUTPUT,err)
  !CALL OC_Equations_OutputTypeSet(diffusionEquations,OC_EQUATIONS_ELEMENT_MATRIX_OUTPUT,err)
  CALL OC_EquationsSet_EquationsCreateFinish(diffusionEquationsSet,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY PROBLEM
  !-----------------------------------------------------------------------------------------------------------

  !Define the problem
  CALL OC_Problem_Initialise(elasticityProblem,err)
  elasticityProblemSpecification = [OC_PROBLEM_ELASTICITY_CLASS,OC_PROBLEM_LINEAR_ELASTICITY_TYPE, &
    & OC_PROBLEM_NO_SUBTYPE]
  CALL OC_Problem_CreateStart(ELASTICITY_PROBLEM_USER_NUMBER,context,elasticityProblemSpecification,elasticityProblem,err)
  CALL OC_Problem_CreateFinish(elasticityProblem,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY CONTROL LOOPS
  !-----------------------------------------------------------------------------------------------------------

  !Create control loops
  CALL OC_ControlLoop_Initialise(diffusionControlLoop,err)
  CALL OC_Problem_ControlLoopCreateStart(elasticityProblem,err)
  CALL OC_Problem_ControlLoopCreateFinish(elasticityProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY SOLVERS
  !-----------------------------------------------------------------------------------------------------------

  !Create problem solvers
  CALL OC_Solver_Initialise(elasticitySolver,err)
  CALL OC_Problem_SolversCreateStart(elasticityProblem,err)
  CALL OC_Problem_SolverGet(elasticityProblem,OC_CONTROL_LOOP_NODE,1,elasticitySolver,err)
  CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_NO_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_MONITOR_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_PROGRESS_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_TIMING_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_SOLVER_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(elasticitySolver,OC_SOLVER_MATRIX_OUTPUT,err)
  CALL OC_Solver_LinearTypeSet(elasticitySolver,OC_SOLVER_LINEAR_DIRECT_SOLVE_TYPE,err)
  !CALL OC_Solver_LinearTypeSet(elasticitySolver,OC_SOLVER_LINEAR_ITERATIVE_SOLVE_TYPE,err)
  !CALL OC_Solver_LinearIterativeMaximumIterationsSet(elasticitySolver,1000000,err)
  !CALL OC_Solver_LinearIterativeGMRESRestartSet(elasticitySolver,NUMBER_OF_NODES,err)
  CALL OC_Problem_SolversCreateFinish(elasticityProblem,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY SOLVER EQUATIONS
  !-----------------------------------------------------------------------------------------------------------

  !Create nonlinear equations and add finite elasticity equations set to solver equations
  CALL OC_SolverEquations_Initialise(elasticitySolverEquations,err)
  CALL OC_Problem_SolverEquationsCreateStart(elasticityProblem,err)
  !Get the solver equations
  CALL OC_Solver_SolverEquationsGet(elasticitySolver,elasticitySolverEquations,Err)
  !Set the sparsity type
  !CALL OC_SolverEquations_SparsityTypeSet(elasticitySolverEquations,OC_SOLVER_FULL_MATRICES,err)
  CALL OC_SolverEquations_SparsityTypeSet(elasticitySolverEquations,OC_SOLVER_SPARSE_MATRICES,err)
  !Add in the elasticity equations set
  CALL OC_SolverEquations_EquationsSetAdd(elasticitySolverEquations,elasticityEquationsSet,elasticitySolverEquationsSetIndex,err)
  CALL OC_Problem_SolverEquationsCreateFinish(elasticityProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY BOUNDARY CONDITIONS
  !-----------------------------------------------------------------------------------------------------------

  !Prescribe boundary conditions (absolute nodal parameters)
  CALL OC_BoundaryConditions_Initialise(elasticityBoundaryConditions,err)
  CALL OC_SolverEquations_BoundaryConditionsCreateStart(elasticitySolverEquations,elasticityBoundaryConditions,err)

  SELECT CASE(LOADING_CASE)
  CASE(CANTILEVER_LOADING_CASE)
  
    IF(NUMBER_OF_DIMENSIONS==2) THEN
      !Set the left edge to be built in
      DO yNodeIdx = 1,NUMBER_OF_Y_NODES
        nodeNumber = 1+(yNodeIdx-1)*NUMBER_OF_X_NODES
        CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
        IF(nodeDomain==computationalNodeNumber) THEN
          !Fix the node in the x and y directions
          WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
          CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
            & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
          CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
            & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        ENDIF
      ENDDO !yNodeIdx
      
      !Set the mid right edge node to have a rightward displacement/force
      midNodeNumber = (1+FLOOR(NUMBER_OF_Y_NODES/2.0_OC_RP))*NUMBER_OF_X_NODES
      
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,midNodeNumber,nodeDomain,err)
      IF(nodeDomain == computationalNodeNumber) THEN
        !Downward force at the node
        WRITE(*,'("Setting a downward force boundary condition for node ",I0)') midNodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,-MAX_FORCE,err)
      ENDIF
      
    ELSE
      
      !Set the left face to be built in
      DO zNodeIdx = 1,NUMBER_OF_Z_NODES
        DO yNodeIdx = 1,NUMBER_OF_Y_NODES
          nodeNumber = 1+(yNodeIdx-1)*NUMBER_OF_X_NODES+(zNodeIdx-1)*NUMBER_OF_X_NODES*NUMBER_OF_Z_NODES
          CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
          IF(nodeDomain==computationalNodeNumber) THEN
            !Fix the node in the x, y & z directions
            WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
            CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
              & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
            CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
              & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
            CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
              & 1,OC_NO_GLOBAL_DERIV,nodeNumber,3,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
          ENDIF
        ENDDO !yNodeIdx
      ENDDO !zNodeIdx
      
      !Set the mid right face node to have a rightward displacement/force
      midNodeNumber = (1 + FLOOR(NUMBER_OF_Y_NODES/2.0_OC_RP))*NUMBER_OF_X_NODES+ &
        & (1 + FLOOR(NUMBER_OF_Z_NODES/2.0_OC_RP))*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,midNodeNumber,nodeDomain,err)
      IF(nodeDomain == computationalNodeNumber) THEN
        !Downward force at the node
        WRITE(*,'("Setting a downward force boundary condition for node ",I0)') midNodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,-MAX_FORCE,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,3,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
    ENDIF

  CASE(SIMPLY_SUPPORTED_LOADING_CASE)

    IF(NUMBER_OF_DIMENSIONS==2) THEN
      
      !Set the bottom left element to be built in
      nodeNumber = 1
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 2
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the bottom right element to be simply supported
      nodeNumber = NUMBER_OF_X_NODES-1
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the mid bottom node to have a downward force
      midNodeNumber = 1+FLOOR(NUMBER_OF_X_NODES/2.0_OC_RP)
      
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,midNodeNumber,nodeDomain,err)
      IF(nodeDomain == computationalNodeNumber) THEN
        !Downward force at the node
        WRITE(*,'("Setting a downward force boundary condition for node ",I0)') midNodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,-MAX_FORCE,err)
      ENDIF
      
    ELSE
      
      !Set the bottom left element to be built in
      nodeNumber = 1
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 2
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 1 + NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 2 + NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the x and y directions
        WRITE(*,'("Setting a built in boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the bottom right element to be simply supported
      nodeNumber = NUMBER_OF_X_NODES-1
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES - 1 + NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES + NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the bottom left deep element to be simply supported
      nodeNumber = 1 + (NUMBER_OF_Z_NODES-2)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 2 + (NUMBER_OF_Z_NODES-2)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 1 + (NUMBER_OF_Z_NODES-1)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES 
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = 2 + (NUMBER_OF_Z_NODES-1)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the bottom right deep element to be simply supported
      nodeNumber = NUMBER_OF_X_NODES - 1 + (NUMBER_OF_Z_NODES - 2)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES  + (NUMBER_OF_Z_NODES - 2)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES - 1 + (NUMBER_OF_Z_NODES - 1)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      nodeNumber = NUMBER_OF_X_NODES + (NUMBER_OF_Z_NODES - 1)*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
      IF(nodeDomain==computationalNodeNumber) THEN
        !Fix the node in the y directions
        WRITE(*,'("Setting a simply supported boundary condition for node ",I0)') nodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
      !Set the mid bottom face node to have a downward force
      midNodeNumber = 1 + FLOOR(NUMBER_OF_X_NODES/2.0_OC_RP) + &
        & ( 1 + FLOOR(NUMBER_OF_Z_NODES/2.0_OC_RP))*NUMBER_OF_X_NODES*NUMBER_OF_Y_NODES      
      CALL OC_Decomposition_NodeDomainGet(decomposition,1,midNodeNumber,nodeDomain,err)
      IF(nodeDomain == computationalNodeNumber) THEN
        !Downward force at the node
        WRITE(*,'("Setting a downward force boundary condition for node ",I0)') midNodeNumber
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,2,OC_BOUNDARY_CONDITION_FIXED,-MAX_FORCE,err)
        CALL OC_BoundaryConditions_SetNode(elasticityBoundaryConditions,elasticityDependentField,OC_FIELD_T_VARIABLE_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,midNodeNumber,3,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
      ENDIF
      
    ENDIF

  CASE DEFAULT
    CALL HandleError("Invalid loading case.")
  END SELECT
 
  CALL OC_SolverEquations_BoundaryConditionsCreateFinish(elasticitySolverEquations,err)
   
  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION PROBLEM
  !-----------------------------------------------------------------------------------------------------------

  !Create the diffusion problem
  CALL OC_Problem_Initialise(diffusionProblem,err)
  diffusionProblemSpecification=[OC_PROBLEM_CLASSICAL_FIELD_CLASS,OC_PROBLEM_DIFFUSION_EQUATION_TYPE, &
    & OC_PROBLEM_LINEAR_DIFFUSION_SUBTYPE]
  CALL OC_Problem_CreateStart(DIFFUSION_PROBLEM_USER_NUMBER,context,diffusionProblemSpecification,diffusionProblem,err)
  CALL OC_Problem_CreateFinish(diffusionProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION CONTROL LOOPS
  !-----------------------------------------------------------------------------------------------------------

  !Create control loops
  CALL OC_Problem_ControlLoopCreateStart(diffusionProblem,err)
  CALL OC_Problem_ControlLoopGet(diffusionProblem,OC_CONTROL_LOOP_NODE,diffusionControlLoop,err)
  CALL OC_Problem_ControlLoopCreateFinish(diffusionProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION SOLVERS
  !-----------------------------------------------------------------------------------------------------------

  !Create problem solvers
  CALL OC_Solver_Initialise(diffusionSolver,err)
  CALL OC_Solver_Initialise(diffusionLinearSolver,err)  
  CALL OC_Problem_SolversCreateStart(diffusionProblem,err)
  CALL OC_Problem_SolverGet(diffusionProblem,OC_CONTROL_LOOP_NODE,1,diffusionSolver,err)
  CALL OC_Solver_OutputTypeSet(diffusionSolver,OC_SOLVER_NO_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(diffusionSolver,OC_SOLVER_PROGRESS_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(diffusionSolver,OC_SOLVER_SOLVER_OUTPUT,err)
  !CALL OC_Solver_OutputTypeSet(diffusionSolver,OC_SOLVER_MATRIX_OUTPUT,err)
  CALL OC_Solver_DynamicLinearSolverGet(diffusionSolver,diffusionLinearSolver,err)
  CALL OC_Solver_LinearTypeSet(diffusionLinearSolver,OC_SOLVER_LINEAR_DIRECT_SOLVE_TYPE,err)
  !CALL OC_Solver_LinearTypeSet(diffusionLinearSolver,OC_SOLVER_LINEAR_ITERATIVE_SOLVE_TYPE,err)
  !CALL OC_Solver_LinearIterativeMaximumIterationsSet(diffusionLinearSolver,1000000,err)
  !CALL OC_Solver_LinearIterativeGMRESRestartSet(elasticitySolver,NUMBER_OF_NODES,err)
  CALL OC_Problem_SolversCreateFinish(diffusionProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION SOLVER EQUATIONS
  !-----------------------------------------------------------------------------------------------------------

  !Create diffusion solver equations and add diffusion equations set to solver equations
  CALL OC_SolverEquations_Initialise(diffusionSolverEquations,err)
  CALL OC_Problem_SolverEquationsCreateStart(diffusionProblem,err)
  !Get the solver equations
  CALL OC_Solver_SolverEquationsGet(diffusionSolver,diffusionSolverEquations,Err)
  !Set the sparsity type
  CALL OC_SolverEquations_SparsityTypeSet(diffusionSolverEquations,OC_SOLVER_SPARSE_MATRICES,err)
  !Add in the diffusion equations set
  CALL OC_SolverEquations_EquationsSetAdd(diffusionSolverEquations,diffusionEquationsSet,diffusionSolverEquationsSetIndex,err)
  CALL OC_Problem_SolverEquationsCreateFinish(diffusionProblem,err)

  !-----------------------------------------------------------------------------------------------------------
  ! DIFFUSION BOUNDARY CONDITIONS
  !-----------------------------------------------------------------------------------------------------------

  !Prescibe boundary conditions for the diffusion problem
  CALL OC_BoundaryConditions_Initialise(diffusionBoundaryConditions,err)
  CALL OC_SolverEquations_BoundaryConditionsCreateStart(diffusionSolverEquations,diffusionBoundaryConditions,err)

  !Set the value of phi on the boundary to zero.
  DO nodeIdx=1,numberOfLocalNodes
    CALL OC_Decomposition_NodeNumberGet(decomposition,1,nodeIdx,nodeNumber,err)
    CALL OC_Decomposition_NodeOnBoundaryGet(decomposition,1,nodeNumber,onBoundary,err)
    IF(onBoundary) THEN
      CALL OC_BoundaryConditions_SetNode(diffusionBoundaryConditions,diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE, &
        & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,OC_BOUNDARY_CONDITION_FIXED,0.0_OC_RP,err)
    ENDIF
  ENDDO !nodeIdx
    
  CALL OC_SolverEquations_BoundaryConditionsCreateFinish(diffusionSolverEquations,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! STRUCTURE FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the structure field
  CALL OC_Field_Initialise(structureField,err)
  CALL OC_Field_CreateStart(STRUCTURE_FIELD_USER_NUMBER,region,structureField,err)
  !Set the field label
  CALL OC_Field_LabelSet(structureField,"Structure",err)
  !Set the field type
  CALL OC_Field_TypeSet(structureField,OC_FIELD_GENERAL_TYPE,err)
  !Set the decomposition to use
  CALL OC_Field_DecompositionSet(structureField,decomposition,err)
  !Set the geometric field
  CALL OC_Field_GeometricFieldSet(structureField,geometricField,err)
  !Set the dependent field
  CALL OC_Field_DependentTypeSet(structureField,OC_FIELD_DEPENDENT_TYPE,err)
  !Set the field variables
  CALL OC_Field_NumberOfVariablesSet(structureField,1,err)
  CALL OC_Field_VariableTypesSet(structureField,[OC_FIELD_U_VARIABLE_TYPE],err)
  !Set the field variable labels
  CALL OC_Field_VariableLabelSet(structureField,OC_FIELD_U_VARIABLE_TYPE,"Str",err)
  !Set the data type to integer
  CALL OC_Field_DataTypeSet(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_INTG_TYPE,err)
  !Set the number of components
  CALL OC_Field_NumberOfComponentsSet(structureField,OC_FIELD_U_VARIABLE_TYPE,1,err)
  !Set the mesh components 
  CALL OC_Field_ComponentMeshComponentSet(structureField,OC_FIELD_U_VARIABLE_TYPE,1,1,err)
  !Set the interpolation types
  CALL OC_Field_ComponentInterpolationSet(structureField,OC_FIELD_U_VARIABLE_TYPE,1, &
    & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  CALL OC_Field_ScalingTypeSet(structureField,OC_FIELD_ARITHMETIC_MEAN_SCALING,err)
  !Finish creating the field
  CALL OC_Field_CreateFinish(structureField,err)

  !Initialise the structure field to 1 (all elements in the structure). If you wish to start with holes set the hole
  !element numbers to zero.
  CALL OC_Field_ComponentValuesInitialise(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & 1,1_OC_Intg,err)
  
  !-----------------------------------------------------------------------------------------------------------
  ! STRAIN ENERGY DENSITY FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the sed field
  CALL OC_Field_Initialise(sedField,err)
  CALL OC_Field_CreateStart(SED_FIELD_USER_NUMBER,region,sedField,err)
  !Set the field label
  CALL OC_Field_LabelSet(sedField,"StrainEnergyDensity",err)
  !Set the field type
  CALL OC_Field_TypeSet(sedField,OC_FIELD_GENERAL_TYPE,err)
  !Set the decomposition to use
  CALL OC_Field_DecompositionSet(sedField,decomposition,err)
  !Set the geometric field
  CALL OC_Field_GeometricFieldSet(sedField,geometricField,err)
  !Set the dependent field
  CALL OC_Field_DependentTypeSet(sedField,OC_FIELD_DEPENDENT_TYPE,err)
  !Set the field variables
  CALL OC_Field_NumberOfVariablesSet(sedField,1,err)
  CALL OC_Field_VariableTypesSet(sedField,[OC_FIELD_U_VARIABLE_TYPE],err)
  !Set the field variable labels
  CALL OC_Field_VariableLabelSet(sedField,OC_FIELD_U_VARIABLE_TYPE,"SED",err)
  !Set the data type to double precision
  CALL OC_Field_DataTypeSet(sedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_DP_TYPE,err)
  !Set the number of components
  CALL OC_Field_NumberOfComponentsSet(sedField,OC_FIELD_U_VARIABLE_TYPE,1,err)
  !Set the mesh components 
  CALL OC_Field_ComponentMeshComponentSet(sedField,OC_FIELD_U_VARIABLE_TYPE,1,1,err)
  !Set the interpolation types
  CALL OC_Field_ComponentInterpolationSet(sedField,OC_FIELD_U_VARIABLE_TYPE,1, &
    & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  CALL OC_Field_ScalingTypeSet(sedField,OC_FIELD_ARITHMETIC_MEAN_SCALING,err)
  !Finish creating the field
  CALL OC_Field_CreateFinish(sedField,err)

  !-----------------------------------------------------------------------------------------------------------
  ! TOPOLOGICAL DERIVATIVE FIELD
  !-----------------------------------------------------------------------------------------------------------

  !Create the td field
  CALL OC_Field_Initialise(tdField,err)
  CALL OC_Field_CreateStart(TD_FIELD_USER_NUMBER,region,tdField,err)
  !Set the field label
  CALL OC_Field_LabelSet(tdField,"TopologicalDerivative",err)
  !Set the field type
  CALL OC_Field_TypeSet(tdField,OC_FIELD_GENERAL_TYPE,err)
  !Set the decomposition to use
  CALL OC_Field_DecompositionSet(tdField,decomposition,err)
  !Set the geometric field
  CALL OC_Field_GeometricFieldSet(tdField,geometricField,err)
  !Set the dependent field
  CALL OC_Field_DependentTypeSet(tdField,OC_FIELD_DEPENDENT_TYPE,err)
  !Set the field variables
  CALL OC_Field_NumberOfVariablesSet(tdField,2,err)
  CALL OC_Field_VariableTypesSet(tdField,[OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_V_VARIABLE_TYPE],err)
  !Set the field variable labels
  CALL OC_Field_VariableLabelSet(tdField,OC_FIELD_U_VARIABLE_TYPE,"TD",err)
  CALL OC_Field_VariableLabelSet(tdField,OC_FIELD_v_VARIABLE_TYPE,"TDN",err)
  !Set the data type to double precision
  CALL OC_Field_DataTypeSet(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_DP_TYPE,err)
  CALL OC_Field_DataTypeSet(tdField,OC_FIELD_V_VARIABLE_TYPE,OC_FIELD_DP_TYPE,err)
  !Set the number of components
  CALL OC_Field_NumberOfComponentsSet(tdField,OC_FIELD_U_VARIABLE_TYPE,1,err)
  CALL OC_Field_NumberOfComponentsSet(tdField,OC_FIELD_V_VARIABLE_TYPE,1,err)
  !Set the mesh components 
  CALL OC_Field_ComponentMeshComponentSet(tdField,OC_FIELD_U_VARIABLE_TYPE,1,1,err)
  CALL OC_Field_ComponentMeshComponentSet(tdField,OC_FIELD_V_VARIABLE_TYPE,1,1,err)
  !Set the interpolation types
  CALL OC_Field_ComponentInterpolationSet(tdField,OC_FIELD_U_VARIABLE_TYPE,1, &
    & OC_FIELD_ELEMENT_BASED_INTERPOLATION,err)
  CALL OC_Field_ComponentInterpolationSet(tdField,OC_FIELD_V_VARIABLE_TYPE,1, &
    & OC_FIELD_NODE_BASED_INTERPOLATION,err)
  CALL OC_Field_ScalingTypeSet(tdField,OC_FIELD_ARITHMETIC_MEAN_SCALING,err)
  !Finish creating the field
  CALL OC_Field_CreateFinish(tdField,err)

  !-----------------------------------------------------------------------------------------------------------
  ! ELASTICITY AND DIFFUSION MAIN WORKFLOW
  !-----------------------------------------------------------------------------------------------------------

  !Export initial fields
  CALL OC_Fields_Initialise(fields,err)  
  CALL OC_Fields_Create(region,fields,err)
  CALL OC_Fields_NodesExport(fields,"BoneOptimisation_0","FORTRAN",err)
  CALL OC_Fields_ElementsExport(fields,"BoneOptimisation_0","FORTRAN",err)
  
  !Initial structural sum and volume ratio
  NULLIFY(structureValues)
  CALL OC_Field_ParameterSetDataGet(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,structureValues,err)
  rankStrSum = SUM(REAL(structureValues(1:numberOfLocalElements),OC_RP))
  !Reduce sums
#ifdef WITH_MPI
  CALL MPI_Allreduce(rankStrSum,strSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
#endif      
  initialVolumeRatio = strSum/REAL(NUMBER_OF_ELEMENTS,OC_RP)
  !Initial values
  NULLIFY(phiValues)
  CALL OC_Field_ParameterSetDataGet(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,phiValues,err)
  NULLIFY(ymValues)
  CALL OC_Field_ParameterSetDataGet(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,ymValues,err)
  NULLIFY(elasticityValues)
  CALL OC_Field_ParameterSetDataGet(elasticityDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & elasticityValues,err)
  NULLIFY(strainValues)
  CALL OC_Field_ParameterSetDataGet(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,strainValues,err)
  NULLIFY(stressValues)
  CALL OC_Field_ParameterSetDataGet(elasticityDerivedField,OC_FIELD_V_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,stressValues,err)
  NULLIFY(strainEnergyValues)
  CALL OC_Field_ParameterSetDataGet(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & strainEnergyValues,err)
  NULLIFY(sedValues)
  CALL OC_Field_ParameterSetDataGet(sedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,sedValues,err)
  NULLIFY(tdValues)
  CALL OC_Field_ParameterSetDataGet(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,tdValues,err)
  NULLIFY(tdnValues)
  CALL OC_Field_ParameterSetDataGet(tdField,OC_FIELD_V_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,tdnValues,err)
  NULLIFY(diffusionSourceValues)
  CALL OC_Field_ParameterSetDataGet(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
    & diffusionSourceValues,err)

  e = 0.0_OC_RP
  eT = 0.0_OC_RP
  A = 0.0_OC_RP
  eTA = 0.0_OC_RP
  eTAe = 0.0_OC_RP
  
  !Topological derivative constants
  A1 = -(3.0*(1.0-POISSONS_RATIO)*(1.0-14.0*POISSONS_RATIO+15.0*POISSONS_RATIO*POISSONS_RATIO))*YOUNGS_MODULUS/ &
    & (2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO))
  A2 = (15.0*YOUNGS_MODULUS*(1.0-POISSONS_RATIO))/(2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO))
  C1 = A1+2.0*A2
  C2 = A1/C1  
  IF(NUMBER_OF_DIMENSIONS==2) THEN
    A(1:3,1:3) = RESHAPE([C1,A1,0.0_OC_RP, &
      & A1,C1,0.0_OC_RP, &
      & 0.0_OC_RP,0.0_OC_RP,C1*(1.0_OC_RP - C2)/2.0_OC_RP], &
      & [3,3])
  ELSE
    A(1:6,1:6) = RESHAPE([C1,A1,A1,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP, &
      & A1,C1,A1,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP, &
      & A1,A1,C1,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP, &
      & 0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,A2,0.0_OC_RP,0.0_OC_RP, &
      & 0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,A2,0.0_OC_RP, &
      & 0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,0.0_OC_RP,A2], &
      & [6,6])
  ENDIF

  
  ! KE = YOUNGS_MODULUS/((1.0_OC_RP-POISSONS_RATIO)*(1.0_OC_RP-POISSONS_RATIO))* &
  !   & RESHAPE([1.0_OC_RP,POISSONS_RATIO,0.0_OC_RP, &
  !   & POISSONS_RATIO,1.0_OC_RP,0.0_OC_RP, &
  !   & 0.0_OC_RP,0.0_OC_RP,(1.0_OC_RP-POISSONS_RATIO)/2.0_OC_RP],[3,3])
  
  ! WRITE(*,*) KE

  !CALL PrintArrayNodeRP(phiValues,1,"Phi")
         
  !Loop over the iterations
  time = TIME_START    
  DO iterationIdx=1,MAXIMUM_NUMBER_OF_ITERATIONS

    WRITE(*,*)
    WRITE(*,'("Solving for iteration ",I0)') iterationIdx
    
    !-----------------------------------------------------------------------------------------------------------
    ! ELASTICITY SOLVE
    !-----------------------------------------------------------------------------------------------------------

    CALL OC_Problem_Solve(elasticityProblem,err)

    !CALL PrintArrayNodeRP(elasticityValues,1,"u")
    !CALL PrintArrayNodeRP(elasticityValues,2,"v")
    
    !-----------------------------------------------------------------------------------------------------------
    ! ELASTICITY DERIVED
    !-----------------------------------------------------------------------------------------------------------

    CALL OC_EquationsSet_DerivedVariableCalculate(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_SMALL_STRAIN,err)
    CALL OC_EquationsSet_DerivedVariableCalculate(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_CAUCHY_STRESS,err)
    CALL OC_EquationsSet_DerivedVariableCalculate(elasticityEquationsSet,OC_EQUATIONS_SET_DERIVED_ELASTIC_WORK,err)
     
    !CALL PrintArrayElementRP(strainValues,1,"e11")
    !CALL PrintArrayElementRP(strainValues,2,"e22")
    !CALL PrintArrayElementRP(strainValues,3,"e12")
    
    !-----------------------------------------------------------------------------------------------------------
    ! ELASTICITY OPTIMISATION PARAMETERS
    !-----------------------------------------------------------------------------------------------------------

    rankObjectiveSum = 0.0_OC_RP
    rankSEDSum = 0.0_OC_RP
    rankStrSum = 0.0_OC_RP

    DO elementIdx = 1,numberOfLocalElements
      CALL OC_Decomposition_ElementNumberGet(decomposition,elementIdx,elementNumber,err)
      
      !WRITE(*,'("Element : ",I5)') elementNumber
      
      CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,voigt11Component,e11,err)
      CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,voigt12Component,e12,err)
      CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,voigt22Component,e22,err)
      e(voigt11Component,1) = e11
      e(voigt12Component,1) = e12
      e(voigt22Component,1) = e22
      eT(1,voigt11Component) = e11
      eT(1,voigt12Component) = e12
      eT(1,voigt22Component) = e22        
      IF(NUMBER_OF_DIMENSIONS==3) THEN
        CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & elementNumber,voigt13Component,e13,err)
        CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & elementNumber,voigt23Component,e23,err)
        CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & elementNumber,voigt33Component,e33,err)
        e(voigt13Component,1) = e13
        e(voigt23Component,1) = e23
        e(voigt33Component,1) = e33
        eT(1,voigt13Component) = e13
        eT(1,voigt23Component) = e23
        eT(1,voigt33Component) = e33        
      ENDIF
      CALL OC_Field_ParameterSetGetElement(elasticityDerivedField,OC_FIELD_W_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,1,strainEnergy,err)
      
      !eTA(1,1:numberOfVoigtComponents) = MATMUL(eT(1,1:numberOfVoigtComponents), &
      !  & A(1:numberOfVoigtComponents,1:numberOfVoigtComponents))
      !eTAe = MATMUL(eTA(1,1:numberOfVoigtComponents),e(1:numberOfVoigtComponents,1))
      eTA= MATMUL(eT,A)
      eTAe = MATMUL(eTA,e)
      energy = eTAe(1,1)
      
      CALL OC_Field_ParameterSetGetElement(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,1,strValue,err)
      
      strainEnergyDensity=(YOUNGS_MODULUS_MIN+REAL(strValue,OC_RP)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*strainEnergy
      
      CALL OC_Field_ParameterSetUpdateElement(sedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,1,strainEnergyDensity,err)
      
      topologicalDerivative=(YOUNGS_MODULUS_MIN+REAL(strValue,OC_RP)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*energy
      
      CALL OC_Field_ParameterSetUpdateElement(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & elementNumber,1,topologicalDerivative,err)
      
      rankStrSum=rankStrSum+REAL(strValue,OC_RP)
      rankSEDSum=rankSEDSum+strainEnergyDensity
      rankObjectiveSum=rankObjectiveSum+strainEnergy
      
    ENDDO !elementIdx

    !WRITE(*,'(" Rank str sum = ",F12.5)') rankStrSum
    
    !Reduce sums
#ifdef WITH_MPI
    CALL MPI_Allreduce(rankStrSum,strSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
    CALL MPI_Allreduce(rankSEDSum,sedSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
    CALL MPI_Allreduce(rankObjectiveSum,objectiveSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
#endif    
    !WRITE(*,'(" Reduced str sum = ",F12.5)') strSum    

    !Update fields
    CALL OC_Field_ParameterSetUpdateStart(sedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateStart(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(sedField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)    
 
    !Compute the nodal topological derivative values and sums
    rankTDSum = 0.0_OC_RP
    rankAbsTDSum = 0.0_OC_RP
    
    DO nodeIdx=1,numberOfLocalNodes
      CALL OC_Decomposition_NodeNumberGet(decomposition,1,nodeIdx,nodeNumber,err)
      
      ! Loop over the elements surrounding the node to determine the average
      averageTD = 0.0_OC_RP
      CALL OC_Decomposition_NodeNumberOfSurroundingElementsGet(decomposition,1,nodeNumber,numberOfSurroundingElements,err)
      DO surroundingElementIdx=1,numberOfSurroundingElements
        CALL OC_Decomposition_NodeSurroundingElementGet(decomposition,1,nodeNumber,surroundingElementIdx,surroundingElement,err)
        CALL OC_Field_ParameterSetGetElement(tdField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & surroundingElement,1,topologicalDerivative,err)
        averageTD = averageTD + topologicalDerivative
      ENDDO !surroundingElementIdx        
      averageTD = averageTD/REAL(numberOfSurroundingElements,OC_RP)
      !Set the TD at the node
      CALL OC_Field_ParameterSetUpdateNode(tdField,OC_FIELD_V_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,1,OC_NO_GLOBAL_DERIV, &
        & nodeNumber,1,averageTD,err)
      !Update the sums
      rankTDSum=rankTDSum+averageTD
      rankAbsTDSum=rankAbsTDSum+ABS(averageTD)        
    ENDDO !nodeIdx

    !Reduce sums
#ifdef WITH_MPI
    CALL MPI_Allreduce(rankTDSum,tdSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
    CALL MPI_Allreduce(rankAbsTDSum,absTDSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,mpiIError)
#endif
    
    WRITE(*,'("sumTDN = ",E12.5)') tdSum
    
    !TODO: reduce the values across the ranks
    volumeRatio = strSum/REAL(NUMBER_OF_ELEMENTS,OC_RP)    
    objective(iterationIdx)=objectiveSum
   
    !CALL PrintArrayElementRP(strainEnergyValues,1,"SE")
    !CALL PrintArrayElementRP(sedValues,1,"SED")
    !CALL PrintArrayElementRP(tdValues,1,"TD")
    !CALL PrintArrayNodeRP(tdnValues,1,"TDN")
    
    !-----------------------------------------------------------------------------------------------------------
    ! CALCULATE AUGMENTED LAGRANGIAN PARAMETERS
    !-----------------------------------------------------------------------------------------------------------

    WRITE(*,'("Current volume ratio = ",F9.5)') volumeRatio
    WRITE(*,'("Topological derivative sum = ",F11.5)') tdSum
    WRITE(*,'("ABS Topological derivative sum = ",F11.5)') absTDSum
    
    maximumG = MAX_VOLUME_RATIO+(initialVolumeRatio-MAX_VOLUME_RATIO)* &
      & MAX(0.0_OC_RP,1.0_OC_RP-REAL(iterationIdx,OC_RP)/REAL(N_VOL_ITERATIONS,OC_RP))
    G = volumeRatio - maximumG
    lambdaValue = tdSum/REAL(NUMBER_OF_NODES,OC_RP)*EXP(LEVEL_SET_P_PARAM*(G/maximumG+LEVEL_SET_D_PARAM))
    C = REAL(NUMBER_OF_NODES,OC_RP)/absTDSum

    WRITE(*,'("Maximum G = ",F9.5)') maximumG
    WRITE(*,'("G = ",F9.5)') G
    WRITE(*,'("lambda = ",F9.5)') lambdaValue
    WRITE(*,'("C = ",E12.5)') C

    !Update the diffusion source to be C*topologicalDerivative - lambda
    DO nodeIdx = 1,numberOfLocalNodes
      CALL OC_Decomposition_NodeNumberGet(decomposition,1,nodeIdx,nodeNumber,err)
      CALL OC_Field_ParameterSetGetNode(tdField,OC_FIELD_V_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,1,OC_NO_GLOBAL_DERIV, &
        & nodeNumber,1,topologicalDerivative,err)
      diffusionSource = C*(topologicalDerivative - lambdaValue)
      
      CALL OC_Field_ParameterSetUpdateNode(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
        & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,-diffusionSource,err)
    ENDDO !nodeIdx

    CALL OC_Field_ParameterSetUpdateStart(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(diffusionSourceField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    
    !CALL PrintArrayNodeRP(diffusionSourceValues,1,"Diffusion source")
      
    !-----------------------------------------------------------------------------------------------------------
    ! DIFFUSION SOLVE
    !-----------------------------------------------------------------------------------------------------------

    CALL OC_ControlLoop_TimesSet(diffusionControlLoop,time,time+TIME_STEP,TIME_STEP,err)
    
    CALL OC_Problem_Solve(diffusionProblem,err)

    !CALL PrintArrayNodeRP(phiValues,1,"Phi")
         
    !-----------------------------------------------------------------------------------------------------------
    ! RECALCULATE THE NEW STRUCUTRE FIELD AND VOLUME
    !-----------------------------------------------------------------------------------------------------------

    !Loop over the local elements
    DO elementIdx = 1,numberOfLocalElements
      CALL OC_Decomposition_ElementNumberGet(decomposition,elementIdx,elementNumber,err)
      !Loop over nodes in the element
      CALL OC_Basis_Initialise(elementBasis,err)
      CALL OC_Decomposition_ElementBasisGet(decomposition,1,elementNumber,elementBasis,err)
      CALL OC_Basis_NumberOfLocalNodesGet(elementBasis,numberOfElementNodes,err)
      averagePhi = 0.0_OC_RP
      DO localNodeIdx = 1,numberOfElementNodes
        CALL OC_Decomposition_ElementNodeGet(decomposition,1,elementNumber,localNodeIdx,nodeNumber,err)
        !Get the value of phi at the node
        CALL OC_Field_ParameterSetGetNode(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,nodalPhi,err)
        !Reset the phi limits
        nodalPhi = MIN(1.0_OC_RP,MAX(-1.0_OC_RP,nodalPhi))
        !Update phi 
        CALL OC_Decomposition_NodeDomainGet(decomposition,1,nodeNumber,nodeDomain,err)
        IF(nodeDomain==computationalNodeNumber) THEN
          CALL OC_Field_ParameterSetUpdateNode(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
            & 1,OC_NO_GLOBAL_DERIV,nodeNumber,1,nodalPhi,err)
        ENDIF
        !Update average
        averagePhi = averagePhi + nodalPhi
      ENDDO !localNodeIdx
      averagePhi = averagePhi/REAL(numberOfElementNodes,OC_RP)
      !If the average Phi in the element is less than zero remove the element
      IF(averagePhi <= 0.0_OC_RP) THEN
        !Remove the element from the structure
        CALL OC_Field_ParameterSetUpdateElement(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
          & elementNumber,1,0_OC_Intg,err)
        IF(NUMBER_OF_DIMENSIONS == 2) THEN
          CALL OC_Field_ParameterSetUpdateElement(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
            & elementNumber,1,YOUNGS_MODULUS_MIN,err)
        ELSE
          CALL OC_Field_ParameterSetUpdateElement(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
            & elementNumber,1,LAME_LAMBDA_MIN,err)
          CALL OC_Field_ParameterSetUpdateElement(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE, &
            & elementNumber,2,LAME_MU_MIN,err)
        ENDIF
      ENDIF
    ENDDO !elementIdx

    !Update the fields
    CALL OC_Field_ParameterSetUpdateStart(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateStart(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateStart(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(diffusionDependentField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(structureField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
    CALL OC_Field_ParameterSetUpdateFinish(elasticityMaterialsField,OC_FIELD_U_VARIABLE_TYPE,OC_FIELD_VALUES_SET_TYPE,err)
      
    !CALL PrintArrayNodeRP(phiValues,1,"Phi")         
    !CALL PrintArrayElementIntg(structureValues,1,"Str")
    !CALL PrintArrayElementRP(ymValues,1,"E")
      
    !-----------------------------------------------------------------------------------------------------------
    ! OUTPUT
    !-----------------------------------------------------------------------------------------------------------

    WRITE(*,*)
    WRITE(*,'("Iteration Number = ",I0,", Objective = ",E12.5,", Volume ratio = ",F7.4,", lambda = ",F7.4)') iterationIdx, &
      & objective(iterationIdx),volumeRatio,lambdaValue
   
    !Export results
    WRITE(iterationString,'(I0)') iterationIdx
    filename="BoneOptimisation_"//ADJUSTL(TRIM(iterationString))
    WRITE(*,'("Writing results to ",A)') filename
    CALL OC_Fields_NodesExport(fields,filename,"FORTRAN",err)
    CALL OC_Fields_ElementsExport(fields,filename,"FORTRAN",err)

    time=time+TIME_STEP
    
  ENDDO !iterationIdx

  !-----------------------------------------------------------------------------------------------------------
  ! FINALISE AND CLEANUP
  !-----------------------------------------------------------------------------------------------------------

  !Destroy the context
  CALL OC_Context_Destroy(context,err)
  !Finialise OpenCMISS
  CALL OC_Finalise(err)
  
  WRITE(*,'("Program successfully completed.")')
  STOP  
  
CONTAINS
  
  SUBROUTINE HandleError(errorString)
    
    CHARACTER(LEN=*), INTENT(IN) :: errorString
    
    WRITE(*,'(">>ERROR: ",A)') errorString(1:LEN_TRIM(errorString))
    
    STOP
  END SUBROUTINE HandleError

  SUBROUTINE PrintArrayNodeIntg(values,componentNumber,name)

    INTEGER(OC_Intg), POINTER :: values(:)
    INTEGER(OC_Intg), INTENT(IN) :: componentNumber
    CHARACTER(LEN=*), INTENT(IN) :: name

    INTEGER(OC_Intg) :: xNodeIdx,yNodeIdx

    WRITE(*,*)
    WRITE(*,'(A," :")') name(1:LEN_TRIM(name))
    DO yNodeIdx=NUMBER_OF_Y_NODES,1,-1
      WRITE(*,'(100(I1,X))') (values(xNodeIdx+(yNodeIdx-1)*NUMBER_OF_X_NODES+(componentNumber-1)*NUMBER_OF_NODES),xNodeIdx=1,NUMBER_OF_X_NODES)
    ENDDO !yNodeIdx

  END SUBROUTINE PrintArrayNodeIntg

  SUBROUTINE PrintArrayNodeRP(values,componentNumber,name)

    REAL(OC_RP), POINTER :: values(:)
    INTEGER(OC_Intg), INTENT(IN) :: componentNumber
    CHARACTER(LEN=*), INTENT(IN) :: name

    INTEGER(OC_Intg) :: xNodeIdx,yNodeIdx
    
    WRITE(*,*)
    WRITE(*,'(A," :")') name(1:LEN_TRIM(name))
    DO yNodeIdx=NUMBER_OF_Y_NODES,1,-1
      WRITE(*,'(100(F10.5,X))') (values(xNodeIdx+(yNodeIdx-1)*NUMBER_OF_X_NODES+(componentNumber-1)*NUMBER_OF_NODES),xNodeIdx=1,NUMBER_OF_X_NODES)
    ENDDO !yNodeIdx

  END SUBROUTINE PrintArrayNodeRP

  SUBROUTINE PrintArrayElementIntg(values,componentNumber,name)

    INTEGER(OC_Intg), POINTER :: values(:)
    INTEGER(OC_Intg), INTENT(IN) :: componentNumber
    CHARACTER(LEN=*), INTENT(IN) :: name

    INTEGER(OC_Intg) :: xElementIdx,yElementIdx

    WRITE(*,*)
    WRITE(*,'(A," :")') name(1:LEN_TRIM(name))
    DO yElementIdx=NUMBER_OF_Y_ELEMENTS,1,-1
      WRITE(*,'(100(I1,X))') (values(xElementIdx+(yElementIdx-1)*NUMBER_OF_X_ELEMENTS+(componentNumber-1)*NUMBER_OF_ELEMENTS),xElementIdx=1,NUMBER_OF_X_ELEMENTS)
    ENDDO !yElementIdx
    
  END SUBROUTINE PrintArrayElementIntg

  SUBROUTINE PrintArrayElementRP(values,componentNumber,name)

    REAL(OC_RP), POINTER :: values(:)
    INTEGER(OC_Intg), INTENT(IN) :: componentNumber
    CHARACTER(LEN=*), INTENT(IN) :: name

    INTEGER(OC_Intg) :: xElementIdx,yElementIdx
    
    WRITE(*,*)
    WRITE(*,'(A," :")') name(1:LEN_TRIM(name))
    DO yElementIdx=NUMBER_OF_Y_ELEMENTS,1,-1
      WRITE(*,'(100(F10.5,X))') (values(xElementIdx+(yElementIdx-1)*NUMBER_OF_X_ELEMENTS+(componentNumber-1)*NUMBER_OF_ELEMENTS),xElementIdx=1,NUMBER_OF_X_ELEMENTS)
    ENDDO !yElementIdx

  END SUBROUTINE PrintArrayElementRP

END PROGRAM BoneOptimisation

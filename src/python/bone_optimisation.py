#!/usr/bin/env python
#
# This is an example script for a bone optimisation problem using OpenCMISS calls in python.
# By Chris Bradley
#
#

import sys,os,math
import numpy as np

# Intialise OpenCMISS
from opencmiss.opencmiss import OpenCMISS_Python as oc

#-----------------------------------------------------------------------------------------------------------
# SET PROBLEM PARAMETERS
#-----------------------------------------------------------------------------------------------------------

# Geometric parameters

HEIGHT = 20.0 # mm
WIDTH = 10.0 # mm
LENGTH = 10.0 # mm

# Elasticity parameters

#YOUNGS_MODULUS = 30.0E6 # mg.mm^-1.ms^-2
#YOUNGS_MODULUS_MIN = 30.0 # mg.mm^-1.ms^-2
YOUNGS_MODULUS = 1.0 # mg.mm^-1.ms^-2
YOUNGS_MODULUS_MIN = 0.00001 # mg.mm^-1.ms^-2
POISSONS_RATIO = 0.3
THICKNESS = 1.0 # mm (for plane strain and stress)

# Boundary condition 

DIRICHLET_BCS = 1
NEUMANN_BCS = 2

boundaryConditionType = DIRICHLET_BCS
MAX_DISPLACEMENT = -0.10*HEIGHT;
MAX_FORCE = -10.0 # N.mm^-2

# Diffusion parameters

DIFFUSION_A_PARAM = 1.0
DIFFUSION_TAU_PARAM = 0.001 # Stabilisation parameter

# Optimisation parameters

MAX_VOLUME_FRACTION = 0.50
LEVEL_SET_P_PARAM = 4
LEVEL_SET_D_PARAM = -0.02
N_VOL_ITERATIONS = 100

# Time information
TIME_START = 0.00
TIME_STEP = 0.10

MAXIMUM_NUMBER_OF_ITERATIONS = 10 # Maximum number of iterations in the main loop

PHI_ZERO_TOLERANCE = 0.00001 # Tolerance for the average phi value in an element to remove an element from the structure

# Generic parameters

LINEAR_LAGRANGE = 1
QUADRATIC_LAGRANGE = 2
CUBIC_LAGRANGE = 3
CUBIC_HERMITE = 4
LINEAR_SIMPLEX = 5
QUADRATIC_SIMPLEX = 6
CUBIC_SIMPLEX = 7

if (boundaryConditionType == DIRICHLET_BCS):
    DISPLACEMENT_BC = MAX_DISPLACEMENT
elif (BOUNDARY_CONDITION_TYPE == NEUMANN_BCS):
    DISPLACEMENT_BC = MAX_FORCE
else:
    print('Invalid boundary condition type')
    exit()
   
(CONTEXT_USER_NUMBER,
 COORDINATE_SYSTEM_USER_NUMBER,
 REGION_USER_NUMBER,
 BASIS_USER_NUMBER,
 GENERATED_MESH_USER_NUMBER,
 MESH_USER_NUMBER,
 DECOMPOSITION_USER_NUMBER,
 DECOMPOSER_USER_NUMBER,
 GEOMETRIC_FIELD_USER_NUMBER,
 ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,
 ELASTICITY_MATERIALS_FIELD_USER_NUMBER,
 ELASTICITY_ANALYTIC_FIELD_USER_NUMBER,
 ELASTICITY_DERIVED_FIELD_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_USER_NUMBER,
 ELASTICITY_PROBLEM_USER_NUMBER,
 DIFFUSION_DEPENDENT_FIELD_USER_NUMBER,
 DIFFUSION_STRUCTURE_FIELD_USER_NUMBER,
 DIFFUSION_MATERIALS_FIELD_USER_NUMBER,
 DIFFUSION_SOURCE_FIELD_USER_NUMBER,
 DIFFUSION_EQUATIONS_SET_FIELD_USER_NUMBER,
 DIFFUSION_EQUATIONS_SET_USER_NUMBER,
 DIFFUSION_PROBLEM_USER_NUMBER
 ) = range(1,24)

NUMBER_OF_GAUSS_XI = 4

numberOfGlobalXElements = 10
numberOfGlobalYElements = 6
interpolationType = LINEAR_LAGRANGE
#interpolationType = LINEAR_SIMPLEX

# Override with command line arguments if need be
if len(sys.argv) > 1:
    if len(sys.argv) > 5:
        sys.exit('ERROR: too many arguments- currently only accepting up to 4 options: numberOfGlobalXElements numberOfGlobalYElements interpolationType')
    numberOfGlobalXElements = int(sys.argv[1])
    if len(sys.argv) > 2:
        numberOfGlobalYElements = int(sys.argv[2])
    if len(sys.argv) > 3:
        interpolationType = int(sys.argv[3])
numberOfGlobalZElements = 1

if (numberOfGlobalXElements <= 1):
    sys.exit('ERROR: number of global X elements must be greater than 1.')
if (numberOfGlobalYElements <= 1):
    sys.exit('ERROR: number of global Y elements must be greater than 1.')

if (interpolationType == LINEAR_LAGRANGE):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE
    numberOfNodesXi = 2
    numberOfGaussXi = 2
elif (interpolationType == QUADRATIC_LAGRANGE):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE
    numberOfNodesXi = 3
    numberOfGaussXi = 3
elif (interpolationType == CUBIC_LAGRANGE):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE
    numberOfNodesXi = 4
    numberOfGaussXi = 4
elif (interpolationType == CUBIC_HERMITE):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.CUBIC_HERMITE
    numberOfNodesXi = 2
    numberOfGaussXi = 4
elif (interpolationType == LINEAR_SIMPLEX):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX
    numberOfNodesXi = 2
    gaussOrder = 4
elif (interpolationType == QUADRATIC_SIMPLEX):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX
    numberOfNodesXi = 3
    gaussOrder = 4
elif (interpolationType == CUBIC_SIMPLEX):
    interpolationTypeXi = oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX
    numberOfNodesXi = 4
    gaussOrder = 5
else:
    sys.exit('The interpolation type of ',interpolationType,' is invalid.')

haveHermite = (interpolationType == CUBIC_HERMITE)
haveSimplex = (interpolationType == LINEAR_SIMPLEX or interpolationType == QUADRATIC_SIMPLEX or interpolationType == CUBIC_SIMPLEX)

if (haveSimplex):
    elementFactor = 2
else:
    elementFactor = 1
numberOfElements = numberOfGlobalXElements*numberOfGlobalYElements*elementFactor
numberOfXNodes = numberOfGlobalXElements*(numberOfNodesXi-1)+1
numberOfYNodes = numberOfGlobalYElements*(numberOfNodesXi-1)+1
numberOfNodes = numberOfXNodes*numberOfYNodes            
numberOfDimensions = 2
numberOfXi = numberOfDimensions
if (not haveSimplex):
    numberOfGauss = pow(numberOfGaussXi,numberOfXi)

#-----------------------------------------------------------------------------------------------------------
# CONTEXT AND WORLD REGION
#-----------------------------------------------------------------------------------------------------------

context = oc.Context()
context.Create(CONTEXT_USER_NUMBER)

worldRegion = oc.Region()
context.WorldRegionGet(worldRegion)

#-----------------------------------------------------------------------------------------------------------
# DIAGNOSTICS AND COMPUTATIONAL NODE INFORMATION
#-----------------------------------------------------------------------------------------------------------

oc.OutputSetOn("BoneOptimisation")

#oc.DiagnosticsSetOn(oc.DiagnosticTypes.IN,[1,2,3,4,5],"",["BoundaryConditionsVariable_NeumannIntegrate"])

# Get the computational nodes information
computationEnvironment = oc.ComputationEnvironment()
context.ComputationEnvironmentGet(computationEnvironment)
numberOfComputationalNodes = computationEnvironment.NumberOfWorldNodesGet()
computationalNodeNumber = computationEnvironment.WorldNodeNumberGet()

worldWorkGroup = oc.WorkGroup()
computationEnvironment.WorldWorkGroupGet(worldWorkGroup)

#-----------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
#-----------------------------------------------------------------------------------------------------------

coordinateSystem = oc.CoordinateSystem()
coordinateSystem.CreateStart(COORDINATE_SYSTEM_USER_NUMBER,context)
coordinateSystem.DimensionSet(numberOfDimensions)
coordinateSystem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# REGION
#-----------------------------------------------------------------------------------------------------------

region = oc.Region()
region.CreateStart(REGION_USER_NUMBER,worldRegion)
region.LabelSet("Bone")
region.CoordinateSystemSet(coordinateSystem)
region.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# BASIS
#-----------------------------------------------------------------------------------------------------------

basis = oc.Basis()
basis.CreateStart(BASIS_USER_NUMBER,context)
if (haveSimplex):
    basis.TypeSet(oc.BasisTypes.SIMPLEX)
else:
    basis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
basis.NumberOfXiSet(numberOfXi)
basis.InterpolationXiSet([interpolationTypeXi]*numberOfXi)
if (haveSimplex):
    basis.QuadratureOrderSet(gaussOrder)
else:
    basis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*numberOfXi)
basis.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# MESH
#-----------------------------------------------------------------------------------------------------------

generatedMesh = oc.GeneratedMesh()
generatedMesh.CreateStart(GENERATED_MESH_USER_NUMBER,region)
generatedMesh.TypeSet(oc.GeneratedMeshTypes.REGULAR)
generatedMesh.BasisSet([basis])
if (numberOfDimensions == 2):
    generatedMesh.ExtentSet([LENGTH,HEIGHT])
    generatedMesh.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements])
else:
    generatedMesh.ExtentSet([LENGTH,WIDTH,HEIGHT])
    generatedMesh.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements,numberOfGlobalZElements])
mesh = oc.Mesh()
generatedMesh.CreateFinish(MESH_USER_NUMBER,mesh)

#-----------------------------------------------------------------------------------------------------------
# MESH DECOMPOSITION
#-----------------------------------------------------------------------------------------------------------

decomposition = oc.Decomposition()
decomposition.CreateStart(DECOMPOSITION_USER_NUMBER,mesh)
decomposition.TypeSet(oc.DecompositionTypes.CALCULATED)
decomposition.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DECOMPOSER
#-----------------------------------------------------------------------------------------------------------

decomposer = oc.Decomposer()
decomposer.CreateStart(DECOMPOSER_USER_NUMBER,worldRegion,worldWorkGroup)
decompositionIndex = decomposer.DecompositionAdd(decomposition)
decomposer.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GEOMETRIC FIELD
#-----------------------------------------------------------------------------------------------------------

geometricField = oc.Field()
geometricField.CreateStart(GEOMETRIC_FIELD_USER_NUMBER,region)
geometricField.DecompositionSet(decomposition)
geometricField.TypeSet(oc.FieldTypes.GEOMETRIC)
geometricField.VariableLabelSet(oc.FieldVariableTypes.U,"Geometry")
geometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,1,1)
geometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,2,1)
if (numberOfDimensions == 3):
    geometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,3,1)
geometricField.CreateFinish()

# Set geometry from the generated mesh
generatedMesh.GeometricParametersCalculate(geometricField)

#-----------------------------------------------------------------------------------------------------------
# ELASTITICY EQUATION SETS
#-----------------------------------------------------------------------------------------------------------

# Create linear elasiticity equations set
elasticityEquationsSetField = oc.Field()
elasticityEquationsSet = oc.EquationsSet()
if (numberOfDimensions == 2):
    elasticityEquationsSetSpecification = [oc.EquationsSetClasses.ELASTICITY,
                                           oc.EquationsSetTypes.LINEAR_ELASTICITY,
                                           oc.EquationsSetSubtypes.TWO_DIMENSIONAL_PLANE_STRESS]
else:
    elasticityEquationsSetSpecification = [oc.EquationsSetClasses.ELASTICITY,
                                           oc.EquationsSetTypes.LINEAR_ELASTICITY,
                                           oc.EquationsSetSubtypes.THREE_DIMENSIONAL_ISOTROPIC]
elasticityEquationsSet.CreateStart(ELASTICITY_EQUATIONS_SET_USER_NUMBER,region,geometricField,
                         elasticityEquationsSetSpecification,
                         ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,elasticityEquationsSetField)
elasticityEquationsSet.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS SET DEPENDENT
#-----------------------------------------------------------------------------------------------------------

elasticityDependentField = oc.Field()
elasticityEquationsSet.DependentCreateStart(ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,elasticityDependentField)
elasticityDependentField.LabelSet("ElasticityDependent")
elasticityDependentField.VariableLabelSet(oc.FieldVariableTypes.U,"Displacement")
elasticityDependentField.VariableLabelSet(oc.FieldVariableTypes.T,"Traction")
elasticityEquationsSet.DependentCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS SET MATERIALS
#-----------------------------------------------------------------------------------------------------------

elasticityMaterialsField = oc.Field()
elasticityEquationsSet.MaterialsCreateStart(ELASTICITY_MATERIALS_FIELD_USER_NUMBER,elasticityMaterialsField)
elasticityMaterialsField.LabelSet("ElasticityMaterials")
elasticityMaterialsField.VariableLabelSet(oc.FieldVariableTypes.U,"ElasticityMaterials")
elasticityMaterialsField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
elasticityMaterialsField.ComponentInterpolationSet(oc.FieldVariableTypes.U,2,oc.FieldInterpolationTypes.CONSTANT)
elasticityEquationsSet.MaterialsCreateFinish()    
# Initialise the analytic field values
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1,YOUNGS_MODULUS)
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   2,POISSONS_RATIO)
if(numberOfDimensions==2):
    elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                       3,THICKNESS)

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS SET DERIVED
#-----------------------------------------------------------------------------------------------------------

# Create a field for the derived field. Have three variables U - Small strain tensor, V - Cauchy stress, W - Elastic Work
if(numberOfDimensions==2):
    numberOfTensorComponents = 3
else:
    numberOfTensorComponents = 6
elasticityDerivedField = oc.Field()
elasticityDerivedField.CreateStart(ELASTICITY_DERIVED_FIELD_USER_NUMBER,region)
elasticityDerivedField.LabelSet("ElasticityDerived")
elasticityDerivedField.TypeSet(oc.FieldTypes.GENERAL)
elasticityDerivedField.DecompositionSet(decomposition)
elasticityDerivedField.GeometricFieldSet(geometricField)
elasticityDerivedField.DependentTypeSet(oc.FieldDependentTypes.DEPENDENT)
elasticityDerivedField.NumberOfVariablesSet(3)
elasticityDerivedField.VariableTypesSet([oc.FieldVariableTypes.U,oc.FieldVariableTypes.V,oc.FieldVariableTypes.W])
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U,"SmallStrain")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.V,"CauchyStress")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.W,"ElasticWork")
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U,numberOfTensorComponents)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.V,numberOfTensorComponents)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.W,1)
for componentIdx in range(1,numberOfTensorComponents+1):
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,1)
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.V,componentIdx,1)
elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.W,1,1)
for componentIdx in range(1,numberOfTensorComponents+1):
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U,componentIdx,oc.FieldInterpolationTypes.ELEMENT_BASED)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.V,componentIdx,oc.FieldInterpolationTypes.ELEMENT_BASED)
elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.W,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
elasticityDerivedField.CreateFinish()

# Create the derived equations set fields
elasticityEquationsSet.DerivedCreateStart(ELASTICITY_DERIVED_FIELD_USER_NUMBER,elasticityDerivedField)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.SMALL_STRAIN,oc.FieldVariableTypes.U)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,oc.FieldVariableTypes.V)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.ELASTIC_WORK,oc.FieldVariableTypes.W)
elasticityEquationsSet.DerivedCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS
#-----------------------------------------------------------------------------------------------------------

elasticityEquations = oc.Equations()
elasticityEquationsSet.EquationsCreateStart(elasticityEquations)
#elasticityEquations.SparsityTypeSet(oc.EquationsSparsityTypes.FULL)
elasticityEquations.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
elasticityEquations.OutputTypeSet(oc.EquationsOutputTypes.NONE)
#elasticityEquations.OutputTypeSet(oc.EquationsOutputTypes.TIMING)
#elasticityEquations.OutputTypeSet(oc.EquationsOutputTypes.MATRIX)
#elasticityEquations.OutputTypeSet(oc.EquationsOutputTypes.ELEMENT_MATRIX)
elasticityEquationsSet.EquationsCreateFinish()


#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS SET
#-----------------------------------------------------------------------------------------------------------

# Create a diffusion equations set
diffusionEquationsSetField = oc.Field()
diffusionEquationsSet = oc.EquationsSet()
diffusionEquationsSetSpecification = [oc.EquationsSetClasses.CLASSICAL_FIELD,
                                      oc.EquationsSetTypes.DIFFUSION_EQUATION,
                                      oc.EquationsSetSubtypes.GENERALISED_DIFFUSION]
diffusionEquationsSet.CreateStart(DIFFUSION_EQUATIONS_SET_USER_NUMBER,region,geometricField,
                                  diffusionEquationsSetSpecification,
                                  DIFFUSION_EQUATIONS_SET_FIELD_USER_NUMBER,diffusionEquationsSetField)
diffusionEquationsSet.CreateFinish()


#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS SET DEPENDENT
#-----------------------------------------------------------------------------------------------------------

diffusionDependentField = oc.Field()
diffusionEquationsSet.DependentCreateStart(DIFFUSION_DEPENDENT_FIELD_USER_NUMBER,diffusionDependentField)
diffusionDependentField.LabelSet("DiffusionDependent")
diffusionDependentField.VariableLabelSet(oc.FieldVariableTypes.U,"Phi")
diffusionDependentField.VariableLabelSet(oc.FieldVariableTypes.DELUDELN,"DelPhiDelN")
diffusionEquationsSet.DependentCreateFinish()

# Initialise the dependent phi field to 1. If you wish to start with holes set the hole boundary nodes to 0.
diffusionDependentField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,1.0)

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION STRUCTURE FIELD
#-----------------------------------------------------------------------------------------------------------

diffusionStructureField = oc.Field()
diffusionStructureField.CreateStart(DIFFUSION_STRUCTURE_FIELD_USER_NUMBER,region)
# Set the type
diffusionStructureField.TypeSet(oc.FieldTypes.GENERAL)
# Set the decomposition
diffusionStructureField.DecompositionSet(decomposition)
# Set the geometric field
diffusionStructureField.GeometricFieldSet(geometricField)
# Set the label
diffusionStructureField.LabelSet("DiffusionStructure")
# Set the variables
diffusionStructureField.NumberOfVariablesSet(1)
diffusionStructureField.VariableTypesSet([oc.FieldVariableTypes.U])
diffusionStructureField.VariableLabelSet(oc.FieldVariableTypes.U,"Str")
diffusionStructureField.DataTypeSet(oc.FieldVariableTypes.U,oc.FieldDataTypes.INTG)
# Set the components
diffusionStructureField.NumberOfComponentsSet(oc.FieldVariableTypes.U,1)
diffusionStructureField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,1,1)
diffusionStructureField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
# Finish the field
diffusionStructureField.CreateFinish()

# Initialise the structure field to 1 (all elements in the structure). If you wish to start with holes set the hole
# element numbers to 0.
diffusionStructureField.ComponentValuesInitialiseIntg(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,1)

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS SET MATERIALS
#-----------------------------------------------------------------------------------------------------------

diffusionMaterialsField = oc.Field()
diffusionEquationsSet.MaterialsCreateStart(DIFFUSION_MATERIALS_FIELD_USER_NUMBER,diffusionMaterialsField)
diffusionMaterialsField.LabelSet("DiffusionMaterials")
diffusionMaterialsField.VariableLabelSet(oc.FieldVariableTypes.U,"DiffusionMaterials")
diffusionEquationsSet.MaterialsCreateFinish()    
# Initialise the diffusion materials field values
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1,DIFFUSION_A_PARAM)
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   2,DIFFUSION_TAU_PARAM)
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   3,DIFFUSION_TAU_PARAM)

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS SET SOURCE
#-----------------------------------------------------------------------------------------------------------

diffusionSourceField = oc.Field()
diffusionEquationsSet.SourceCreateStart(DIFFUSION_SOURCE_FIELD_USER_NUMBER,diffusionSourceField)
diffusionSourceField.LabelSet("DiffusionSource")
diffusionSourceField.VariableLabelSet(oc.FieldVariableTypes.U,"DiffusionSource")
# Set the source to be element based
diffusionSourceField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
diffusionEquationsSet.SourceCreateFinish()    

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS
#-----------------------------------------------------------------------------------------------------------

diffusionEquations = oc.Equations()
diffusionEquationsSet.EquationsCreateStart(diffusionEquations)
#diffusionEquations.SparsityTypeSet(oc.EquationsSparsityTypes.FULL)
diffusionEquations.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
diffusionEquations.OutputTypeSet(oc.EquationsOutputTypes.NONE)
#diffusionEquations.OutputTypeSet(oc.EquationsOutputTypes.TIMING)
#diffusionEquations.OutputTypeSet(oc.EquationsOutputTypes.MATRIX)
#diffusionEquations.OutputTypeSet(oc.EquationsOutputTypes.ELEMENT_MATRIX)
diffusionEquationsSet.EquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY PROBLEM
#-----------------------------------------------------------------------------------------------------------

elasticityProblem = oc.Problem()
elasticityProblemSpecification = [oc.ProblemClasses.ELASTICITY,
                                  oc.ProblemTypes.LINEAR_ELASTICITY,
                                  oc.ProblemSubtypes.NONE]
elasticityProblem.CreateStart(ELASTICITY_PROBLEM_USER_NUMBER,context,elasticityProblemSpecification)
elasticityProblem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY CONTROL LOOPS
#-----------------------------------------------------------------------------------------------------------

elasticityProblem.ControlLoopCreateStart()
elasticityProblem.ControlLoopCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY SOLVER
#-----------------------------------------------------------------------------------------------------------

# Create problem solver
elasticitySolver = oc.Solver()
elasticityProblem.SolversCreateStart()
elasticityProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,elasticitySolver)
elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.NONE)
#elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.MONITOR)
#elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.PROGRESS)
#elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.TIMING)
#elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.SOLVER)
#elasticitySolver.OutputTypeSet(oc.SolverOutputTypes.MATRIX)
elasticitySolver.LinearTypeSet(oc.LinearSolverTypes.DIRECT)
elasticityProblem.SolversCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY SOLVER EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create solver equations and add equations set to solver equations
elasticitySolver = oc.Solver()
elasticitySolverEquations = oc.SolverEquations()
elasticityProblem.SolverEquationsCreateStart()
elasticityProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,elasticitySolver)
elasticitySolver.SolverEquationsGet(elasticitySolverEquations)
#elasticitySolverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.FULL)
elasticitySolverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.SPARSE)
elasticityEquationsSetIndex = elasticitySolverEquations.EquationsSetAdd(elasticityEquationsSet)
elasticityProblem.SolverEquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY BOUNDARY CONDITIONS
#-----------------------------------------------------------------------------------------------------------

bottomLeftNodeNumber = 1
topLeftNodeNumber = numberOfNodes - numberOfXNodes+1
midRightNodeNumber = math.floor(numberOfYNodes/2)*numberOfXNodes+numberOfXNodes

elasticityBoundaryConditions = oc.BoundaryConditions()
elasticitySolverEquations.BoundaryConditionsCreateStart(elasticityBoundaryConditions)

# Set the bottom left and top left nodes to be fixed
nodeDomain = decomposition.NodeDomainGet(bottomLeftNodeNumber,1)
if (nodeDomain == computationalNodeNumber):
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                         oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,bottomLeftNodeNumber,1,
                                         oc.BoundaryConditionsTypes.FIXED,0.0)
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                         oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,bottomLeftNodeNumber,2,
                                         oc.BoundaryConditionsTypes.FIXED,0.0)
    if (haveHermite):
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,bottomLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,bottomLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,bottomLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,bottomLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,bottomLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,bottomLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)                
    
nodeDomain = decomposition.NodeDomainGet(topLeftNodeNumber,1)
if (nodeDomain == computationalNodeNumber):
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                         oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,topLeftNodeNumber,1,
                                         oc.BoundaryConditionsTypes.FIXED,0.0)
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                         oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,topLeftNodeNumber,2,
                                         oc.BoundaryConditionsTypes.FIXED,0.0)
    if (haveHermite):
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,topLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,topLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,topLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,topLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,topLeftNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,topLeftNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)                
    
# Set the mid right node to have a downward displacement/force 
nodeDomain = decomposition.NodeDomainGet(midRightNodeNumber,1)
if (nodeDomain == computationalNodeNumber):
    if (boundaryConditionType == DIRICHLET_BCS):
        #Set downward displacement on the mid right node
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,midRightNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,midRightNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,MAX_DISPLACEMENT)
    else:
        #Set downward force on the mid right node
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,midRightNodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,midRightNodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,MAX_FORCE)

elasticitySolverEquations.BoundaryConditionsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION PROBLEM
#-----------------------------------------------------------------------------------------------------------

diffusionProblem = oc.Problem()
diffusionProblemSpecification = [oc.ProblemClasses.CLASSICAL_FIELD,
                                 oc.ProblemTypes.DIFFUSION_EQUATION,
                                 oc.ProblemSubtypes.LINEAR_DIFFUSION]
diffusionProblem.CreateStart(DIFFUSION_PROBLEM_USER_NUMBER,context,diffusionProblemSpecification)
diffusionProblem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION CONTROL LOOPS
#-----------------------------------------------------------------------------------------------------------

# Create diffusion control loops
diffusionProblem.ControlLoopCreateStart()
diffusionControlLoop = oc.ControlLoop()
diffusionProblem.ControlLoopGet([oc.ControlLoopIdentifiers.NODE],diffusionControlLoop)
#diffusionControlLoop.OutputTypeSet(oc.ControlLoopOutputTypes.TIMING)
diffusionProblem.ControlLoopCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION SOLVER
#-----------------------------------------------------------------------------------------------------------

# Create problem solver
dynamicSolver = oc.Solver()
diffusionProblem.SolversCreateStart()
diffusionProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,dynamicSolver)
dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.NONE)
#dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.SOLVER)
#dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.MATRIX)
#dynamicSolver.LinearTypeSet(oc.LinearSolverTypes.ITERATIVE)
#dynamicSolver.LinearIterativeAbsoluteToleranceSet(1.0E-12)
#dynamicSolver.LinearIterativeRelativeToleranceSet(1.0E-12)
diffusionProblem.SolversCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION SOLVER EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create diffusion solver equations and add diffusion equations set to the diffusion solver equations
dynamicSolver = oc.Solver()
diffusionSolverEquations = oc.SolverEquations()
diffusionProblem.SolverEquationsCreateStart()
diffusionProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,dynamicSolver)
dynamicSolver.SolverEquationsGet(diffusionSolverEquations)
diffusionSolverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.SPARSE)
diffusionEquationsSetIndex = diffusionSolverEquations.EquationsSetAdd(diffusionEquationsSet)
diffusionProblem.SolverEquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION BOUNDARY CONDITIONS
#-----------------------------------------------------------------------------------------------------------

diffusionBoundaryConditions = oc.BoundaryConditions()
diffusionSolverEquations.BoundaryConditionsCreateStart(diffusionBoundaryConditions)

# Set the value of Phi on the boundary to zero
for yNodeIdx in range(1,numberOfYNodes+1):
    for xNodeIdx in range(1,numberOfXNodes+1):
        nodeNumber = (yNodeIdx-1)*numberOfXNodes+xNodeIdx
        if( ( (yNodeIdx == 1) or (yNodeIdx == numberOfYNodes ) ) or ( (xNodeIdx == 1) or (xNodeIdx == numberOfXNodes) )):
            nodeDomain = decomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                diffusionBoundaryConditions.SetNode(diffusionDependentField,oc.FieldVariableTypes.U,1,
                                                     oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                                     oc.BoundaryConditionsTypes.FIXED,0.0)           
diffusionSolverEquations.BoundaryConditionsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# INITIALISE OPTIMISATION PROBLEM
#-----------------------------------------------------------------------------------------------------------

strSum = 0.0
for yElementIdx in range(1,numberOfGlobalYElements+1):
    for xElementIdx in range(1,numberOfGlobalXElements+1):
        elementNumber = xElementIdx + (yElementIdx-1)*numberOfGlobalXElements
        elementDomain = decomposition.ElementDomainGet(elementNumber)
        if (elementDomain == computationalNodeNumber):
            strValue = diffusionStructureField.ParameterSetGetElementIntg(oc.FieldVariableTypes.U,
                                                                          oc.FieldParameterSetTypes.VALUES,
                                                                          elementNumber,1)
            strSum = strSum + float(strValue)

#TODO: reduce sum across the ranks
initialVolume = strSum/float(numberOfElements)
 
#-----------------------------------------------------------------------------------------------------------
# MAIN LOOP START
#-----------------------------------------------------------------------------------------------------------

iterationNumber = 0
time = TIME_START

continueLoop = True

A1 = (3.0*(1.0-POISSONS_RATIO)*(1.0-14.0*POISSONS_RATIO+15.0*POISSONS_RATIO*POISSONS_RATIO))/(2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO))
A2 = (15.0*YOUNGS_MODULUS*(1.0-POISSONS_RATIO))/(2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO))
C1 = A1+2.0*A2
C2 = A1/C1
A = C1*np.array([[1.0,C2,0.0],
              [C2,1.0,0.0],
              [0.0,0.0,(1.0-C2)/2.0]])

#print(A)

objective = np.array([0.0]*MAXIMUM_NUMBER_OF_ITERATIONS)

while continueLoop:

    iterationNumber = iterationNumber + 1
    time = time + TIME_STEP

    #-----------------------------------------------------------------------------------------------------------
    # ELASTICITY SOLVE
    #-----------------------------------------------------------------------------------------------------------
    
    elasticityProblem.Solve()
    
    # Calculate the derived fields
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.SMALL_STRAIN)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.ELASTIC_WORK)

    elasticitySolution = elasticityDependentField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

    #print(elasticitySolution)

    #-----------------------------------------------------------------------------------------------------------
    # ELASTICITY OPTIMISATION PARAMETERS
    #-----------------------------------------------------------------------------------------------------------
    
    objectiveSum = 0.0
    strainEnergyDensitySum = 0.0
    topologicalDerivativeSum = 0.0
    absTopologicalDerivativeSum = 0.0
    strSum = 0.0
    for yElementIdx in range(1,numberOfGlobalYElements+1):
        for xElementIdx in range(1,numberOfGlobalXElements+1):
            elementNumber = xElementIdx + (yElementIdx-1)*numberOfGlobalXElements
            elementDomain = decomposition.ElementDomainGet(elementNumber)
            if (elementDomain == computationalNodeNumber):
                e11=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                    elementNumber,1)
                e22=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                    elementNumber,2)
                t12=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                    elementNumber,3)
                strainEnergy=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.W,
                                                                             oc.FieldParameterSetTypes.VALUES,
                                                                             elementNumber,1)
                etilde=np.array([[e11,e22,t12]])
                etildeA = np.matmul(etilde,A)
                e=np.array([[e11],
                            [e22],
                            [t12]])
                #print(e)
                etildeAe = np.matmul(etildeA,e)
                energy = etildeAe[0]
                strValue = diffusionStructureField.ParameterSetGetElementIntg(oc.FieldVariableTypes.U,
                                                                              oc.FieldParameterSetTypes.VALUES,
                                                                              elementNumber,1)
                strainEnergyDensity=(YOUNGS_MODULUS+float(strValue)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*strainEnergy
                objectiveSum = objectiveSum+strainEnergy

                topologicalDerivative = (YOUNGS_MODULUS+float(strValue)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*float(energy)

                # Store the topologicalDerivative value in the diffusion source field
                diffusionSourceField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                                 elementNumber,1,topologicalDerivative)

                #print("element = %d, strValue = %d, SED = %f, TD = %f" % (elementNumber,strValue,strainEnergyDensity,topologicalDerivative))
                strSum = strSum + float(strValue)
                strainEnergyDensitySum=strainEnergyDensitySum + strainEnergyDensity
                topologicalDerivativeSum = topologicalDerivativeSum + topologicalDerivative
                absTopologicalDerivativeSum = absTopologicalDerivativeSum + abs(topologicalDerivative)
                objectiveSum = objectiveSum+strainEnergy
                
    #TODO: reduce objective, volume etc. sum across the ranks
    volume = strSum/float(numberOfElements)
    objective[iterationNumber] = objectiveSum
    
    #-----------------------------------------------------------------------------------------------------------
    # CALCULATE AUGMENTED LAGRANGIAN PARAMETERS
    #-----------------------------------------------------------------------------------------------------------

    maximumG = MAX_VOLUME_FRACTION+(initialVolume-MAX_VOLUME_FRACTION)*max(0,1-iterationNumber/N_VOL_ITERATIONS)
    G = volume - maximumG
    lambdaValue = topologicalDerivativeSum/float(numberOfElements)*math.exp(LEVEL_SET_P_PARAM*(G/maximumG+LEVEL_SET_D_PARAM))
    C = float(numberOfElements)/absTopologicalDerivativeSum

    # Update the diffusion source to be C*topologicalDerivative - lambda
    for yElementIdx in range(1,numberOfGlobalYElements+1):
        for xElementIdx in range(1,numberOfGlobalXElements+1):
            elementNumber = xElementIdx + (yElementIdx-1)*numberOfGlobalXElements
            elementDomain = decomposition.ElementDomainGet(elementNumber)
            if (elementDomain == computationalNodeNumber):
                topologicalDerivative = diffusionSourceField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                      elementNumber,1)
                diffusionSourceValue = C*topologicalDerivative - lambdaValue
                diffusionSourceField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                                 elementNumber,1,diffusionSourceValue)
                
    #-----------------------------------------------------------------------------------------------------------
    # DIFFUSION SOLVE
    #-----------------------------------------------------------------------------------------------------------

    diffusionControlLoop.TimesSet(time,time+TIME_STEP,TIME_STEP)
    diffusionProblem.Solve()

    diffusionSolution = diffusionDependentField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

    #print(diffusionSolution)
    
    #-----------------------------------------------------------------------------------------------------------
    # RECALCULATE THE NEW STRUCUTRE FIELD AND VOLUME
    #-----------------------------------------------------------------------------------------------------------

    for yElementIdx in range(1,numberOfGlobalYElements+1):
        for xElementIdx in range(1,numberOfGlobalXElements+1):
            elementNumber = xElementIdx + (yElementIdx-1)*numberOfGlobalXElements
            elementDomain = decomposition.ElementDomainGet(elementNumber)
            if (elementDomain == computationalNodeNumber):
                bottomLeftNodeNumber = (xElementIdx-1)+1+(yElementIdx-1)*numberOfXNodes
                bottomRightNodeNumber = bottomLeftNodeNumber+1
                topLeftNodeNumber = bottomLeftNodeNumber+numberOfXNodes
                topRightNodeNumber = topLeftNodeNumber+1
                bottomLeftPhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                              oc.FieldParameterSetTypes.VALUES,
                                                                              1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                              bottomLeftNodeNumber,1)
                bottomLeftPhi = min(1.0,max(-1.0,bottomLeftPhi))
                diffusionDependentField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 bottomLeftNodeNumber,1,bottomLeftPhi)
                bottomRightPhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                               1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                               bottomRightNodeNumber,1)
                bottomRightPhi = min(1.0,max(-1.0,bottomRightPhi))                
                diffusionDependentField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 bottomLeftNodeNumber,1,bottomRightPhi)
                topLeftPhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                           topLeftNodeNumber,1)
                topLeftPhi = min(1.0,max(-1.0,topLeftPhi))                
                diffusionDependentField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 bottomLeftNodeNumber,1,topLeftPhi)
                topRightPhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                            oc.FieldParameterSetTypes.VALUES,
                                                                            1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                            topRightNodeNumber,1)
                topRightPhi = min(1.0,max(-1.0,topRightPhi))                
                diffusionDependentField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 bottomLeftNodeNumber,1,topRightPhi)
                averagePhi = (bottomLeftPhi + bottomRightPhi + topLeftPhi + topRightPhi)/4.0
                if(averagePhi <= PHI_ZERO_TOLERANCE):
                    diffusionStructureField.ParameterSetUpdateElementIntg(oc.FieldVariableTypes.U,
                                                                          oc.FieldParameterSetTypes.VALUES,
                                                                          elementNumber,1,0)
                    elasticityMaterialsField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,
                                                                         oc.FieldParameterSetTypes.VALUES,
                                                                         elementNumber,1,YOUNGS_MODULUS_MIN)
                 
    #-----------------------------------------------------------------------------------------------------------
    # OUTPUT
    #-----------------------------------------------------------------------------------------------------------

    filenameFormat = "Bone_{Iteration:0d}"
    filename = filenameFormat.format(Iteration=iterationNumber)
    fields = oc.Fields()
    fields.CreateRegion(region)
    fields.NodesExport(filename,"FORTRAN")
    fields.ElementsExport(filename,"FORTRAN")
    fields.Finalise()
    
    #-----------------------------------------------------------------------------------------------------------
    # STATISTICS AND CHECK CONVERGENCE
    #-----------------------------------------------------------------------------------------------------------

    print("Iteration = %d, Objective = %f, Volume ratio = %f, lambda = %f" % (iterationNumber,objective[iterationNumber]/float(numberOfElements),volume,lambdaValue))
    
    if( (iterationNumber>=MAXIMUM_NUMBER_OF_ITERATIONS) ):
        continueLoop = False

#-----------------------------------------------------------------------------------------------------------
# MAIN LOOP END
#-----------------------------------------------------------------------------------------------------------

endwhile

# Finalise OpenCMISS
oc.Finalise()

#!/usr/bin/env python
#
# This is an example script for a bone optimisation problem using OpenCMISS calls in python.
# By Chris Bradley
#
# See Masaki Otomore, Takayuki Yamada, Kazuhiro Izui, and Shinji Nishiwaki, 2015, "Matlab code for
# a level set-based topology optimization method using a reaction diffusion equation", Struct.
# Multidisc. Optim., 51:1159-1172. DOI:10.1007/s00158-014-1190-z
#

import sys
import os
import math
import numpy as np
from mpi4py import MPI
import meshio

# Intialise OpenCMISS
from opencmiss.opencmiss import OpenCMISS_Python as oc
 
def OutputFields(filename):

    # Output .ex files
    fields = oc.Fields()
    fields.CreateRegion(region)
    fields.NodesExport(filename,"FORTRAN")
    fields.ElementsExport(filename,"FORTRAN")
    fields.Finalise()

    # Setup .vtk output
    #outputMesh = outputRegion.MeshGet(outputRegion,MESH_USER_NUMBER)
    outputNodes = oc.MeshNodes()
    mesh.NodesGet(1,outputNodes)
    outputNumberOfNodes = outputNodes.NumberOfNodesGet()
    outputNumberOfDimensions = coordinateSystem.DimensionGet()

    outputNumberOfNodeComponents = outputNumberOfDimensions*2

    outputMeshElements = oc.MeshElements()
    elementBasis = oc.Basis()
    outputNumberOfElements = mesh.NumberOfElementsGet()
    mesh.ElementsGet(1,outputMeshElements)
    outputNumberOfElementComponents = 5+2*numberOfVoigtComponents+2+3
    
    outputFileName = filename + "_solution.vtk"
    
    nodesList = [
        [0 for componentIdx in range(0,outputNumberOfNodeComponents)] for nodeIdx in range(0,outputNumberOfNodes)
    ]
    elementsList = [
        [0 for componentIdx in range(0,outputNumberOfElementComponents)] for elementIdx in range(0,outputNumberOfElements)
    ]

    # Get node data
    for nodeIdx in range(0, outputNumberOfNodes):
        nodeNumber = nodeIdx + 1
        nodeDomain = decomposition.NodeDomainGet(1,nodeNumber)
        if (nodeDomain == computationalNodeNumber):
            nodeGeometryX = geometricField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 nodeNumber,1)
            nodeGeometryY = geometricField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 nodeNumber,2) 
            nodeGeometryZ = geometricField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                 nodeNumber,3)
            nodeDisplacementX = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                               1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                               nodeNumber,1)
            nodeDisplacementY = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                               1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                               nodeNumber,2)
            nodeDisplacementZ = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                               1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                               nodeNumber,3)
            nodeTractionX = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.T,
                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                           1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                           nodeNumber,1)
            nodeTractionY = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.T,
                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                           1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                           nodeNumber,2)
            nodeTractionZ = elasticityDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.T,
                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                           1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                           nodeNumber,3)
            nodePhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                    1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                    nodeNumber,1)
            nodeDiffusionSource = diffusionSourceField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,
                                                                             oc.FieldParameterSetTypes.VALUES,
                                                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                             nodeNumber,1)
            nodeTDN = tdField.ParameterSetGetNodeDP(oc.FieldVariableTypes.V,
                                                    oc.FieldParameterSetTypes.VALUES,
                                                    1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                    nodeNumber,1)             
            nodesList[nodeIdx] = [
                nodeNumber,
                nodeGeometryX,
                nodeGeometryY,
                nodeGeometryZ,
                nodeDisplacementX,
                nodeDisplacementY,
                nodeDisplacementZ,
                nodeTractionX,
                nodeTractionY,
                nodeTractionZ,
                nodePhi,
                nodeDiffusionSource,
                nodeTDN
            ]
            
    # Get element data
    for elementIdx in range(0, outputNumberOfElements):
        elementNumber = elementIdx + 1
        elementDomain = decomposition.ElementDomainGet(elementNumber)
        if (elementDomain == computationalNodeNumber):
            outputMeshElements.BasisGet(elementNumber,elementBasis)
            numberOfLocalNodes = elementBasis.NumberOfLocalNodesGet()
            elementNodes = outputMeshElements.NodesGet(elementNumber,numberOfLocalNodes)
            elementYoungsModulus = elasticityMaterialsField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                     oc.FieldParameterSetTypes.VALUES,
                                                                                     elementNumber,1)
            elementPoissonsRatio = elasticityMaterialsField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                      oc.FieldParameterSetTypes.VALUES,
                                                                                      elementNumber,2)
            elementCauchyStress11 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt11Component)
            elementCauchyStress22 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt22Component)
            elementCauchyStress33 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt33Component)
            elementCauchyStress12 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt12Component)
            elementCauchyStress13 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt13Component)
            elementCauchyStress23 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt23Component)
            elementSmallStrain11 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt11Component)
            elementSmallStrain22 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt22Component)
            elementSmallStrain33 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt33Component)
            elementSmallStrain12 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt12Component)
            elementSmallStrain13 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt13Component)
            elementSmallStrain23 = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.V,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    elementNumber,voigt23Component)
            elementElasticWork = elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.W,
                                                                                 oc.FieldParameterSetTypes.VALUES,
                                                                                 elementNumber,1)
            elementStructure = structureField.ParameterSetGetElementIntg(oc.FieldVariableTypes.U,
                                                                         oc.FieldParameterSetTypes.VALUES,
                                                                         elementNumber,1)
            elementSED = sedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                           oc.FieldParameterSetTypes.VALUES,
                                                           elementNumber,1)
            elementTD = tdField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                         oc.FieldParameterSetTypes.VALUES,
                                                         elementNumber,1)
            elementsList[elementIdx] = [
                elementNumber,
                elementNodes[0],
                elementNodes[1],
                elementNodes[2],
                elementNodes[3],
                elementCauchyStress11,
                elementCauchyStress22,
                elementCauchyStress33,
                elementCauchyStress23,
                elementCauchyStress13,
                elementCauchyStress12,
                elementSmallStrain11,
                elementSmallStrain22,
                elementSmallStrain33,
                elementSmallStrain23,
                elementSmallStrain13,
                elementSmallStrain12,
                elementElasticWork,
                elementYoungsModulus,
                elementPoissonsRatio,
                elementStructure,
                elementSED,
                elementTD
            ]
            
    nodesList = np.array(nodesList)
    elementsList = np.array(elementsList)
    
    points = np.array(nodesList[:, 1:4])
    cells = [("tetra", np.array(elementsList)[:, 1:5] - 1)]

    # Get values
    position = nodesList[:,1:4]
    displacement = nodesList[:,4:7]
    traction = nodesList[:,7:10]
    phi = nodesList[:,10]
    diffusionSource = nodesList[:,11]
    tdn = nodesList[:,12]

    cauchyStress11 = elementsList[:,5]
    cauchyStress22 = elementsList[:,6]
    cauchyStress33 = elementsList[:,7]
    cauchyStress23 = elementsList[:,8]
    cauchyStress13 = elementsList[:,9]
    cauchyStress12 = elementsList[:,10]
    smallStrain11 = elementsList[:,11]
    smallStrain22 = elementsList[:,12]
    smallStrain33 = elementsList[:,13]
    smallStrain23 = elementsList[:,14]
    smallStrain13 = elementsList[:,15]
    smallStrain12 = elementsList[:,16]
    elasticWork = elementsList[:,17]
    youngsModulus = elementsList[:,18]
    poissonsRatio = elementsList[:,19]
    structure = elementsList[:,20]
    sed = elementsList[:,21]
    td = elementsList[:,22]

    # Write solution mesh
    solutionMesh = meshio.Mesh(points,cells)
    solutionMesh.point_data = {
        "Position": position,
        "Displacement": displacement,
        "Traction": traction,
        "Phi": phi,
        "DiffusionSource": diffusionSource,
        "TDN": tdn,
    }
    solutionMesh.cell_data = {
        "CauchyStress11": cauchyStress11,
        "CauchyStress22": cauchyStress22,
        "CauchyStress33": cauchyStress33,
        "CauchyStress23": cauchyStress23,
        "CauchyStress13": cauchyStress13,
        "CauchyStress12": cauchyStress12,
        "SmallStrain11": smallStrain11,
        "SmallStrain22": smallStrain22,
        "SmallStrain33": smallStrain33,
        "SmallStrain23": smallStrain23,
        "SmallStrain13": smallStrain13,
        "SmallStrain12": smallStrain12,
        "ElasticWork": elasticWork,
        "YoungsModulus": youngsModulus,
        "PoissonsRatio": poissonsRatio,
        "Structure": structure,
        "SED": sed,
        "TD": td,
    }
    meshio.write(outputFileName,solutionMesh)
        
    displacedPoints = position + displacement*SCALE_DISPLACEMENT
    outputFileNameDisplaced = filename + "_displaced.vtk"
    solutionMesh.points = displacedPoints
    meshio.write(outputFileNameDisplaced,solutionMesh)    

#-----------------------------------------------------------------------------------------------------------
# SET PROBLEM PARAMETERS
#-----------------------------------------------------------------------------------------------------------

# Boundary condition 
DOWNWARD_FORCE = 10.0 # N.mm^-2
DIRICHLET_VECTOR = -5.0

# Output parameters
SCALE_DISPLACEMENT = 1.0e3

# Elasticity parameters
YOUNGS_MODULUS = 30.0e6 # mg.mm^-1.ms^-2
YOUNGS_MODULUS_MIN = 0.000001 # mg.mm^-1.ms^-2
POISSONS_RATIO = 0.3
THICKNESS = 1.0 # mm (for plane strain and stress)

# Diffusion parameters
DIFFUSION_A_PARAM = 1.0
DIFFUSION_TAU_PARAM = 0.001 # Stabilisation parameter

# Optimisation parameters
MAX_VOLUME_RATIO = 0.50
LEVEL_SET_P_PARAM = 4
LEVEL_SET_D_PARAM = -0.02
N_VOL_ITERATIONS = 100

# Time information
TIME_START = 0.00
TIME_STEP = 0.05

#MAXIMUM_NUMBER_OF_ITERATIONS = 10 # Maximum number of iterations in the main loop
MAXIMUM_NUMBER_OF_ITERATIONS = 200 # Maximum number of iterations in the main loop

DEBUG = True
#DEBUG = False

# Generic parameters

LINEAR_LAGRANGE = 1
QUADRATIC_LAGRANGE = 2
CUBIC_LAGRANGE = 3
CUBIC_HERMITE = 4
LINEAR_SIMPLEX = 5
QUADRATIC_SIMPLEX = 6
CUBIC_SIMPLEX = 7

# User numbers
(CONTEXT_USER_NUMBER,
 COORDINATE_SYSTEM_USER_NUMBER,
 REGION_USER_NUMBER,
 BASIS_USER_NUMBER,
 MESH_USER_NUMBER,
 DECOMPOSITION_USER_NUMBER,
 DECOMPOSER_USER_NUMBER,
 GEOMETRIC_FIELD_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,
 ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,
 ELASTICITY_MATERIALS_FIELD_USER_NUMBER,
 ELASTICITY_DERIVED_FIELD_USER_NUMBER,
 DIFFUSION_EQUATIONS_SET_USER_NUMBER,
 DIFFUSION_EQUATIONS_SET_FIELD_USER_NUMBER,
 DIFFUSION_DEPENDENT_FIELD_USER_NUMBER,
 DIFFUSION_MATERIALS_FIELD_USER_NUMBER,
 DIFFUSION_SOURCE_FIELD_USER_NUMBER,
 STRUCTURE_FIELD_USER_NUMBER,
 SED_FIELD_USER_NUMBER,
 TD_FIELD_USER_NUMBER,
 ELASTICITY_PROBLEM_USER_NUMBER,
 DIFFUSION_PROBLEM_USER_NUMBER
 ) = range(1,24)

baseFileName = 'adapted_aligned_QA_approved_volumetric_mesh+1'

# Override defaults with command line arguments if need be
if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        sys.exit('ERROR: too many arguments- currently only accepting up to 6 options: meshfilename')
    baseFileName = int(sys.argv[1])


inputFileName = baseFileName + ".mesh"
outputFileName = baseFileName + "_solution.vtk"
dirichletFileName = baseFileName + "_dirichlet_BC.npy"
neumannFileName = baseFileName + "_neumann_BC.npy"

NUMBER_OF_DIMENSIONS = 3

INTERPOLATION_TYPE = LINEAR_SIMPLEX

if (INTERPOLATION_TYPE == LINEAR_LAGRANGE):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE
    NUMBER_OF_NODES_XI = 2
    NUMBER_OF_GAUSS_XI = 2
elif (INTERPOLATION_TYPE == QUADRATIC_LAGRANGE):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE
    NUMBER_OF_NODES_XI = 3
    NUMBER_OF_GAUSS_XI = 3
elif (INTERPOLATION_TYPE == CUBIC_LAGRANGE):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE
    NUMBER_OF_NODES_XI = 4
    NUMBER_OF_GAUSS_XI = 4
elif (INTERPOLATION_TYPE == CUBIC_HERMITE):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.CUBIC_HERMITE
    NUMBER_OF_NODES_XI = 2
    NUMBER_OF_GAUSS_XI = 4
elif (INTERPOLATION_TYPE == LINEAR_SIMPLEX):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX
    NUMBER_OF_NODES_XI = 2
    GAUSS_ORDER = 4
elif (INTERPOLATION_TYPE == QUADRATIC_SIMPLEX):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX
    NUMBER_OF_NODES_XI = 3
    GAUSS_ORDER = 4
elif (INTERPOLATION_TYPE == CUBIC_SIMPLEX):
    INTERPOLATION_TYPE_XI = oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX
    NUMBER_OF_NODES_XI = 4
    GAUSS_ORDER = 5
else:
    sys.exit('The interpolation type of ',INTERPOLATION_TYPE,' is invalid.')

HAVE_HERMITE = (INTERPOLATION_TYPE == CUBIC_HERMITE)
HAVE_SIMPLEX = (INTERPOLATION_TYPE == LINEAR_SIMPLEX or
                INTERPOLATION_TYPE == QUADRATIC_SIMPLEX or
                INTERPOLATION_TYPE == CUBIC_SIMPLEX)

NUMBER_OF_XI = NUMBER_OF_DIMENSIONS
if (not HAVE_SIMPLEX):
    NUMBER_OF_GAUSS = pow(NUMBER_OF_GAUSS_XI,NUMBER_OF_XI)


         
#-----------------------------------------------------------------------------------------------------------
# READ MESH ETC. FILES
#-----------------------------------------------------------------------------------------------------------

inputMesh = meshio.read(inputFileName)

inputCoords = inputMesh.points
inputNodes = range(1, len(inputCoords) + 1)
NUMBER_OF_NODES = len(inputCoords)

inputElementNodes = inputMesh.cells_dict["tetra"] + 1
inputElements = range(1, len(inputMesh.cells_dict["tetra"]) + 1)
NUMBER_OF_ELEMENTS = len(inputElements)
                     

print("Mesh loaded:")
print("  Filename : ",inputFileName)
print("  Number of vertices : ",NUMBER_OF_NODES)
print("  Number of elements : ",NUMBER_OF_ELEMENTS)

# Load BC files

dirichletNodes = np.load(dirichletFileName) + 1
NUMBER_OF_DIRICHLET = len(dirichletNodes)

neumannNodes = np.load(neumannFileName) + 1
NUMBER_OF_NEUMANN = len(neumannNodes)

print("BCs loaded:")
print("  Dirichlet BCs:")
print("    Filename : ",dirichletFileName)
print("    Number of Dirichlet BCs : ",NUMBER_OF_DIRICHLET)
print("  Neumann BCs:")
print("    Filename : ",neumannFileName)
print("    Number of Neumann BCs : ",NUMBER_OF_NEUMANN)

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

#oc.DiagnosticsSetOn(oc.DiagnosticTypes.ALL,[1,2,3,4,5],"Diagnostics",["LinearElasticity_StrainMatrixCalculateGauss"])

# Get the computational nodes information
computationEnvironment = oc.ComputationEnvironment()
context.ComputationEnvironmentGet(computationEnvironment)
numberOfComputationalNodes = computationEnvironment.NumberOfWorldNodesGet()
computationalNodeNumber = computationEnvironment.WorldNodeNumberGet()

worldWorkGroup = oc.WorkGroup()
computationEnvironment.WorldWorkGroupGet(worldWorkGroup)

#-----------------------------------------------------------------------------------------------------------
# OTHER CONSTANTS
#-----------------------------------------------------------------------------------------------------------

numberOfVoigtComponents = oc.NumberOfVoigtComponentsGet(NUMBER_OF_DIMENSIONS)
voigt11Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,1)
voigt12Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,2)
voigt22Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,2)
if (NUMBER_OF_DIMENSIONS == 3):
    voigt13Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,3)
    voigt23Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,3)
    voigt33Component = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,3,3)    

#-----------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
#-----------------------------------------------------------------------------------------------------------

coordinateSystem = oc.CoordinateSystem()
coordinateSystem.CreateStart(COORDINATE_SYSTEM_USER_NUMBER,context)
coordinateSystem.DimensionSet(NUMBER_OF_DIMENSIONS)
coordinateSystem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# REGION
#-----------------------------------------------------------------------------------------------------------

region = oc.Region()
region.CreateStart(REGION_USER_NUMBER,worldRegion)
region.LabelSet("BoneOptimisation")
region.CoordinateSystemSet(coordinateSystem)
region.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# BASIS
#-----------------------------------------------------------------------------------------------------------

basis = oc.Basis()
basis.CreateStart(BASIS_USER_NUMBER,context)
if (HAVE_SIMPLEX):
    basis.TypeSet(oc.BasisTypes.SIMPLEX)
else:
    basis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
basis.NumberOfXiSet(NUMBER_OF_XI)
basis.InterpolationXiSet([INTERPOLATION_TYPE_XI]*NUMBER_OF_XI)
if (HAVE_SIMPLEX):
    basis.QuadratureOrderSet(GAUSS_ORDER)
else:
    basis.QuadratureNumberOfGaussXiSet([NUMBER_OF_GAUSS_XI]*NUMBER_OF_XI)
basis.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# MESH
#-----------------------------------------------------------------------------------------------------------

nodes = oc.Nodes()
nodes.CreateStart(region,NUMBER_OF_NODES)
nodes.CreateFinish()

mesh = oc.Mesh()
mesh.CreateStart(MESH_USER_NUMBER,region,NUMBER_OF_DIMENSIONS)
mesh.NumberOfElementsSet(NUMBER_OF_ELEMENTS)
mesh.NumberOfComponentsSet(1)
meshElements = oc.MeshElements()

meshElements.CreateStart(mesh,1,basis)

for elementIdx in inputElements:
    localNodes = np.array(inputElementNodes[elementIdx - 1],dtype=np.int32)
    meshElements.NodesSet(elementIdx,localNodes)

meshElements.CreateFinish()
mesh.CreateFinish()
 
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

numberOfLocalElements = decomposition.NumberOfLocalElementsGet()
numberOfLocalNodes = decomposition.NumberOfLocalNodesGet(1)

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
if (NUMBER_OF_DIMENSIONS == 3):
    geometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,3,1)
geometricField.ScalingTypeSet(oc.FieldScalingTypes.ARITHMETIC_MEAN)
geometricField.CreateFinish()

# Set geometry
for nodeIdx in range(1,NUMBER_OF_NODES+1):
    nodeDomain = decomposition.NodeDomainGet(1,nodeIdx)
    if (nodeDomain == computationalNodeNumber):
        x = inputCoords[nodeIdx - 1,0].item()
        y = inputCoords[nodeIdx - 1,1].item()
        geometricField.ParameterSetUpdateNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                              1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeIdx,1,x)
        geometricField.ParameterSetUpdateNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                              1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeIdx,2,y)
        if(NUMBER_OF_DIMENSIONS==3):
            z = inputCoords[nodeIdx - 1,2].item()
            geometricField.ParameterSetUpdateNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                  1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeIdx,3,z)

# Update the geometric field            
geometricField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

#-----------------------------------------------------------------------------------------------------------
# ELASTITICY EQUATION SETS
#-----------------------------------------------------------------------------------------------------------

# Create linear elasiticity equations set
elasticityEquationsSetField = oc.Field()
elasticityEquationsSet = oc.EquationsSet()
if (NUMBER_OF_DIMENSIONS == 2):
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
if (NUMBER_OF_DIMENSIONS == 2):
    elasticityMaterialsField.ComponentInterpolationSet(oc.FieldVariableTypes.U,2,oc.FieldInterpolationTypes.CONSTANT)    
    elasticityMaterialsField.ComponentInterpolationSet(oc.FieldVariableTypes.U,3,oc.FieldInterpolationTypes.CONSTANT)
else:
    elasticityMaterialsField.ComponentInterpolationSet(oc.FieldVariableTypes.U,2,oc.FieldInterpolationTypes.ELEMENT_BASED)
elasticityEquationsSet.MaterialsCreateFinish()

# Initialise the materials field values
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1,YOUNGS_MODULUS)
elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   2,POISSONS_RATIO)
if (NUMBER_OF_DIMENSIONS == 2):
    elasticityMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                       3,THICKNESS)

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS SET DERIVED
#-----------------------------------------------------------------------------------------------------------

# Create a field for the derived field. Have three variables U - Small strain tensor, V - Cauchy stress, W - Elastic Work
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
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U,numberOfVoigtComponents)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.V,numberOfVoigtComponents)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.W,1)
for componentIdx in range(1,numberOfVoigtComponents+1):
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,1)
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.V,componentIdx,1)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U,componentIdx,
                                                     oc.FieldInterpolationTypes.ELEMENT_BASED)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.V,componentIdx,
                                                     oc.FieldInterpolationTypes.ELEMENT_BASED)
elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.W,1,1)
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
# DIFFUSION DEPENDENT FIELD
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
# DIFFUSION EQUATIONS SET MATERIALS
#-----------------------------------------------------------------------------------------------------------

diffusionMaterialsField = oc.Field()
diffusionEquationsSet.MaterialsCreateStart(DIFFUSION_MATERIALS_FIELD_USER_NUMBER,diffusionMaterialsField)
diffusionMaterialsField.LabelSet("DiffusionMaterials")
diffusionMaterialsField.VariableLabelSet(oc.FieldVariableTypes.U,"DiffusionMaterials")
diffusionEquationsSet.MaterialsCreateFinish()    
# Initialise the diffusion materials field values, a.del u/del t + div(sigma.grad u) + s = 0
diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1,DIFFUSION_A_PARAM)
diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1+voigt11Component,-DIFFUSION_TAU_PARAM)
diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1+voigt22Component,-DIFFUSION_TAU_PARAM)
diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                   1+voigt12Component,0.0)
if (NUMBER_OF_DIMENSIONS == 3):
    diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                      1+voigt33Component,-DIFFUSION_TAU_PARAM)
    diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                      1+voigt13Component,0.0)
    diffusionMaterialsField.ComponentValuesInitialise(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                      1+voigt23Component,0.0)

#-----------------------------------------------------------------------------------------------------------
# DIFFUSION EQUATIONS SET SOURCE
#-----------------------------------------------------------------------------------------------------------

diffusionSourceField = oc.Field()
diffusionEquationsSet.SourceCreateStart(DIFFUSION_SOURCE_FIELD_USER_NUMBER,diffusionSourceField)
diffusionSourceField.LabelSet("DiffusionSource")
diffusionSourceField.VariableLabelSet(oc.FieldVariableTypes.U,"DiffusionSource")
# Set the source to be element based
diffusionSourceField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.NODE_BASED)
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
#elasticitySolver.LinearTypeSet(oc.LinearSolverTypes.ITERATIVE)
#elasticitySolver.LinearIterativeMaximumIterationsSet(1000000)
#elasticitySolver.LinearIterativeGMRESRestartSet(NUMBER_OF_NODES)
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

elasticityBoundaryConditions = oc.BoundaryConditions()
elasticitySolverEquations.BoundaryConditionsCreateStart(elasticityBoundaryConditions)

# Set Dirichlet BCs

dirichletNodesBC = []
for nodeIdx in dirichletNodes:
    nodeNumber = int(nodeIdx)
    nodeDomain = decomposition.NodeDomainGet(1,nodeNumber)
    if (nodeDomain == computationalNodeNumber):
        # Fix the node in x, y (& z)
        if(DEBUG):
            print("Setting a no displacement boundary condition for node ",nodeNumber)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        if (NUMBER_OF_DIMENSIONS==3):
            elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.U,1,
                                                 oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
        # Show a vector in the x direction where nodes are fixed
        dirichletNodesBC.append([nodeNumber,DIRICHLET_VECTOR,0.0,0.0])
dirichletNodesBC = np.array(dirichletNodesBC)

# Set Neumann BCs

neumannNodesBC = []
for nodeIdx in neumannNodes:
    nodeNumber = int(nodeIdx)
    nodeDomain = decomposition.NodeDomainGet(1,nodeNumber)
    if (nodeDomain == computationalNodeNumber):
        # Set downward force at the node in the y direction
        if(DEBUG):
            print("Setting a downward force boundary condition for node ",nodeNumber)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
        elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,1,
                                             oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,DOWNWARD_FORCE)
        if (NUMBER_OF_DIMENSIONS==3):
            elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,1,
                                                 oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
        # Show a vector in the x direction where nodes are fixed
        neumannNodesBC.append([nodeNumber,0.0,DOWNWARD_FORCE,0.0])
neumannNodesBC = np.array(neumannNodesBC)

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
linearSolver = oc.Solver()
diffusionProblem.SolversCreateStart()
diffusionProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,dynamicSolver)
dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.NONE)
#dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.PROGRESS)
#dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.SOLVER)
#dynamicSolver.OutputTypeSet(oc.SolverOutputTypes.MATRIX)
dynamicSolver.DynamicLinearSolverGet(linearSolver)
linearSolver.LinearTypeSet(oc.LinearSolverTypes.DIRECT)
#linearSolver.LinearTypeSet(oc.LinearSolverTypes.ITERATIVE)
#linearSolver.LinearIterativeMaximumIterationsSet(1000000)
#linearSolver.LinearIterativeGMRESRestartSet(NUMBER_OF_NODES)
#linearSolver.LinearIterativeAbsoluteToleranceSet(1.0E-12)
#linearSolver.LinearIterativeRelativeToleranceSet(1.0E-12)
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
for nodeIdx in range(1,numberOfLocalNodes+1):
    nodeNumber = decomposition.NodeNumberGet(1,nodeIdx)
    onBoundary = decomposition.NodeOnBoundaryGet(1,nodeNumber)
    if (onBoundary):
        diffusionBoundaryConditions.SetNode(diffusionDependentField,oc.FieldVariableTypes.U,1,
                                            oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                            oc.BoundaryConditionsTypes.FIXED,0.0)
                
diffusionSolverEquations.BoundaryConditionsCreateFinish()
                
#-----------------------------------------------------------------------------------------------------------
# STRUCTURE FIELD
#-----------------------------------------------------------------------------------------------------------

structureField = oc.Field()
structureField.CreateStart(STRUCTURE_FIELD_USER_NUMBER,region)
# Set the type
structureField.TypeSet(oc.FieldTypes.GENERAL)
# Set the decomposition
structureField.DecompositionSet(decomposition)
# Set the geometric field
structureField.GeometricFieldSet(geometricField)
# Set the label
structureField.LabelSet("Structure")
# Set the variables
structureField.NumberOfVariablesSet(1)
structureField.VariableTypesSet([oc.FieldVariableTypes.U])
structureField.VariableLabelSet(oc.FieldVariableTypes.U,"Str")
structureField.DataTypeSet(oc.FieldVariableTypes.U,oc.FieldDataTypes.INTG)
# Set the components
structureField.NumberOfComponentsSet(oc.FieldVariableTypes.U,1)
structureField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,1,1)
structureField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
# Finish the field
structureField.CreateFinish()

# Initialise the structure field to 1 (all elements in the structure). If you wish to start with holes set the hole
# element numbers to 0.
structureField.ComponentValuesInitialiseIntg(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,1)

#-----------------------------------------------------------------------------------------------------------
# SED FIELD
#-----------------------------------------------------------------------------------------------------------

sedField = oc.Field()
sedField.CreateStart(SED_FIELD_USER_NUMBER,region)
sedField.LabelSet("StrainEnergyDensity")
sedField.TypeSet(oc.FieldTypes.GENERAL)
sedField.DecompositionSet(decomposition)
sedField.GeometricFieldSet(geometricField)
sedField.DependentTypeSet(oc.FieldDependentTypes.DEPENDENT)
sedField.NumberOfVariablesSet(1)
sedField.VariableTypesSet([oc.FieldVariableTypes.U])
sedField.VariableLabelSet(oc.FieldVariableTypes.U,"SED")
sedField.NumberOfComponentsSet(oc.FieldVariableTypes.U,1)
sedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,1,1)
sedField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
sedField.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# TOPOLOGICAL DERIVATIVE FIELD
#-----------------------------------------------------------------------------------------------------------

tdField = oc.Field()
tdField.CreateStart(TD_FIELD_USER_NUMBER,region)
tdField.LabelSet("TopologicalDerivative")
tdField.TypeSet(oc.FieldTypes.GENERAL)
tdField.DecompositionSet(decomposition)
tdField.GeometricFieldSet(geometricField)
tdField.DependentTypeSet(oc.FieldDependentTypes.DEPENDENT)
tdField.NumberOfVariablesSet(2)
tdField.VariableTypesSet([oc.FieldVariableTypes.U,oc.FieldVariableTypes.V])
tdField.VariableLabelSet(oc.FieldVariableTypes.U,"TD")
tdField.VariableLabelSet(oc.FieldVariableTypes.V,"TDN")
tdField.NumberOfComponentsSet(oc.FieldVariableTypes.U,1)
tdField.NumberOfComponentsSet(oc.FieldVariableTypes.V,1)
tdField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,1,1)
tdField.ComponentMeshComponentSet(oc.FieldVariableTypes.V,1,1)
tdField.ComponentInterpolationSet(oc.FieldVariableTypes.U,1,oc.FieldInterpolationTypes.ELEMENT_BASED)
tdField.ComponentInterpolationSet(oc.FieldVariableTypes.V,1,oc.FieldInterpolationTypes.NODE_BASED)
tdField.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY AND DIFFUSION MAIN WORKFLOW
#-----------------------------------------------------------------------------------------------------------

OutputFields("BoneOptimisation_0")

# Initialise the structural sum and volume
rankStrSum = 0.0
for elementIdx in range(1,numberOfLocalElements+1):
    elementNumber = decomposition.ElementNumberGet(elementIdx)
    strValue = structureField.ParameterSetGetElementIntg(oc.FieldVariableTypes.U,
                                                         oc.FieldParameterSetTypes.VALUES,
                                                         elementNumber,1)
    rankStrSum = rankStrSum + float(strValue)

# Reduce sum across the ranks
strSum = MPI.COMM_WORLD.allreduce(rankStrSum,op=MPI.SUM)

initialVolume = strSum/float(NUMBER_OF_ELEMENTS)
 
iterationNumber = 0
time = TIME_START

continueLoop = True

#Topological derivative constants 
A1 = -(3.0*(1.0-POISSONS_RATIO)*(1.0-14.0*POISSONS_RATIO+15.0*POISSONS_RATIO*POISSONS_RATIO))*YOUNGS_MODULUS/ \
    (2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO)*(1.0-2.0*POISSONS_RATIO))
A2 = (15.0*YOUNGS_MODULUS*(1.0-POISSONS_RATIO))/(2.0*(1.0+POISSONS_RATIO)*(7.0-5.0*POISSONS_RATIO))
C1 = A1+2.0*A2
C2 = A1/C1

if (NUMBER_OF_DIMENSIONS == 2):
    A = np.array([[C1,A1,0.0],
                  [A1,C1,0.0],
                  [0.0,0.0,C1*(1.0-C2)/2.0]])
else:
    A = np.array([[C1,A1,A1,0.0,0.0,0.0],
                  [A1,C1,A1,0.0,0.0,0.0],
                  [A1,A1,C1,0.0,0.0,0.0],
                  [0.0,0.0,0.0,A2,0.0,0.0],
                  [0.0,0.0,0.0,0.0,A2,0.0],
                  [0.0,0.0,0.0,0.0,0.0,A2]])

    
#print("A1 = ",A1)
#print("A2 = ",A2)
#print("C1 = ",C1)
#print("C2 = ",C2)
#print("A  = ",A)

#print(A)

objective = np.array([0.0]*(MAXIMUM_NUMBER_OF_ITERATIONS+1)) #1-indexed

diffusionValues = diffusionDependentField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
structureValues = structureField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
youngsModulusValues = elasticityMaterialsField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

#-----------------------------------------------------------------------------------------------------------
# MAIN LOOP START
#-----------------------------------------------------------------------------------------------------------
     
while continueLoop:

    iterationNumber = iterationNumber + 1
    time = time + TIME_STEP

    print("")
    print("Iteration = ",iterationNumber)

    #-----------------------------------------------------------------------------------------------------------
    # ELASTICITY SOLVE
    #-----------------------------------------------------------------------------------------------------------

    elasticityProblem.Solve()
    
    elasticitySolution = elasticityDependentField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

    # Calculate the derived fields
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.SMALL_STRAIN)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.ELASTIC_WORK)
 
    #strainSolution = elasticityDerivedField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    #stressSolution = elasticityDerivedField.ParameterSetDataGet(oc.FieldVariableTypes.V,oc.FieldParameterSetTypes.VALUES)
    #workSolution = elasticityDerivedField.ParameterSetDataGet(oc.FieldVariableTypes.W,oc.FieldParameterSetTypes.VALUES)
     
    #-----------------------------------------------------------------------------------------------------------
    # ELASTICITY OPTIMISATION PARAMETERS
    #-----------------------------------------------------------------------------------------------------------
    
    rankObjectiveSum = 0.0
    rankSEDSum = 0.0
    rankStrSum = 0.0
    for elementIdx in range(1,numberOfLocalElements+1):
        elementNumber = decomposition.ElementNumberGet(elementIdx)

        #print("Element : ",elementNumber)
                
        e11=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                            oc.FieldParameterSetTypes.VALUES,
                                                            elementNumber,voigt11Component)
        e22=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                            oc.FieldParameterSetTypes.VALUES,
                                                            elementNumber,voigt22Component)
        e12=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                            oc.FieldParameterSetTypes.VALUES,
                                                            elementNumber,voigt12Component)
        if (NUMBER_OF_DIMENSIONS == 2):
            eT=np.array([[e11,e22,e12]])
            e=np.array([[e11],
                        [e22],
                        [e12]])
        else:
            e13=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                oc.FieldParameterSetTypes.VALUES,
                                                                elementNumber,voigt13Component)
            e23=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                oc.FieldParameterSetTypes.VALUES,
                                                                elementNumber,voigt23Component)
            e33=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,
                                                                oc.FieldParameterSetTypes.VALUES,
                                                                elementNumber,voigt33Component)
            eT=np.array([[e11,e22,e33,e23,e13,e12]])
            e=np.array([[e11],
                        [e22],
                        [e12],
                        [e23],
                        [e13],
                        [e12]])
            
        strainEnergy=elasticityDerivedField.ParameterSetGetElementDP(oc.FieldVariableTypes.W,
                                                                     oc.FieldParameterSetTypes.VALUES,
                                                                     elementNumber,1)

                
        eTA = np.matmul(eT,A)
        eTAe = np.matmul(eTA,e)
        #print("eT   = ",etilde)
        #print("e    = ",e)
        #print("eTA  = ",etildeA)
        #print("eTAe = ",etildeAe)
        energy = eTAe[0]
             
        strValue = structureField.ParameterSetGetElementIntg(oc.FieldVariableTypes.U,
                                                             oc.FieldParameterSetTypes.VALUES,
                                                             elementNumber,1)

        strainEnergyDensity=(YOUNGS_MODULUS_MIN+float(strValue)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*strainEnergy
        
        # Store the strain energy density value 
        sedField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                             elementNumber,1,strainEnergyDensity)
                
        topologicalDerivative = (YOUNGS_MODULUS_MIN+float(strValue)*(YOUNGS_MODULUS-YOUNGS_MODULUS_MIN))*float(energy[0])
                
        # Store the topological derivative value 
        tdField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                            elementNumber,1,topologicalDerivative)
                
        rankStrSum = rankStrSum + float(strValue)
        rankSEDSum = rankSEDSum + strainEnergyDensity
        rankObjectiveSum = rankObjectiveSum + strainEnergy
                

    #Reduce objective, volume etc. sums across the ranks
    strSum = MPI.COMM_WORLD.allreduce(rankStrSum,op=MPI.SUM)
    sedSum = MPI.COMM_WORLD.allreduce(rankSEDSum,op=MPI.SUM)
    objectiveSum = MPI.COMM_WORLD.allreduce(rankObjectiveSum,op=MPI.SUM)
    
    volumeRatio = strSum/float(NUMBER_OF_ELEMENTS)
    objective[iterationNumber] = objectiveSum

    #Update fields
    sedField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    tdField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    sedField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    tdField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    
    seValues = elasticityDerivedField.ParameterSetDataGet(oc.FieldVariableTypes.W,oc.FieldParameterSetTypes.VALUES)
    sedValues = sedField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    tdValues = tdField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

    # Compute the nodal topological derivatives values and sums
    rankTDSum = 0.0
    rankAbsTDSum = 0.0
    for nodeIdx in range(1,numberOfLocalNodes+1):
        nodeNumber = decomposition.NodeNumberGet(1,nodeIdx)
        # Loop over the elements surrounding the node to determine the average
        averageTD = 0.0
        numberOfSurroundingElements = decomposition.NodeNumberOfSurroundingElementsGet(1,nodeNumber)
        for surroundingElementIdx in range(1,numberOfSurroundingElements+1):
            surroundingElementNumber = decomposition.NodeSurroundingElementGet(1,nodeNumber,surroundingElementIdx)
            
            topologicalDerivative = tdField.ParameterSetGetElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                                     surroundingElementNumber,1)
            averageTD = averageTD + topologicalDerivative
            
        averageTD = averageTD/float(numberOfSurroundingElements)
        # Set the topological derivative at the node
        tdField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.V,oc.FieldParameterSetTypes.VALUES,1, \
                                         oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,averageTD)
        # Update sums
        rankTDSum = rankTDSum + averageTD
        rankAbsTDSum = rankAbsTDSum + abs(averageTD)
        
    #Reduce objective, volume etc. sums across the ranks
    tdSum = MPI.COMM_WORLD.allreduce(rankTDSum,op=MPI.SUM)
    absTDSum = MPI.COMM_WORLD.allreduce(rankAbsTDSum,op=MPI.SUM)
   
    tdnValues = tdField.ParameterSetDataGet(oc.FieldVariableTypes.V,oc.FieldParameterSetTypes.VALUES)
    
    #-----------------------------------------------------------------------------------------------------------
    # CALCULATE AUGMENTED LAGRANGIAN PARAMETERS
    #-----------------------------------------------------------------------------------------------------------

    print("Current volume ratio = ",volumeRatio)
    print("Topological derivative sum = ",tdSum)
    print("ABS topological derivative sum = ",absTDSum)
          
    maximumG = MAX_VOLUME_RATIO+(initialVolume-MAX_VOLUME_RATIO)*max(0,1-iterationNumber/N_VOL_ITERATIONS)
    print("Max G = ",maximumG)
    G = volumeRatio - maximumG
    print("G = ",G)
    lambdaValue = tdSum/float(NUMBER_OF_NODES)*math.exp(LEVEL_SET_P_PARAM*(G/maximumG+LEVEL_SET_D_PARAM))
    print("lambda = ",lambdaValue)
    C = float(NUMBER_OF_NODES)/absTDSum
    print("C = ",C)

    # Update the diffusion source to be C*topologicalDerivative - lambda
    for nodeIdx in range(1,numberOfLocalNodes+1):
        nodeNumber = decomposition.NodeNumberGet(1,nodeIdx)
        topologicalDerivative = tdField.ParameterSetGetNodeDP(oc.FieldVariableTypes.V,oc.FieldParameterSetTypes.VALUES,1,
                                                              oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1)
        diffusionSourceValue = C*(topologicalDerivative - lambdaValue)
        diffusionSourceField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1, \
                                                      oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1, \
                                                      -diffusionSourceValue)
        
    # Update the diffusion source field across the ranks
    diffusionSourceField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    diffusionSourceField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
        
    diffusionSourceValues = diffusionSourceField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    
    #-----------------------------------------------------------------------------------------------------------
    # DIFFUSION SOLVE
    #-----------------------------------------------------------------------------------------------------------

    diffusionControlLoop.TimesSet(time,time+TIME_STEP,TIME_STEP)
    diffusionProblem.Solve()

    diffusionValues = diffusionDependentField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    structureValues = structureField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    youngsModulusValues = elasticityMaterialsField.ParameterSetDataGet(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    
    #-----------------------------------------------------------------------------------------------------------
    # RECALCULATE THE NEW STRUCUTRE FIELD AND VOLUME
    #-----------------------------------------------------------------------------------------------------------

    # Loop over the local elements
    for elementIdx in range(1,numberOfLocalElements+1):
        elementNumber = decomposition.ElementNumberGet(elementIdx)
        elementBasis = oc.Basis()
        decomposition.ElementBasisGet(1,elementNumber,elementBasis)
        numberOfElementNodes = elementBasis.NumberOfLocalNodesGet()
        
        # Find the average value of Phi in the element
        averagePhi = 0.0
        for localNodeIdx in range(1,numberOfElementNodes+1):
            nodeNumber = decomposition.ElementNodeGet(1,elementNumber,localNodeIdx)
            nodalPhi = diffusionDependentField.ParameterSetGetNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,
                                                                     oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1)
            # Reset phi to the limits
            nodalPhi = min(1.0,max(-1.0,nodalPhi))                
            # Update phi
            nodeDomain = decomposition.NodeDomainGet(1,nodeNumber)
            if (nodeDomain == computationalNodeNumber):
                diffusionDependentField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,
                                                                 oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                                                 nodalPhi)
            averagePhi = averagePhi + nodalPhi
            
        averagePhi = averagePhi/float(numberOfElementNodes)    
        # If the average phi in the elmeent is less than zero remove the element
        if (averagePhi < 0.0):
            structureField.ParameterSetUpdateElementIntg(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                         elementNumber,1,0)
            elasticityMaterialsField.ParameterSetUpdateElementDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                                 elementNumber,1,YOUNGS_MODULUS_MIN)
            

    # Update the fields across the ranks
    diffusionDependentField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    structureField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    elasticityMaterialsField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    diffusionDependentField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    structureField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    elasticityMaterialsField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    
    #-----------------------------------------------------------------------------------------------------------
    # OUTPUT
    #-----------------------------------------------------------------------------------------------------------

    filenameFormat = "BoneOptimisation_{Iteration:0d}"
    filename = filenameFormat.format(Iteration=iterationNumber)

    OutputFields(filename)
    
    #-----------------------------------------------------------------------------------------------------------
    # STATISTICS AND CHECK CONVERGENCE
    #-----------------------------------------------------------------------------------------------------------

    print("Iteration = %d, Objective = %f, Volume ratio = %f, lambda = %f" % (iterationNumber,objective[iterationNumber]/float(NUMBER_OF_ELEMENTS),volumeRatio,lambdaValue))
    
    if( (iterationNumber>=MAXIMUM_NUMBER_OF_ITERATIONS) ):
        continueLoop = False

#-----------------------------------------------------------------------------------------------------------
# MAIN LOOP END
#-----------------------------------------------------------------------------------------------------------

# Finalise OpenCMISS
oc.Finalise()

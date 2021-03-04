module NIDDL_TestCases

using LinearAlgebra
using SparseArrays
using SuiteSparse
using SharedArrays
using Distributed
using LinearMaps
using IterativeSolvers
using Random
using TimerOutputs
using GmshSDK

import Base: +,-,*,/,zero
import LinearAlgebra: norm, ldiv!

using NIDDL_FEM
using NIDDL

import NIDDL: indices_full_domain, indices_skeleton
import NIDDL: indices_domain, indices_transmission_boundary
import NIDDL: size_multi_trace, dof_weights
import NIDDL: get_matrix, get_matrix_no_transmission_BC, get_rhs
import NIDDL: get_transmission_matrix, DtN
import NIDDL: matrix

abstract type Medium end
abstract type Geometry end
abstract type TransmissionParameters end
abstract type TestCase end

abstract type BoundaryCondition end
abstract type PhysicalBC <: BoundaryCondition end
abstract type TransmissionBC <: BoundaryCondition end

include("meshing.jl")
include("unitvector.jl")
include("problem.jl")
include("boundary_condition.jl")
include("transmission_condition.jl")
include("non_local_transmission_conditions.jl")
include("geometry.jl")
include("medium.jl")
include("planewave.jl")
include("helmholtz.jl")
include("vector_helmholtz.jl")
include("maxwell.jl")
include("test_case.jl")
include("DtN.jl")
include("vtk_output.jl")
include("residual.jl")
include("ddm.jl")

export
    Geometry,
    BoundaryCondition,
    PhysicalBC,
    TransmissionBC,

    # boundary_condition.jl
    DirichletBC,
    DirichletWeakBC,
    NeumannBC,
    RobinBC,
    apply,
    rhs,
    physical_boundary,

    # ddm.jl
    InputData, StandardInputData, InductiveInputData,
    indices_full_domain,
    indices_skeleton,
    indices_domain,
    indices_transmission_boundary,
    size_multi_trace,
    dof_weights,
    get_matrix,
    get_matrix_no_transmission_BC,
    get_rhs,
    get_transmission_matrix,
    DtN,

    # geometry.jl
    LayersGeo,
    get_mesh_and_domains, get_problems,
    get_skeleton_problems,

    # helmholtz.jl
    HelmholtzPb,
    dofdim,

    # vector_helmholtz.jl
    VectorHelmholtzPb,

    # maxwell.jl
    MaxwellPb,
    PNEDtoP1,
    FarField,
    bistatic,

    # medium.jl
    AcousticMedium,
    ElectromagneticMedium,
    dissipative_medium,
    speed,
    wavenumber,
    acoef, bcoef, ccoef,

    # meshing.jl
    construct_mesh,
    extract_elements,
    construct_domains,
    detect_boundaries,
    geometrical_partitioning,
    element_layer,
    detect_boundary_elements,

    # non_local_transmission_conditions.jl
    DtN_neighbours_TP, DtN_neighbours_TBC,
    DtN_TP, DtN_TBC,
    matrix,

    # planewave.jl
    PlaneWave,
    AcousticPW,
    ElectromagneticPW,
    wavelength,
    incidence,
    direction,
    polarisation,
    get_planewave,
    absorbing_condition,
    neumann_condition,

    # problem.jl
    Problem,
    get_mass_matrix,
    get_matrix_building_blocks,
    solve,
    solve_gmres,
    cond,
    A_norm,
    A_L2norm,
    A_HDseminorm,
    A_HDnorm,
    toP1,

    # residual.jl
    get_resfunc,

    # unitvector.jl
    UnitVector,

    # test_case.jl
    TestCase,
    ScatteringTC,
    RandomTC,

    # transmission_condition.jl
    transmission_boundary,
    TransmissionParameters,
    Idl2TP, Idl2TBC,
    DespresTP, DespresTBC,
    SndOrderTP, SndOrderTBC,

    # vtk_output.jl
    get_vtkfile,
    save_mesh,
    save_vector,
    save_partition,
    save_medium,
    save_solutions,
    save_solutions_partition

end # module

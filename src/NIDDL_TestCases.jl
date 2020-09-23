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

import Base: +,-,*,/,zero
import LinearAlgebra: norm, ldiv!

using NIDDL_FEM
using NIDDL

import NIDDL: dofdim, get_matrix, get_rhs, transmission_boundary, boundary, matrix, DtN

# For this package to work, simlink the following two files in src directory
# - gmsh.jl
# - libgmsh.dylib (MAC OS) or ligmsh.so (LINUX)
include("gmsh.jl")

abstract type Medium end
abstract type Geometry end
abstract type TransmissionParameters end
abstract type TestCase end

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
include("maxwell.jl")
include("test_case.jl")
include("DtN.jl")
include("vtk_output.jl")
include("residual.jl")

export
    Geometry,

    # boundary_condition.jl
    DirichletBC,
    DirichletWeakBC,
    NeumannBC,
    RobinBC,
    apply,
    rhs,
    physical_boundary,

    # geometry.jl
    LayersGeo,
    get_mesh_and_domains, get_problems,

    # helmholtz.jl
    HelmholtzPb,

    # maxwell.jl
    MaxwellPb,
    PNEDtoP1,
    FarField,
    bistatic,

    # medium.jl
    AcousticMedium,
    ElectromagneticMedium,
    dissipative_medium,
    problem_type,
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

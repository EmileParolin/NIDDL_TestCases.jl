#################################
# Defines a dissipative problem #
#################################

"""
This is a hack, the implementation is not very clean...
"""
function dissipative_pb(pb_type, medium, Ω, pbcs)
    newBCs = Vector{BoundaryCondition}(undef,0)
    for bc in pbcs
        newbc = deepcopy(bc)
        if typeof(bc) <: RobinBC
            # Scalar or vector problem
            scalar_func = pb_type == HelmholtzPb || pb_type == VectorHelmholtzPb
            vector_func = pb_type == MaxwellPb
            @assert scalar_func ⊻ vector_func
            # Homogeneous Robin
            if scalar_func
                func = (args...)->Complex{Float64}(0)
            elseif vector_func
                func = (args...)->zeros(Complex{Float64},dim(Ω))
            end
            newbc = RobinBC(bc.Γ, x -> im*ccoef(medium)(x), func)
        end
        if !(typeof(bc) <: DirichletWeakBC)
            push!(newBCs, newbc)
        end
    end
    newpb = pb_type(medium, Ω, newBCs)
    return newpb
end


"""
Create a dissipative problem in the region of ω attached to the boundary Γ.
Necessarily Γ is a subset of γ = boundary(ω).
On the part of the boundary γ which does not belong to Γ or any of the physical
boundaries Γs, we impose robin boundary conditions.
"""
function dissipative_pb_in_tubular_region(ω, Γ, Γs, tc, pb_type, medium; fbc=:robin)
    γ = boundary(ω)
    @assert issubset(Γ, γ)
    # Extracting tubular region
    ωstrip = Domain([ωi for ωi in ω if !(isempty(intersect(Γ, boundary(ωi))))])
    γstrip = boundary(ωstrip)
    # Physical boundary conditions
    pbc = tc([intersect(γstrip, Γi) for Γi in Γs])
    # Looking for part of the boundary of ωstrip without boundary
    # condition, excluding Γ itself, and applying a default
    # Robin condition = 1st order ABC (used for thickened interfaces)
    Γ_R = remove(Γ, γstrip)
    Γ_R = remove(union([bc.Γ for bc in pbc]...), Γ_R)
    if fbc == :dirichlet
        push!(pbc, DirichletBC(Γ_R, Complex{Float64}(0)))
    elseif fbc == :neumann
        push!(pbc, NeumannBC(Γ_R, (args...)->Complex{Float64}(0)))
    elseif fbc == :robin
        push!(pbc, RobinBC(Γ_R, x -> im*ccoef(medium)(x),
                           (args...)->Complex{Float64}(0)))
    else
        error("Fictitious boundary condition not recognized.")
    end
    filter!(bc -> !isempty(bc.Γ), pbc)
    # Returing the (dissipative) problem associated to ωstrip
    return dissipative_pb(pb_type, medium, ωstrip, pbc)
end

###############################################
# transmission operator with dissipative DtNs #
###############################################

struct DtN_neighbours_TP <: TransmissionParameters
    z::Complex{Float64}
    pb_type::DataType
    medium::Medium
    fbc::Symbol          # boundary condition to impose on fictitious BCs
    function DtN_neighbours_TP(;z=1,pb_type=missing,medium=missing,fbc=:robin)
        return new(z,pb_type,medium,fbc)
    end
end
mutable struct DtN_neighbours_TBC <: TransmissionBC
    tp::DtN_neighbours_TP
    Γ::Domain
    pbs::Vector{Problem} # Problems of all DtN operators
    T::SparseMatrixCSC{Complex{Float64},Int64} # To store matrix (for multiple uses)
    function DtN_neighbours_TBC(tp,Γ,Ωs,Γs,tc)
        pbs = Problem[]
        for ω in Ωs
            γ = boundary(ω)
            if issubset(Γ, γ)
                # Adding the (dissipative) problem associated to ωstrip
                push!(pbs, dissipative_pb_in_tubular_region(ω, Γ, Γs, tc,
                                                            tp.pb_type,
                                                            tp.medium;
                                                            fbc=tp.fbc))
            end
        end
        new(tp,Γ,pbs,spzeros(Float64,0,0))
    end
end
function (tp::DtN_neighbours_TP)(Γ::Domain;
                                 Ωs=Ωs,Γs=Γs,tc=tc,kwargs...)
    return DtN_neighbours_TBC(tp,Γ,Ωs,Γs,tc)
end

function matrix(m::Mesh,pb::Problem,bc::DtN_neighbours_TBC)
    @assert length(bc.pbs) <= 2
    # Checking if matrix is already computed
    if length(bc.T) == 0
        T = sum([DtN(m,dtn_pb,bc.Γ) for dtn_pb in bc.pbs]) / length(bc.pbs)
        # Storing matrix (DDM requires at least twice its evaluation)
        bc.T = pb.medium.k0 * bc.tp.z * T
    end
    return bc.T
end


"""
From the initial system A, we compute an augmented system newA by adding the
auxiliary system B.  The coupling is done through the interface Σ between the
domains ΩA and ΩB.

If   A = [ A00  A0Σ ]    and    B = [ B00  B0Σ ]
         [ AΣ0  AΣΣ ]               [ BΣ0  BΣΣ ]

then newA  = [ A00  A0Σ          ]
             [ AΣ0  AΣΣ+BΣΣ  BΣ0 ]
             [          B0Σ  B00 ]

The tricky bit is that A00 and B00 might have different sizes and those sizes
cannot be known a priori through the use of number_of_elements / restriction
methods... However, it is assumed that the lines and columns of Σ both in A and B
are the same (and match the ones of the restriction matrix on Σ).
"""
function add_auxiliary_matrix(m::Mesh,A,ΩA::Domain,B,ΩB::Domain,Σ::Domain,d::Integer)
    @assert size(A,1) == size(A,2)
    @assert size(B,1) == size(B,2)
    @assert isempty(setdiff(tags(Σ,d), tags(ΩA,d)))
    @assert isempty(setdiff(tags(Σ,d), tags(ΩB,d)))
    # Size of new matrix
    NA = size(A,1)
    NB = size(B,1)
    NΣ = number_of_elements(m,Σ,d)
    N = NA + NB - NΣ
    # New matrix, initialized with A contributions
    newA = sparse(findnz(A)...,N,N)
    # Definition of some mappings
    RΩA = restriction(m,ΩA,d)
    RΩB = restriction(m,ΩB,d)
    RΣ = restriction(m,Σ,d)
    IB = sparse(I,NB,NB)
    IΣinB = sparse(findnz(RΩB*transpose(RΣ)*RΣ*transpose(RΩB))...,NB,NB)
    i,j,v = findnz(IB - IΣinB)
    MBtoB0 = sparse(collect(1:length(j)),j,ones(Bool,length(j)),length(j),NB)
    i,j,v = findnz(IΣinB)
    MBtoΣ = sparse(collect(1:length(j)),j,ones(Bool,length(j)),length(j),NB)
    MBtoΣtoΩA = RΩA * transpose(RΣ) * MBtoΣ
    # Extraction of block informations
    i00, j00, v00 = findnz(MBtoB0    * B * transpose(MBtoB0))
    iΣΣ, jΣΣ, vΣΣ = findnz(MBtoΣtoΩA * B * transpose(MBtoΣtoΩA))
    i0Σ, j0Σ, v0Σ = findnz(MBtoB0    * B * transpose(MBtoΣtoΩA))
    iΣ0, jΣ0, vΣ0 = findnz(MBtoΣtoΩA * B * transpose(MBtoB0))
    # Adding B contributions
    newA += sparse(NA.+i00, NA.+j00, v00, N, N)
    newA += sparse(    iΣΣ,     jΣΣ, vΣΣ, N, N)
    newA += sparse(NA.+i0Σ,     j0Σ, v0Σ, N, N)
    newA += sparse(    iΣ0, NA.+jΣ0, vΣ0, N, N)
    return newA
end


function apply(A,m::Mesh,pb::Problem,bc::DtN_neighbours_TBC)
    # For information: how much does the operator cost?
    NT = sum([number_of_elements(m, dtn_pb.Ω, dofdim(dtn_pb))
              for dtn_pb in bc.pbs])
    Npb = number_of_elements(m, pb.Ω, dofdim(pb))
    @info "   --> #DOF T operator $(NT)"
    @info "   --> Ratio #DOF Top / #DOF pb: $(floor(100*(NT / Npb))/100)"
    # Actual construction
    for dtn_pb in bc.pbs
        K = 1/length(bc.pbs) * get_matrix(m,dtn_pb)
        A = add_auxiliary_matrix(m,A,pb.Ω,K,dtn_pb.Ω,bc.Γ,dofdim(pb))
    end
    return A
end


function rhs(b,m::Mesh,pb::Problem,bc::DtN_neighbours_TBC)
    # size of new RHS
    NΣ = number_of_elements(m,bc.Γ,dofdim(pb))
    Naux = sum([length(get_rhs(m,pb))-NΣ for pb in bc.pbs])
    N = length(b) + Naux
    # New RHS
    newb = zeros(eltype(b),N)
    newb[1:length(b)] = b
    return newb
end


##############################################
# transmission operator with dissipative DtN #
##############################################

"""
This defines a DtN using FEM where the domain is the domain of the problem
associated to the transmission interface.

This is the classic implementation of a DtN operator.
"""
struct DtN_TP <: TransmissionParameters
    z::Complex{Float64}
    pb_type::DataType
    medium::Medium
    fbc::Symbol          # boundary condition to impose on fictitious BCs
    function DtN_TP(;z=1,pb_type=missing,medium=missing,fbc=:robin)
        return new(z,pb_type,medium,fbc)
    end
end
mutable struct DtN_TBC <: TransmissionBC
    tp::DtN_TP
    Γ::Domain
    Γs::Vector{Domain}
    tc::TestCase
    T::SparseMatrixCSC{Complex{Float64},Int64} # To store matrix (for multiple uses)
    function DtN_TBC(tp,Γ,Γs,tc)
        new(tp,Γ,Γs,tc,spzeros(Float64,0,0))
    end
end
function (tp::DtN_TP)(Γ::Domain; Γs=Γs,tc=tc,kwargs...)
    return DtN_TBC(tp,Γ,Γs,tc)
end

function matrix(m::Mesh,pb::Problem,bc::DtN_TBC)
    @assert boundary(pb.Ω) == bc.Γ
    # Checking if matrix is already computed
    if length(bc.T) == 0
        # Dissipative problem
        dtn_pb = dissipative_pb_in_tubular_region(pb.Ω, bc.Γ, bc.Γs, bc.tc,
                                                  bc.tp.pb_type, bc.tp.medium;
                                                  fbc=bc.tp.fbc)
        # DtN
        T = DtN(m, dtn_pb, bc.Γ)
        # Storing matrix (DDM requires at least twice its evaluation)
        bc.T = pb.medium.k0 * bc.tp.z * T
    end
    return bc.T
end

function apply(A,m::Mesh,pb::Problem,bc::DtN_TBC)
    @assert boundary(pb.Ω) == bc.Γ
    # Dissipative problem
    dtn_pb = dissipative_pb_in_tubular_region(pb.Ω, bc.Γ, bc.Γs, bc.tc,
                                              bc.tp.pb_type, bc.tp.medium;
                                              fbc=bc.tp.fbc)
    # For information: how much does the operator cost?
    NT = number_of_elements(m, dtn_pb.Ω, dofdim(dtn_pb))
    Npb = number_of_elements(m, pb.Ω, dofdim(pb))
    @info "   --> #DOF T operator $(NT)"
    @info "   --> Ratio #DOF Top / #DOF pb: $(floor(100*(NT / Npb))/100)"
    # Add auxiliary problem
    K = get_matrix(m,dtn_pb)
    A = add_auxiliary_matrix(m,A,pb.Ω,K,dtn_pb.Ω,bc.Γ,dofdim(pb))
    return A
end


function rhs(b,m::Mesh,pb::Problem,bc::DtN_TBC)
    @assert boundary(pb.Ω) == bc.Γ
    # Dissipative problem
    dtn_pb = dissipative_pb_in_tubular_region(pb.Ω, bc.Γ, bc.Γs, bc.tc,
                                              bc.tp.pb_type, bc.tp.medium;
                                              fbc=bc.tp.fbc)
    # size of new RHS
    NΣ = number_of_elements(m,bc.Γ,dofdim(pb))
    Naux = length(get_rhs(m,dtn_pb)) - NΣ
    N = length(b) + Naux
    # New RHS
    newb = zeros(eltype(b),N)
    newb[1:length(b)] = b
    return newb
end

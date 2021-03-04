#######################
# Boundary Conditions #
#######################

function physical_boundary(pb::Problem)
    union([bc.Γ for bc in pb.BCs if typeof(bc) <: PhysicalBC]...)
end

struct DirichletBC <: PhysicalBC
    Γ::Domain
    α::Complex{Float64} # only handling constant Dirichlet BC
end


"""
Dirichlet boundary condition implemented via Lagrange Multipliers.
"""
struct DirichletWeakBC <: PhysicalBC
    Γ::Domain
    α::Union{Complex{Float64},Vector{Complex{Float64}}} # only handling constant Dirichlet BC
end


struct NeumannBC <: PhysicalBC
    Γ::Domain
    f::Function
end


struct RobinBC <: PhysicalBC
    Γ::Domain
    α::Function # Should be ik
    f::Function # RHS
end

###################
# Apply functions #
###################

"""Apply Dirichlet boundary condition on matrix.

We use pseudo elimination:
    - the matrix blocks (diagonal and off-diagonal blocks) related to the DOF
    concerned are removed 
    - the diagonal block is replaced by an identity matrix
"""
function apply(A,m::Mesh,pb::Problem,bc::DirichletBC)
    # Restriction matrices
    RΓ = restriction(m,bc.Γ,dofdim(pb))
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    # Identity matrices
    Id = sparse(I,size(A)...) #
    Id_Γ = RΩ*transpose(RΓ)*RΓ*transpose(RΩ)
    # Elimination matrix
    R = Id - Id_Γ 
    return R*A + Id_Γ
end


function apply(A,m::Mesh,pb::Problem,bc::DirichletWeakBC)
    # New contribution
    RΓ = restriction(m,bc.Γ,dofdim(pb))
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    M = RΩ * get_mass_matrix(m,bc.Γ,pb) * transpose(RΓ)
    i,j,v = findnz(M)
    # Size of new matrix
    NA = size(A,1)
    NΓ = number_of_elements(m,bc.Γ,dofdim(pb))
    N = NA + NΓ
    # New matrix, initialized with A contributions
    newA = sparse(findnz(A)...,N,N)
    # Adding new contributions
    newA += sparse(NA.+j,     i, v, N, N)
    newA += sparse(    i, NA.+j, v, N, N)
    return newA
end


function apply(A,m::Mesh,pb::Problem,bc::NeumannBC)
    return A
end


function apply(A,m::Mesh,pb::Problem,bc::RobinBC)
    M = get_mass_matrix(m,bc.Γ,pb; coef=bc.α)
    # Taking care of potential auxiliary equations
    RΩ = restriction(m, pb.Ω, dofdim(pb))
    MAtoΩ = sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(A,1))
    return A #- transpose(MAtoΩ)*RΩ * M * transpose(RΩ)*MAtoΩ
end

#################
# RHS functions #
#################

function rhs(b,m::Mesh,pb::Problem,bc::DirichletBC)
    # Taking care of potential auxiliary equations
    RΓ = restriction(m,bc.Γ,dofdim(pb))
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    MbtoΩ = sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(b,1))
    IΓ = transpose(MbtoΩ)*RΩ*transpose(RΓ)*RΓ*transpose(RΩ)*MbtoΩ
    return b + bc.α * IΓ * ones(length(b))
end


function rhs(b,m::Mesh,pb::Problem,bc::DirichletWeakBC)
    # Additionnal linear term
    Γ = bc.Γ
    q = quadrature(Γ)
    W = weight_matrix(m,Γ,q)
    u = unknown_fe_type(Γ,pb)
    Mu = assemble(u,m,Γ,q)
    Mf = assemble(functype(pb),m,Γ,q,(args...)->bc.α)
    bΓ = Vector(femdot(Mu,W,Mf)[:])
    # Restriction matrix
    RΓ = restriction(m,bc.Γ,dofdim(pb))
    # size of new RHS
    N = length(b) + number_of_elements(m,bc.Γ,dofdim(pb))
    # New RHS
    newb = zeros(eltype(b),N)
    newb[1:length(b)] = b
    newb[length(b)+1:end] = RΓ * bΓ
    return newb
end


function rhs(b,m::Mesh,pb::Problem,bc::Union{NeumannBC,RobinBC})
    # Additionnal linear term
    Γ = bc.Γ
    q = quadrature(Γ)
    W = weight_matrix(m,Γ,q)
    u = unknown_fe_type(Γ,pb)
    Mu = assemble(u,m,Γ,q)
    Mf = assemble(functype(pb),m,Γ,q,bc.f)
    bΓ = Vector(femdot(Mu,W,Mf)[:])
    # Taking care of potential auxiliary equations
    RΩ = restriction(m,pb.Ω,dofdim(pb))
    MbtoΩ = sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(b,1))
    return b - transpose(MbtoΩ)*RΩ * bΓ
end

####################################
# Transmission Boundary Conditions #
####################################

function transmission_boundary(pb::Problem)
    union([bc.Γ for bc in pb.BCs if typeof(bc) <: TransmissionBC]...)
end

"""
Variational formulation

    a(u,u^t) - ik t(u,u^t) = RHS

The implementation is very close to the one of the Robin boundary condition.
"""
function apply(A,m::Mesh,pb::Problem,bc::TransmissionBC)
    T = matrix(m,pb,bc)
    # Taking care of potential auxiliary equations
    RΓ = restriction(m, bc.Γ, dofdim(pb))
    RΩ = restriction(m, pb.Ω, dofdim(pb))
    MAtoΩ = sparse(I, number_of_elements(m,pb.Ω,dofdim(pb)), size(A,1))
    P = RΓ*transpose(RΩ)*MAtoΩ
    return A - im * transpose(P) * T * P
end

"""
Transmission boundary conditions do not modify RHS of local problems.
"""
rhs(b,m::Mesh,pb::Problem,bc::TransmissionBC) = b

########
# Idl2 #
########

struct Idl2TP <: TransmissionParameters
    z::Complex{Float64}
    Idl2TP(;z=1) = new(z)
end
struct Idl2TBC <: TransmissionBC
    tp::Idl2TP
    Γ::Domain
end
(tp::Idl2TP)(Γ::Domain; kwargs...) = Idl2TBC(tp, Γ)

function matrix(m::Mesh,pb::Problem,bc::Idl2TBC)
    N = number_of_elements(m,bc.Γ,dofdim(pb))
    M = sparse(I,N,N)
    return pb.medium.k0 * bc.tp.z * M
end

###########
# Despres #
###########

struct DespresTP <: TransmissionParameters
    z::Complex{Float64}
    DespresTP(;z=1) = new(z)
end
struct DespresTBC <: TransmissionBC
    tp::DespresTP
    Γ::Domain
end
(tp::DespresTP)(Γ::Domain; kwargs...) = DespresTBC(tp, Γ)

function matrix(m::Mesh,pb::Problem,bc::DespresTBC)
    RΓ = restriction(m, bc.Γ, dofdim(pb))
    coef = x -> ccoef(pb.medium)(x) / pb.medium.k0
    M = RΓ * get_mass_matrix(m,bc.Γ,pb; coef=coef) * transpose(RΓ)
    return pb.medium.k0 * bc.tp.z * M
end

#############
# 2nd order #
#############

"""
Second order transmission operator.

    (∂n + ik T) u = ...

with

    T = z (Id - α ∂tt)    and    z = 1,   α = 1 / (2 k^2)

Note that the minus sign in front of the α disappears after the integration by
parts.
"""
struct SndOrderTP <: TransmissionParameters
    z::Complex{Float64}
    α::Complex{Float64}
    SndOrderTP(;z=1,α=0.5) = new(z,α)
end
struct SndOrderTBC <: TransmissionBC
    tp::SndOrderTP
    Γ::Domain
end
(tp::SndOrderTP)(Γ::Domain; kwargs...) = SndOrderTBC(tp, Γ)

function matrix(m::Mesh,pb::Problem,bc::SndOrderTBC)
    RΓ = restriction(m, bc.Γ, dofdim(pb))
    M, K = get_matrix_building_blocks(m,bc.Γ,pb)
    return pb.medium.k0 * bc.tp.z * RΓ * (M + bc.tp.α * K) * transpose(RΓ)
end

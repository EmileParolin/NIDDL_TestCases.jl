"""
Helmholtz problem
"""
mutable struct HelmholtzPb <: Problem
    medium::AcousticMedium
    Ω::Domain
    BCs::Array{BoundaryCondition,1}
    b0::Vector{Complex{Float64}}     # RHS
end
function HelmholtzPb(medium, Ω, BCs)
    b0 = Vector{Complex{Float64}}(undef, 0)
    return HelmholtzPb(medium, Ω, BCs, b0)
end

"""
    DOFs are nodes.
"""
dofdim(pb::HelmholtzPb) = 0
dofdim(::Type{<:HelmholtzPb}) = 0

function functype(pb::HelmholtzPb)
    if dim(pb.Ω) == 3 return ScaTriFunc
    elseif dim(pb.Ω) == 2 return ScaEdgFunc
    elseif dim(pb.Ω) == 1 return ScaNodFunc
    end
end

function unknown_fe_type(Ω::Domain,pb::HelmholtzPb)
    if dim(Ω) == 3 return P1tet
    elseif dim(Ω) == 2 return P1tri
    elseif dim(Ω) == 1 return P1edg
    elseif dim(Ω) == 0 return P0nod
    end
end

function D_unknown_fe_type(Ω::Domain,pb::HelmholtzPb)
    if dim(Ω) == 3 return gradP1tet
    elseif dim(Ω) == 2 return gradP1tri
    elseif dim(Ω) == 1 return gradP1edg
    end
end

"""
Helmholtz problem
"""
struct HelmholtzPb <: Problem
    medium::AcousticMedium
    Ω::Domain
    BCs::Array{BoundaryCondition,1}
end

dofdim(pb::HelmholtzPb) = 0 # DOFs are points

function functype(pb::HelmholtzPb)
    if dim(pb.Ω) == 3 return ScaTriFunc
    elseif dim(pb.Ω) == 2 return ScaEdgFunc
    end
end

function unknown_fe_type(Ω::Domain,pb::HelmholtzPb)
    if dim(Ω) == 3 return P1tet
    elseif dim(Ω) == 2 return P1tri
    elseif dim(Ω) == 1 return P1edg
    end
end

function D_unknown_fe_type(Ω::Domain,pb::HelmholtzPb)
    if dim(Ω) == 3 return gradP1tet
    elseif dim(Ω) == 2 return gradP1tri
    elseif dim(Ω) == 1 return gradP1edg
    end
end

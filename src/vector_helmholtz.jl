"""
Vector Helmholtz problem
"""
struct VectorHelmholtzPb <: Problem
    medium::AcousticMedium
    Ω::Domain
    BCs::Array{BoundaryCondition,1}
end

dofdim(pb::VectorHelmholtzPb) = 1 # DOFs are edges

function functype(pb::VectorHelmholtzPb)
    if dim(pb.Ω) == 3 return ScaTriFunc
    elseif dim(pb.Ω) == 2 return ScaEdgFunc
    end
end

function unknown_fe_type(Ω::Domain,pb::VectorHelmholtzPb)
    if dim(Ω) == 3 return error("Not implemented")
    elseif dim(Ω) == 2 return RT
    elseif dim(Ω) == 1 return P0edg
    end
end

function D_unknown_fe_type(Ω::Domain,pb::VectorHelmholtzPb)
    if dim(Ω) == 3 return error("Not implemented")
    elseif dim(Ω) == 2 return divRT
    elseif dim(Ω) == 1 return error("Not implemented")
    end
end


####################################
# L2 projection from Nedelec to P1 #
####################################

struct PRTtoP1
    P1_P1_LU
    P1_RTx
    P1_RTy
    P1_RTz
end

function PRTtoP1(m::Mesh,Ω::Domain)
    # Restriction matrices
    RΩ0 = restriction(m,Ω,0) # on vertices
    RΩ1 = restriction(m,Ω,1) # on edges
    # Quadrature
    q = quadrature(Ω)
    Wtri = weight_matrix(m,Ω,q)
    # Finite Elements definitions
    MP1 = assemble(P1tri,m,Ω,q)
    MRTtri = assemble(RT,m,Ω,q)
    # Finite Elements matrices
    P1_P1 = RΩ0 * femdot(MP1,Wtri,MP1) * transpose(RΩ0)
    P1_RTx =  RΩ0 *femdot(MP1,Wtri,MRTtri[[1]]) * transpose(RΩ1)
    P1_RTy =  RΩ0 *femdot(MP1,Wtri,MRTtri[[2]]) * transpose(RΩ1)
    P1_RTz =  RΩ0 *femdot(MP1,Wtri,MRTtri[[3]]) * transpose(RΩ1)
    # Factorization
    P1_P1_LU = factorize(P1_P1)
    return PRTtoP1(P1_P1_LU,P1_RTx,P1_RTy,P1_RTz)
end

function (P::PRTtoP1)(u::Array{Complex{Float64},1})
    # L2 projection
    ux = P.P1_P1_LU \ (P.P1_RTx * u)
    uy = P.P1_P1_LU \ (P.P1_RTy * u)
    uz = P.P1_P1_LU \ (P.P1_RTz * u)
    return vcat(ux',uy',uz')
end

toP1(pb::VectorHelmholtzPb,m::Mesh,Ω::Domain,u) = Matrix(PRTtoP1(m,Ω)(u))
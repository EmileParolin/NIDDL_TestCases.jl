"""
Maxwell problem
"""
mutable struct MaxwellPb <: Problem
    medium::ElectromagneticMedium
    Ω::Domain
    BCs::Array{BoundaryCondition,1}
    b0::Vector{Complex{Float64}}     # RHS
end
function MaxwellPb(medium, Ω, BCs)
    b0 = Vector{Complex{Float64}}(undef, 0)
    return MaxwellPb(medium, Ω, BCs, b0)
end

"""
    DOFs are edges (RT0 = 1st order RT).
"""
dofdim(pb::MaxwellPb) = 1
dofdim(::Type{<:MaxwellPb}) = 1

functype(pb::MaxwellPb) = VecTriFunc
unknown_fe_type(Ω::Domain,pb::MaxwellPb) = dim(Ω) == 3 ? NEDtet : NEDtri
D_unknown_fe_type(Ω::Domain,pb::MaxwellPb) = dim(Ω) == 3 ? curlNEDtet : curlNEDtri

####################################
# L2 projection from Nedelec to P1 #
####################################

struct PNEDtoP1
    P1_P1_LU
    P1_NEDx
    P1_NEDy
    P1_NEDz
end

function PNEDtoP1(m::Mesh,Ω::Domain)
    # Restriction matrices
    RΩ0 = restriction(m,Ω,0) # on vertices
    RΩ1 = restriction(m,Ω,1) # on edges
    # Quadrature
    q = quadrature(Ω)
    Wtet = weight_matrix(m,Ω,q)
    # Finite Elements definitions
    MP1 = assemble(P1tet,m,Ω,q)
    MNEDtet = assemble(NEDtet,m,Ω,q)
    # Finite Elements matrices
    P1_P1 = RΩ0 * femdot(MP1,Wtet,MP1) * transpose(RΩ0)
    P1_NEDx =  RΩ0 *femdot(MP1,Wtet,MNEDtet[[1]]) * transpose(RΩ1)
    P1_NEDy =  RΩ0 *femdot(MP1,Wtet,MNEDtet[[2]]) * transpose(RΩ1)
    P1_NEDz =  RΩ0 *femdot(MP1,Wtet,MNEDtet[[3]]) * transpose(RΩ1)
    # Factorization
    P1_P1_LU = factorize(P1_P1)
    return PNEDtoP1(P1_P1_LU,P1_NEDx,P1_NEDy,P1_NEDz)
end

function (P::PNEDtoP1)(u::Array{Complex{Float64},1})
    # L2 projection
    ux = P.P1_P1_LU \ (P.P1_NEDx * u)
    uy = P.P1_P1_LU \ (P.P1_NEDy * u)
    uz = P.P1_P1_LU \ (P.P1_NEDz * u)
    return vcat(ux',uy',uz')
end

toP1(pb::MaxwellPb,m::Mesh,Ω::Domain,u) = Matrix(PNEDtoP1(m,Ω)(u))

################################
# Far field computations (RCS) #
################################

"""
    FarField(m::Mesh,pb::MaxwellPb,u,bc::RobinBC)

Far field pattern in the direction x (unit vector)

\\[
Einf(x) = ik/4π x × [Z ∫_Γ J(y) e^{-ikx⋅y} dγ(y)] × x
         +ik/4π x ×    ∫_Γ M(y) e^{-ikx⋅y} dγ(y)
\\]

J = +n × H^+ - n × H^-
M = +n × E^+ - n × E^-

The input u is a volumic field E (Nedelec),
its trace is γu = n×E×n (Nedelec). Hence

M = n×E×n (when interpreted as RT = n×NED)
J = n × H
  = n × 1/ik curl E
  = -n×E×n + 1/ik Er (where Er is the RHS of Robin BC)
"""
struct FarField
    k # Wavenumber (in vacuum) of problem
    y # Gauss points on which to perform integration
    J
    M
    function FarField(m::Mesh,pb::MaxwellPb,u,Γ::Domain,E_inc::Function)
        # Coefficient
        coef = im*pb.medium.k0/(4π)
        # Quadrature
        qtri = triquad[3]
        Wtri = weight_matrix(m,Γ,qtri)
        Rq = restriction(m,Γ,qtri,2)
        y = nodes(m,Γ,qtri,2) # (dim,nbgausspts*nbelts)
        # Finite Elements definitions
        MRT = assemble(RT,m,Γ,qtri)
        MNED = assemble(NEDtri,m,Γ,qtri)
        # Evaluation of Einc traces
        nxEinc = assemble(VecTriFunc,m,Γ,qtri,(x,ielt,n)->n×E_inc(x,n))
        nxEincxn = assemble(VecTriFunc,m,Γ,qtri,(x,ielt,n)->n×E_inc(x,n)×n)
        # Evaluation of the trace of u (dim,nbgausspts*nbelts)
        Msca = vcat([transpose(Rq*Wtri*MRT[i]*u) for i in 1:3]...)
        Jsca =-vcat([transpose(Rq*Wtri*MNED[i]*u) for i in 1:3]...)
        # Evaluation of the trace of incident field (dim,nbgausspts*nbelts)
        Minc =-vcat([transpose(Rq*Wtri*nxEinc[i]) for i in 1:3]...)
        Jinc = vcat([transpose(Rq*Wtri*nxEincxn[i]) for i in 1:3]...)
        return new(pb.medium.k0, y, coef*(Jsca+Jinc), coef*(Msca+Minc))
    end
end
"""
`x` must be a unit vector (norm=1).
"""
function (FF::FarField)(x)
    k = FF.k
    # Initialisation
    res = zeros(Complex{Float64},3)
    # Looping on Gauss points (computation of quadrature)
    for iy in 1:size(FF.y,2)
        # Getting info
        y = FF.y[:,iy]
        J = FF.J[:,iy]
        M = FF.M[:,iy]
        # Computing far field contribution
        res += (x×J×x + x×M) * exp(-im*k*(x⋅y))
    end
    return res
end

"""
The bistatic RCS is defined as

rcs [m^2] = lim(R→∞) 4π R^2 |E_sc(R)|^2 / |E_inc(R)|^2
RCS [dB m^2] = 10 log10 ( rcs )

We suppose that the incident field is normalised |E_inc(R)|^2 = 1

The bistatic RCS is computed in the plane containing the direction of incidence
and the direction of polarisation.
"""
function bistatic(FF, inc::UnitVector; N=100)
    # Initialisation
    uinf = Array{Complex{Float64},2}(undef,3,N) # Far Field at infinity
    rcs = Dict([(:θ,Array{Float64,1}(undef,N)), # Radar Cross Section
                (:ϕ,Array{Float64,1}(undef,N))])
    # Directions of observation
    δθ = collect(-π:2π/(N-1):π) # Increment
    θs = inc.θ .+ δθ
    ϕs = inc.ϕ * ones(Float64,N)
    for i in 1:N
        # Corrections so that 0 <= θ <= π and 0 <= ϕ < 2π
        if θs[i] < 0
            θs[i] = abs(θs[i])
            ϕs[i] = ϕs[i] >= π ? ϕs[i]-π : ϕs[i]+π
        elseif π < θs[i]
            θs[i] = 2π - θs[i]
            ϕs[i] = ϕs[i] >= π ? ϕs[i]-π : ϕs[i]+π
        end
    end
    xinf = UnitVector.(θs, ϕs)
    # Loop on direction of observation
    for (ix,x) in enumerate(xinf)
        # Computing far field in direction x
        FFx = FF(er(x))
        uinf[:,ix] = FFx
        # RCS computation
        rcs[:θ][ix] = 10*log10(4π*abs(dot(FFx, eθ(x))))
        rcs[:ϕ][ix] = 10*log10(4π*abs(dot(FFx, eϕ(x))))
    end
    return δθ, xinf, uinf, rcs
end

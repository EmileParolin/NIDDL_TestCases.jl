abstract type PlaneWave end

wavelength(pw::PlaneWave) = 2π/pw.medium.k0
incidence(pw::PlaneWave) = UnitVector(pw.θ0,pw.ϕ0)
direction(pw::PlaneWave) = er(incidence(pw))

############
# Acoustic #
############

"""
Suppose that we take a time convention in e^{-iσ wt} with the sign σ ∈ {-1,1}.
The acoustic plane wave writes

    pw(x) = p0 e^{iσ (k d⋅x - wt)}
"""
struct AcousticPW <: PlaneWave
    medium::Medium      # medium
    θ0::Real            # incidence direction angle
    ϕ0::Real            # incidence direction angle
    d::Array{Float64,1} # direction of incidence
    function AcousticPW(medium::AcousticMedium=AcousticMedium(;k0=1),θ0=π,ϕ0=0)
        d = er(UnitVector(θ0,ϕ0)) # (π,0) gives -z
        new(medium,θ0,ϕ0,d)
    end
end
(pw::AcousticPW)(x) = exp(im*pw.medium.k0*(pw.d⋅x))

"""
We take a spherical out-going wave centered at the origin

    u(r) = e^{iσ (kr - wt)}

Where the sign σ is taken as in the definition of the acoustic plane wave.

Now we look for the absorbing condition, i.e. we consider a large sphere of
radius R. We have

    u(R) = e^{iσ (kR - wt)}
    ∇ u(R) = iσ kν e^{iσ (kR - wt)}

where ν(θ,ϕ) is the unit normal vector to the sphere. It's clear that

    (-ν ⋅ ∇ + iσ k) u(R) = 0

So the first order absorbing condition for a general surface with unit normal
vector n writes

    (-n ⋅ ∇ + iσ k) u(R) ≈ 0

For the plane wave we have

    pw(x) = p0 e^{iσ (k d⋅x - wt)}
    ∇ pw(x) = iσ kd p0 e^{iσ (k d⋅x - wt)}

Hence

    (-n ⋅ ∇ + iσ k) pw(x) = iσ k (-n⋅d + 1) pw(x)
"""
function absorbing_condition(pw::AcousticPW)
    a = acoef(pw.medium)
    c = ccoef(pw.medium)
    k0 = pw.medium.k0
    return (x,ielt,n) -> im * (- k0 * (n⋅pw.d) * a(x) + c(x)) * pw(x)
end

function neumann_condition(pw::AcousticPW)
    a = acoef(pw.medium)
    k0 = pw.medium.k0
    return (x,ielt,n) -> im * k0 * (n⋅pw.d) * a(x) * pw(x)
end

function get_planewave(medium::AcousticMedium,θ0,ϕ0,args...)
    return AcousticPW(medium,θ0,ϕ0)
end

####################
# Electromagnetism #
####################

"""
Suppose that we take a time convention in e^{-iσ wt} with the sign σ ∈ {-1,1}.
The electromagnetic plane wave writes

    pw(x) = p0 e^{iσ (k d⋅x - wt)}
"""
struct ElectromagneticPW <: PlaneWave
    medium::Medium      # medium
    θ0::Real            # incidence direction angle
    ϕ0::Real            # incidence direction angle
    pol::Symbol         # polarisation (:θ, :ϕ)
    d::Array{Float64,1} # direction of incidence
    p::Array{Float64,1} # E_0
    function ElectromagneticPW(medium::ElectromagneticMedium=ElectromagneticMedium(;k0=1),θ0=π,ϕ0=0,pol=:θ)
        # incidence direction
        ud = UnitVector(θ0,ϕ0) # (π,0) gives -z
        d = er(ud)
        # polarisation
        if (pol==:θ) p = eθ(ud) elseif (pol==:ϕ) p = eϕ(ud) end
        new(medium,θ0,ϕ0,pol,d,p)
    end
end
(pw::ElectromagneticPW)(x) = pw.p*exp(im*pw.medium.k0*(pw.d⋅x))
polarisation(pw::ElectromagneticPW) = pw.pol

"""
We take a spherical out-going wave centered at the origin

    u(r) = u0 e^{iσ (kr - wt)}

Where the sign σ is taken as in the definition of the acoustic plane wave.

Now we look for the absorbing condition, i.e. we consider a large sphere of
radius R. We have

    u(R) = u0 e^{iσ (kR - wt)}
    curl u(R) = iσ kν × u0 e^{iσ (kR - wt)}

where ν(θ,ϕ) is the unit normal vector to the sphere. It's clear that

    [ν × curl ⋅ + iσ k n × (⋅ × n)] u(R) = 0

So the first order absorbing condition for a general surface with unit normal
vector n writes

    [n × curl ⋅ + iσ k n × (⋅ × n)) u(R) ≈ 0

For the plane wave we have

    pw(x) = p0 e^{iσ (k d⋅x - wt)}
    curl pw(x) = iσ kd × p0 e^{iσ (k d⋅x - wt)}

Hence

    [n × curl ⋅ + iσ k n × (⋅ × n)] pw(x) = iσ k [n × (d × pw(x)) + n × (pw(x) × n)]
"""
function absorbing_condition(pw::ElectromagneticPW)
    a = acoef(pw.medium)
    c = ccoef(pw.medium)
    k0 = pw.medium.k0
    return (x,ielt,n) -> im * (k0 * a(x) * (n × (pw.d × pw(x))) + c(x) * (n × (pw(x) × n)))
end
function neumann_condition(pw::ElectromagneticPW)
    a = acoef(pw.medium)
    k0 = pw.medium.k0
    return (x,ielt,n) -> im * k0 * a(x) * (n × (pw.d × pw(x)))
end

function get_planewave(medium::ElectromagneticMedium,θ0,ϕ0,pol)
    return ElectromagneticPW(medium,θ0,ϕ0,pol)
end

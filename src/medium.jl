############
# Acoustic #
############

"""
ρr and κr should be one-argument functions of the space variable.
"""
struct AcousticMedium <: Medium
    k0::Real # wavenumber in air k0 = ω √(ρ0 / κ0)
    ρr::Function # (relative) density ⩾ 1
    κr::Function # (relative) incompressibility ⩽ 1
    name::String
    function AcousticMedium(;k0=1, ρr=(x, ielt)->1, κr=(x, ielt)->1, name="")
        @assert k0 >= 0
        return new(k0, ρr, κr, name)
    end
end

"""
Produce a dissipative medium from a non-dissipative one.

No checks are performed. Compatible with a time-dependance in e^{-iωt}.
"""
dissipative_medium(m::AcousticMedium) = AcousticMedium(;k0=m.k0, ρr=(x,ielt)->im*m.ρr(x,ielt), κr=(x,ielt)->-im*m.κr(x,ielt))

"""
    c = √(κ / ρ)

c0 is the speed of light in vacuum c0 = √(κ0 / ρ0)
"""
speed_air(m::AcousticMedium) = 343
speed(m::AcousticMedium) = (x, ielt) -> 343 * sqrt(m.κr(x, ielt) / m.ρr(x, ielt))

"""
    k = ω √(ρ / κ)

k0 is the wavenumber in vacuum k0 = ω √(ρ0 / κ0)
"""
wavenumber(m::AcousticMedium) = (x, ielt) -> m.k0 * sqrt(m.ρr(x, ielt) / m.κr(x, ielt))

"""
Coefficient a in equation

    (- div a grad - b) u = f
"""
acoef(m::AcousticMedium) = (x, ielt) -> 1 / m.ρr(x, ielt)

"""
Coefficient b in equation

    (- div a grad - b) u = f
"""
bcoef(m::AcousticMedium) = (x, ielt) -> m.k0^2 / m.κr(x, ielt)

"""
Coefficient c in first order ABC

    (a n ⋅ grad - i c) u = g
"""
ccoef(m::AcousticMedium) = (x, ielt) -> m.k0 / sqrt(m.κr(x, ielt) * m.ρr(x, ielt))

####################
# Electromagnetism #
####################

"""
μr and ϵr should be one-argument functions of the space variable.
"""
struct ElectromagneticMedium <: Medium
    k0::Real # wavenumber in vacuum k0 = ω √(ϵ0 μ0)
    μr::Function # (relative) permeability ⩾ 1
    ϵr::Function # (relative) permittivity ⩾ 1
    name::String
    function ElectromagneticMedium(;k0=1, μr=(x,ielt)->1, ϵr=(x,ielt)->1, name="")
        @assert k0 >= 0
        return new(k0, μr, ϵr, name)
    end
end

"""
Produce a dissipative medium from a non-dissipative one.

No checks are performed. Compatible with a time-dependance in e^{-iωt}.
"""
dissipative_medium(m::ElectromagneticMedium) = ElectromagneticMedium(;k0=m.k0, μr=(x,ielt)->im*m.μr(x,ielt), ϵr=(x,ielt)->im*m.ϵr(x,ielt))

"""
    c = 1 / √(ϵμ)

c0 is the speed of light in vacuum c0 = 1 / √(ϵ0 μ0)
"""
speed_vacuum(m::ElectromagneticMedium) = 299792458
speed(m::ElectromagneticMedium) = (x, ielt) -> 299792458 / sqrt(m.ϵr(x, ielt) * m.μr(x, ielt))

"""
    k = ω √(ϵμ)

k0 is the wavenumber in vacuum k0 = ω √(ϵ0 μ0)
"""
wavenumber(m::ElectromagneticMedium) = (x, ielt) -> m.k0 * sqrt(m.ϵr(x, ielt) * m.μr(x, ielt))

"""
Coefficient a in equation

    (curl a curl - b) u = f
"""
acoef(m::ElectromagneticMedium) = (x, ielt) -> 1 / m.μr(x, ielt)

"""
Coefficient b in equation

    (curl a curl - b) u = f
"""
bcoef(m::ElectromagneticMedium) = (x, ielt) -> m.k0^2 * m.ϵr(x, ielt)

"""
Coefficient c in first order ABC

    (a n × curl - i c) u = g
"""
ccoef(m::ElectromagneticMedium) = (x, ielt) -> m.k0 * sqrt(m.ϵr(x, ielt) / m.μr(x, ielt))
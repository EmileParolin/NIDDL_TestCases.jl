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
    function AcousticMedium(;k0=1, ρr=x->1, κr=x->1, name="")
        @assert k0 >= 0
        return new(k0, ρr, κr, name)
    end
end

"""
Produce a dissipative medium from a non-dissipative one.

No checks are performed. Compatible with a time-dependance in e^{-iωt}.
"""
dissipative_medium(m::AcousticMedium) = AcousticMedium(;k0=m.k0, ρr=x->im*m.ρr(x), κr=x->-im*m.κr(x))

"""
    c = √(κ / ρ)

c0 is the speed of light in vacuum c0 = √(κ0 / ρ0)
"""
speed_air(m::AcousticMedium) = 343
speed(m::AcousticMedium) = x -> 343 * sqrt(m.κr(x) / m.ρr(x))

"""
    k = ω √(ρ / κ)

k0 is the wavenumber in vacuum k0 = ω √(ρ0 / κ0)
"""
wavenumber(m::AcousticMedium) = x -> m.k0 * sqrt(m.ρr(x) / m.κr(x))

"""
Coefficient a in equation

    (- div a grad - b) u = f
"""
acoef(m::AcousticMedium) = x -> 1 / m.ρr(x)

"""
Coefficient b in equation

    (- div a grad - b) u = f
"""
bcoef(m::AcousticMedium) = x -> m.k0^2 / m.κr(x)

"""
Coefficient c in first order ABC

    (a n ⋅ grad - i c) u = g
"""
ccoef(m::AcousticMedium) = x -> m.k0 / sqrt(m.κr(x) * m.ρr(x))

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
    function ElectromagneticMedium(;k0=1, μr=x->1, ϵr=x->1, name="")
        @assert k0 >= 0
        return new(k0, μr, ϵr, name)
    end
end

"""
Produce a dissipative medium from a non-dissipative one.

No checks are performed. Compatible with a time-dependance in e^{-iωt}.
"""
dissipative_medium(m::ElectromagneticMedium) = ElectromagneticMedium(;k0=m.k0, μr=x->im*m.μr(x), ϵr=x->im*m.ϵr(x))

"""
    c = 1 / √(ϵμ)

c0 is the speed of light in vacuum c0 = 1 / √(ϵ0 μ0)
"""
speed_vacuum(m::ElectromagneticMedium) = 299792458
speed(m::ElectromagneticMedium) = x -> 299792458 / sqrt(m.ϵr(x) * m.μr(x))

"""
    k = ω √(ϵμ)

k0 is the wavenumber in vacuum k0 = ω √(ϵ0 μ0)
"""
wavenumber(m::ElectromagneticMedium) = x -> m.k0 * sqrt(m.ϵr(x) * m.μr(x))

"""
Coefficient a in equation

    (curl a curl - b) u = f
"""
acoef(m::ElectromagneticMedium) = x -> 1 / m.μr(x)

"""
Coefficient b in equation

    (curl a curl - b) u = f
"""
bcoef(m::ElectromagneticMedium) = x -> m.k0^2 * m.ϵr(x)

"""
Coefficient c in first order ABC

    (a n × curl - i c) u = g
"""
ccoef(m::ElectromagneticMedium) = x -> m.k0 * sqrt(m.ϵr(x) / m.μr(x))

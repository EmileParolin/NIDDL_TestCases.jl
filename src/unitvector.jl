""" Defines a point on the unit sphere S^2.

Let M in the unit sphere and define P its projection on the plane z=0.
A unit vector is uniquely defined by
θ = (Oz, OM) in [0,π]
ϕ = (Ox, OP) in [0,2π[
"""
struct UnitVector
    θ::Float64
    ϕ::Float64
    function UnitVector(θ,ϕ)
        @assert 0 <= θ <= π && 0 <= ϕ < 2π
        return new(θ,ϕ)
    end
end
er(u::UnitVector) = [sin(u.θ)*cos(u.ϕ), sin(u.θ)*sin(u.ϕ), cos(u.θ)]
eθ(u::UnitVector) = [cos(u.θ)*cos(u.ϕ), cos(u.θ)*sin(u.ϕ),-sin(u.θ)]
eϕ(u::UnitVector) = [-sin(u.ϕ), cos(u.ϕ), 0.]

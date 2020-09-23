#########################
# Plane wave scattering #
#########################

mutable struct ScatteringTC <: TestCase
    d::Integer                 # dimension of problem
    pb_type::DataType          # type of propagative problem
    medium::Medium             # medium
    θ0::Real                   # incidence direction angle
    ϕ0::Real                   # incidence direction angle
    pol::Symbol                # polarisation (:θ, :ϕ)
    bcs::Vector{DataType}      # boundary conditions
end
function ScatteringTC(;d=3, medium=AcousticMedium(;k0=1), θ0=π/2, ϕ0=0, pol=:θ,
                      bcs=[RobinBC,])
    pb_type = problem_type(medium)
    hlm = (pb_type == HelmholtzPb && d in [2,3])
    mxl = pb_type == MaxwellPb && d in [3,]
    @assert hlm || mxl "Problem type $(pb_type) uncompatible with dimension d = $(d)"
    # incidence direction
    ud = UnitVector(θ0,ϕ0) # (π,0) gives -z
    dinc = er(ud)
    @info "Incidence direction $dinc ($θ0,$ϕ0)"
    # polarisation
    if pb_type == MaxwellPb
        if (pol==:θ) p = eθ(ud) elseif (pol==:ϕ) p = eϕ(ud) end
        @info "Polarisation $p ($pol)"
    end
    return ScatteringTC(d,pb_type,medium,θ0,ϕ0,pol,bcs)
end
function (tc::ScatteringTC)(Γs::Array{Domain,1})
    msg = "Mismatch between number of physical boundaries and number of
           physical boundary conditions"
    @assert length(tc.bcs) <= length(Γs) msg
    # Initisation
    pbc = BoundaryCondition[]
    # Plane wave
    pw = get_planewave(tc.medium,tc.θ0,tc.ϕ0,tc.pol)
    f_Robin = absorbing_condition(pw)
    f_Neumann = neumann_condition(pw)
    # Interior boundary condition
    if length(tc.bcs) == 2
        if tc.bcs[1] == DirichletBC
            bcΓint = DirichletBC(Γs[1],Complex{Float64}(0))
        elseif tc.bcs[1] == DirichletWeakBC
            if tc.pb_type == HelmholtzPb
                bcΓint = DirichletWeakBC(Γs[1],Complex{Float64}(0))
            elseif tc.pb_type == MaxwellPb
                bcΓint = DirichletWeakBC(Γs[1],zeros(Complex{Float64},3))
            else
                error("Not a valid problem type.")
            end
        elseif tc.bcs[1] == NeumannBC
            if tc.pb_type == HelmholtzPb
                bcΓint = NeumannBC(Γs[1],(args...)->Complex{Float64}(0))
            elseif tc.pb_type == MaxwellPb
                bcΓint = NeumannBC(Γs[1],(args...)->zeros(Complex{Float64},3))
            else
                error("Not a valid problem type.")
            end
        else
            error("Not implemented")
        end
        push!(pbc, bcΓint)
    end
    # Exterior boundary condition
    if tc.bcs[end] == RobinBC
        # First order absorbing BC
        bcΓext = RobinBC(Γs[end], x -> im*ccoef(tc.medium)(x), f_Robin)
    elseif tc.bcs[end] == NeumannBC
        bcΓext = NeumannBC(Γs[end],f_Neumann)
    elseif tc.bcs[end] == DirichletBC
        bcΓext = DirichletBC(Γs[end],Complex{Float64}(1))
    elseif tc.bcs[end] == DirichletWeakBC
        if tc.pb_type == HelmholtzPb
            bcΓext = DirichletWeakBC(Γs[end],Complex{Float64}(1))
        elseif tc.pb_type == MaxwellPb
            bcΓext = DirichletWeakBC(Γs[end],ones(Complex{Float64},3))
        else
            error("Not a valid problem type.")
        end
    else
        error("Not implemented")
    end
    push!(pbc, bcΓext)
    return pbc
end

function get_name(tc::ScatteringTC)
    name = "$(typeof(tc))_$(tc.pb_type)_$(tc.d)D_k$(tc.medium.k0)_$(tc.medium.name)"
    name *= "_theta$(round(1e3*tc.θ0)/1e3)_phi$(round(1e3*tc.ϕ0)/1e3)"
    if tc.pb_type == MaxwellPb name *= "_pol$(tc.pol)" end
    name = prod(vcat([name,], ["_$(bc)" for bc in tc.bcs]))
    return name
end


#################
# Random source #
#################

struct RandomTC <: TestCase
    d::Integer                 # dimension of problem
    pb_type::DataType          # type of propagative problem
    medium::Medium             # medium
end
function RandomTC(;d=3, medium=medium)
    pb_type = problem_type(medium)
    return RandomTC(d,pb_type,medium)
end
function (tc::RandomTC)(Γs::Array{Domain,1})
    if tc.pb_type == HelmholtzPb
        func = (x,ielt,n) -> (Random.seed!(Int(floor(1.e6*sum(abs.(vcat(x,ielt,n))))));
                              (2*rand(Complex{Float64})-1))
    elseif tc.pb_type == MaxwellPb
        func = (x,ielt,n) -> (Random.seed!(Int(floor(1.e6*sum(abs.(vcat(x,ielt,n))))));
                              (2 .*rand(Complex{Float64},3).-1))
    end
    return [RobinBC(Γ, x -> im*ccoef(tc.medium)(x), func) for Γ in Γs]
end

function get_name(tc::RandomTC)
    name = "$(typeof(tc))_$(tc.pb_type)_$(tc.d)D_k$(tc.medium.k0)"
    return name
end

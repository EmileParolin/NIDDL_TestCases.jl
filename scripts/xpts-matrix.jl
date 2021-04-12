using Revise
using Pkg
Pkg.activate("./")
Pkg.update("NIDDL_FEM")
Pkg.update("NIDDL")
using LinearAlgebra
using SparseArrays
using SuiteSparse
using SharedArrays
using Distributed
using LinearMaps
using IterativeSolvers
using TimerOutputs
using NIDDL_FEM
using NIDDL
using NIDDL_TestCases
using Test
using JLD
using PGFPlots
prefix = pwd() * "/data/"
include("./TriLogLog.jl")
include("./postprod.jl")

## General parameters
function coef_r(x, Δc)
    r = norm(x)
    θ = atan(x[2], x[1])
    ρ = (2/3) * (1 + cos(6θ) / 6)
    if r > ρ
        return 1
    elseif r < ρ/5
        return 2Δc
    else
        ψ = (1 + cos(6θ) / 2)
        return 1 + Δc * ψ
    end
end
function daidai(;name="eraseme")
    d = 2
    pb_type = VectorHelmholtzPb
    k = 5
    Nλ = 250
    as = [1,]
    nΩ = 25
    ϵr = x -> coef_r(x, 3/2)
    μr = x -> coef_r(x, 5/2)
    medium_E = AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->ϵr(x))
    medium = AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->1/ϵr(x))
    tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
    tp = DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
    dd = JunctionsDDM(;implicit=true, precond=true)
    # Geometry, mesh and domains
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                interior=true, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h;);
    Ω = union(Ωs...);
    #save_medium(m, Ω, medium_E, prefix*"xpts-matrix-medium_Maxwell")
    #save_medium(m, Ω, medium, prefix*"xpts-matrix-medium_acoustic")
    # Solver
    solver = Jacobi_S(;tol=1.e-12, maxit=50, r=0.5, light_mode=false)
    solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=false)
    # Problems
    fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
    # Exact discrete solution
    to = TimerOutput()
    @timeit to "Full problem resolution" uexact = solve(m, fullpb);
    # DDM
    gid = InputData(m, fullpb, pbs);
    ddm = DDM(pbs, gid, dd; to=to);
    @timeit to "Resfunc setup" resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact,
        solver; save_solutions_it=false, prefix=prefix);
    @timeit to "Solver" u, x, res = solver(ddm; resfunc=resfunc, to=to);
    # Output
    save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, name);
    JLD.save(prefix*name*".jld",
        "res", res, "tp", typeof(tp), "k", medium.k0, "Nlambda", Nλ, "Nomega", nΩ,
        "medium", medium.name, "nl", g.nl, "cg_min", ddm.gd.cg_min, "cg_max",
        ddm.gd.cg_max, "cg_sum", ddm.gd.cg_sum,)
    return u, x, res, ddm
end

##
name = "eraseme"
u, x, res, ddm = daidai(; name=name)
ax = generate_conv_plot([name,]; dir=prefix)
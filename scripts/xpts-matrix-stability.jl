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

##
function daidai(; k = 1, Nλ = 20, nΩ = 4, name="eraseme", op=:Id)
    d = 2
    pb_type = VectorHelmholtzPb
    as = [1,]
    medium = AcousticMedium(;k0=k)
    tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
    tp = op == :Id ? DespresTP(;z=1) : DtN_TP(;z=1,pb_type=pb_type,medium=dissipative_medium(medium),fbc=:robin)
    dd = JunctionsDDM(;implicit=true, precond=true)
    # Geometry, mesh and domains
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                interior=true, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h;);
    Ω = union(Ωs...);
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
Nλs = 10 .* 2 .^ collect(1:1:6)

##
for Nλ in Nλs
    name = "stability_2D_Nl$(Nλ)";
    u, x, res, ddm = daidai(;k=1, Nλ=Nλ, nΩ=4, name=name*"_Despres", op=:Id);
    u, x, res, ddm = daidai(;k=1, Nλ=Nλ, nΩ=4, name=name*"_DtN",     op=:DtN);
end

##
names = ["stability_2D_Nl$(Nλ)" * endname
        for Nλ in Nλs, endname in ["_Despres", "_DtN"]]
##
ax = generate_param_plot(names; dir=prefix,
                             param_type=:Nl,
                             tol=1.e-8, ertype=:HD,
                             fullgmres=false,
                             marks=["o", "+", "square", "asterisk", "diamond",
                                    "triangle",],
                             colorstyles=["black", "red", "blue", "teal", "cyan",
                                          "orange", "magenta",],
                             styles=["solid" for _ in 1:8],
                             tll=TriLogLog[],
                             func_on_abscissa=x->x)
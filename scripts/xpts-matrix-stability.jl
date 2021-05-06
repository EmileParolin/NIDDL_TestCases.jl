using Revise
using Pkg
Pkg.activate("./")
#Pkg.update("NIDDL_FEM")
#Pkg.update("NIDDL")
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
#using Test
using JLD
#using PGFPlots
prefix = pwd() * "/data/"
#include("./TriLogLog.jl")
#include("./postprod.jl")

##
function daidai(; d=2, k=1, Nλ=20, nΩ=4, a=1, name="eraseme", op=:Id)
    pb_type = d == 2 ? VectorHelmholtzPb : MaxwellPb
    as = [a,]
    medium = d == 2 ? AcousticMedium(;k0=k) : ElectromagneticMedium(;k0=k)
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
for Nλ in Nλs
    name = "stability_2D_Nl$(Nλ)";
    u, x, res, ddm = daidai(;k=1, Nλ=Nλ, nΩ=4, name=name*"_Despres", op=:Id);
    u, x, res, ddm = daidai(;k=1, Nλ=Nλ, nΩ=4, name=name*"_DtN",     op=:DtN);
end

##
names = ["stability_2D_Nl$(Nλ)" * endname
        for Nλ in Nλs, endname in ["_Despres", "_DtN"]]
# Removing last result because issue
names = names[1:end-1,:]
ax = generate_param_plot(names; dir=prefix,
                             param_type=:Nl,
                             tol=1.e-8, ertype=:HD,
                             fullgmres=false,
                             marks=["o", "+", "square", "asterisk", "diamond",
                                    "triangle",],
                             colorstyles=["black", "red", "blue", "teal", "cyan",
                                          "orange", "magenta",],
                             styles=["solid" for _ in 1:8],
                             tll=[TriLogLog(1.5, 1.7, 0.75, 1),],
                             func_on_abscissa=x->x)
ax.plots[1].legendentry = "Despr\\'es"
ax.plots[2].legendentry = "Schur"
ax.xmin = 10^1
ax.xmax = 10^3
ax.xlabel = "Mesh refinement \$\\lambda/h\$"
PGFPlots.save(prefix*"xpts-matrix-stability_2D.pdf", ax)
ax

##
names = ["stability_2D_Nl$(Nλ)" * endname
        for Nλ in Nλs, endname in ["_Despres", "_DtN"]]
ax = generate_param_plot(names; dir=prefix,
                             param_type=:Nl,
                             tol=1.e-8, ertype=:cg_max,
                             fullgmres=false,
                             marks=["o", "+", "square", "asterisk", "diamond",
                                    "triangle",],
                             colorstyles=["black", "red", "blue", "teal", "cyan",
                                          "orange", "magenta",],
                             styles=["solid" for _ in 1:8],
                             tll=TriLogLog[],
                             func_on_abscissa=x->x)
<<<<<<< HEAD
ax.plots[1].legendentry = "Despr\\'es"
ax.plots[2].legendentry = "Schur"
ax.xmin = 10^1
ax.xmax = 10^3
ax.xlabel = "Mesh refinement \$\\lambda/h\$"
ax.ylabel = "Maximum number of inner  CG iterations"
PGFPlots.save(prefix*"xpts-matrix-stability-cgmax_2D.pdf", ax)
ax
=======

##
Nλs = 10 .* 2 .^ collect(4.5:-0.5:1)
for Nλ in Nλs
    name = "stability_3D_Nl$(Nλ)";
    u, x, res, ddm = daidai(;d=3, k=1, Nλ=Nλ, nΩ=32, a=0.5, name=name*"_Despres", op=:Id);
    u, x, res, ddm = daidai(;d=3, k=1, Nλ=Nλ, nΩ=32, a=0.5, name=name*"_DtN",     op=:DtN);
end
>>>>>>> 786a69cd9eeea033ea3236975b111e47e05083a4

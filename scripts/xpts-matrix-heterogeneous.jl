#using Revise
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
using PGFPlots
prefix = pwd() * "/data/"
include("./TriLogLog.jl")
include("./postprod.jl")

## General parameters
function coef_r(x, Δc; d=2)
    r = norm(x)
    θ = d==2 ? atan(x[2], x[1]) : atan(norm(x[1:2]), x[3])
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

##
function daidai(; d = 2, k = 5, Nλ = 250, nΩ = 25, a = 1, name="eraseme", heterogeneous=true, elliptic=false, light_mode=false)
    pb_type = d == 2 ? VectorHelmholtzPb : MaxwellPb
    as = [a,]
    if heterogeneous
        if elliptic
            ϵr = x -> coef_r(x, 3/2) + im * coef_r(x, 3/2) / 6
            μr = x -> coef_r(x, 5/2) + im * coef_r(x, 5/2) / 4
            #medium_E = AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->ϵr(x))
            medium = d == 2 ? AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->1/ϵr(x)) : ElectromagneticMedium(;k0=k, μr=x->μr(x), ϵr=x->ϵr(x))
        else
            ϵr = x -> coef_r(x, 3/2)
            μr = x -> coef_r(x, 5/2)
            #medium_E = AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->ϵr(x))
            medium = d == 2 ? AcousticMedium(;k0=k, ρr=x->μr(x), κr=x->1/ϵr(x)) : ElectromagneticMedium(;k0=k, μr=x->μr(x), ϵr=x->ϵr(x))
        end
    else
        medium = d == 2 ? AcousticMedium(;k0=k) : ElectromagneticMedium(;k0=k)
    end
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
    solver = GMRES_S(;tol=1.e-12, maxit=10000, light_mode=light_mode)
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
    #save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, name);
    JLD.save(prefix*name*".jld",
             "res", res, "tp", typeof(tp), "k", medium.k0, "Nlambda", Nλ, "Nomega", nΩ,
        "medium", medium.name, "nl", g.nl, "cg_min", ddm.gd.cg_min, "cg_max",
        ddm.gd.cg_max, "cg_sum", ddm.gd.cg_sum,)
    return u, x, res, ddm
end

##
for k in 1:5
    nΩ = 25
    ## Heterogeneous
    #Nλ = 250
    #name = "heterogeneous_2D_k$(k)_Nl$(Nλ)_n$(nΩ)";
    #u, x, res, ddm = daidai(;k=k, Nλ=Nλ, nΩ=nΩ, name=name);
    #ax = generate_conv_plot([name,]; dir=prefix);
    ## Homogeneous
    #corr = 2.24 * 1.74 # Product of the means
    #corr = 5.2         # Mean of the product
    #Nλ = Int(floor(250 / sqrt(corr)))
    #k *= sqrt(corr)
    #name = replace("homogeneous_2D_k$(Int64(floor(1000*k))/1000)_Nl$(Nλ)_n$(nΩ)", "."=>"d");
    #u, x, res, ddm = daidai(;k=k, Nλ=Nλ, nΩ=nΩ, name=name, heterogeneous=false);
    #ax = generate_conv_plot([name,]; dir=prefix);
    # Elliptic
    Nλ = 250
    name = replace("elliptic_2D_k$(k)_Nl$(Nλ)_n$(nΩ)", "."=>"d");
    u, x, res, ddm = daidai(;k=k, Nλ=Nλ, nΩ=nΩ, name=name, elliptic=true);
    ax = generate_conv_plot([name,]; dir=prefix);
end

##
names_heterogeneous = ["heterogeneous_2D_k1_Nl250_n25",
                       "heterogeneous_2D_k2_Nl250_n25",
                       "heterogeneous_2D_k3_Nl250_n25",
                       "heterogeneous_2D_k4_Nl250_n25",
                       "heterogeneous_2D_k5_Nl250_n25",]
names_homogeneous   = ["homogeneous_2D_k2d28_Nl109_n25",
                       "homogeneous_2D_k4d56_Nl109_n25",
                       "homogeneous_2D_k6d841_Nl109_n25",
                       "homogeneous_2D_k9d121_Nl109_n25",
                       "homogeneous_2D_k11d401_Nl109_n25",]
names_elliptic   = ["elliptic_2D_k1_Nl250_n25",
                    "elliptic_2D_k2_Nl250_n25",
                    "elliptic_2D_k3_Nl250_n25",
                    "elliptic_2D_k4_Nl250_n25",
                    "elliptic_2D_k5_Nl250_n25",]
for (k, hom, het, ell) in zip(collect(1:5), names_homogeneous, names_heterogeneous, names_elliptic)
    ax = generate_conv_plot([het, hom, ell]; dir=prefix);
    ax.plots[1].legendentry = "Heterogeneous"
    ax.plots[2].legendentry = "Homogeneous"
    ax.plots[3].legendentry = "Dissipative"
    ax.ymin = 5e-9
    PGFPlots.save(prefix*"xpts-matrix-heterogeneous_2D_k$(k)_cvplot.pdf", ax)
end

##
for k in 1:1
    nΩ = 50
    # Heterogeneous
    Nλ = 150
    name = "heterogeneous_3D_k$(Int64(floor(1000*k))/1000)_Nl$(Nλ)_n$(nΩ)";
    u, x, res, ddm = daidai(;d=3, k=k, Nλ=Nλ, nΩ=nΩ, name=name);
    #ax = generate_conv_plot([name,]; dir=prefix);
    # Homogeneous
    corr = 2.24 * 1.74 # Product of the means
    corr = 5.2         # Mean of the product
    Nλ = Int(floor(Nλ / sqrt(corr)))
    k *= sqrt(corr)
    name = replace("homogeneous_3D_k$(Int64(floor(1000*k))/1000)_Nl$(Nλ)_n$(nΩ)", "."=>"d");
    u, x, res, ddm = daidai(;d=3, k=k, Nλ=Nλ, nΩ=nΩ, name=name, heterogeneous=false);
    #ax = generate_conv_plot([name,]; dir=prefix);
end

##
names_heterogeneous = ["heterogeneous_3D_k1.0_Nl150_n50",]
names_homogeneous   = ["homogeneous_3D_k2d28_Nl65_n50",]
for (k, hom, het) in zip(collect(1:1), names_homogeneous, names_heterogeneous)
    ax = generate_conv_plot([het, hom]; dir=prefix);
    ax.plots[1].legendentry = "Heterogeneous"
    ax.plots[2].legendentry = "Homogeneous"
    ax.ymin = 5e-9
    PGFPlots.save(prefix*"xpts-matrix-heterogeneous_3D_k$(k)_cvplot.pdf", ax)
end
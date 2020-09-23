using Test
using NIDDL_FEM
using NIDDL
using NIDDL_TestCases
include("../src/gmsh.jl")

function andiamo(;geos=Geometry[], Nλs=[20,], tcs=TestCase[],
                 tps=TransmissionParameters[], solvers=Solver[],
                 exchange_types=[:basic,], modes=[:implicit])
    @assert length(exchange_types) == length(tps)
    # Runs
    Nits_test = Int64[]
    for (ig, g) in enumerate(geos)
        for (ih, Nλ) in enumerate(Nλs)
            # Mesh and domains
            h = 2π/abs(maximum([tc.medium.k0 for tc in tcs])) / max(10, Nλ)
            m, Ωs, Γs = get_mesh_and_domains(g, h)
            for (ic, tc) in enumerate(tcs)
                for (it, (tp,exchange_type)) in enumerate(zip(tps,exchange_types))
                    for (iM, mode) in enumerate(modes)
                        # Problems
                        fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp; exchange_type=exchange_type)
                        # Exact discrete solution
                        uexact = solve(m,fullpb)
                        # DDM
                        ddm = femDDM(m, fullpb, pbs; mode=mode,
                                     exchange_type=exchange_type)
                        for (is, solver) in enumerate(solvers)
                            solver.x .*= 0
                            resfunc = get_resfunc(m, fullpb, pbs, ddm, uexact, solver);
                            u,x,res = solver(ddm; resfunc=resfunc);
                            # Iteration count
                            push!(Nits_test, length(res[Inf.>res[:,1].>0,1]))
                        end
                    end
                end
            end
        end
    end
    return Nits_test
end

Nits = Dict((2, AcousticMedium, :basic)
            =>[10, 9, 10, 9, 18, 16, 18, 16, 8, 7, 8, 7, 8, 11, 8, 11, 16,
               17, 15, 17, 8, 9, 8, 9, 7, 13, 7, 13, 17, 12, 16, 12, 8, 10,
               8, 10],
            (2, AcousticMedium, :xpts)
            =>[11, 12, 9, 12, 9, 16, 9, 16, 9, 9, 9, 9, 9, 14, 9, 14, 9, 18,
               9, 18, 9, 9, 9, 9, 9, 19, 8, 19, 8, 13, 8, 13, 7, 10, 7, 10],
            (3, AcousticMedium, :basic)
            =>[91, 31, 3, 31, 9, 17, 9, 17, 14, 13, 15, 13, 82, 63, 15, 63, 27,
               21, 25, 21, 15, 15, 16, 15, 13, 34, 11, 34, 24, 19, 19, 19, 12,
               17, 13, 17],
            (3, AcousticMedium, :xpts)
            =>[89, 41, 3, 41, 3, 32, 3, 32, 2, 30, 18, 30, 237, 90, 9, 90, 5,
               36, 14, 36, 5, 35, 22, 35, 62, 44, 14, 44, 9, 38, 15, 38, 20,
               34, 19, 34],
            (3, ElectromagneticMedium, :basic)
            =>[20, 12, 18, 12, 14, 15, 15, 15, 9, 7, 10, 7, 15, 18, 14, 18, 52,
               30, 47, 30, 8, 9, 10, 9, 9, 21, 9, 21, 41, 29, 35, 29, 8, 10,
               9, 10],
            (3, ElectromagneticMedium, :xpts)
            =>[15, 14, 14, 14, 10, 15, 10, 15, 8, 10, 8, 10, 14, 23, 13, 23,
               10, 26, 11, 26, 8, 10, 8, 10, 8, 27, 8, 27, 8, 28, 8, 28, 8, 11,
               8, 11],
           )


@testset "All tests" begin
    @testset "Non-regression" begin
        k = 1
        for (d,medium) in [(2,AcousticMedium(;k0=k)),
                           (3,AcousticMedium(;k0=k)),
                           (3,ElectromagneticMedium(;k0=k)),
                          ]
            @testset "$(typeof(medium)) $(d)D" begin
                Nλs = [20,]
                geos = [LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[0.5, 1,],
                                interior=true, nΩ=2, mode=:cad, nl=0),
                        LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[0.5, 1,],
                                interior=true, nΩ=2, mode=:geo, nl=0),
                        LayersGeo(;d=d, shape=[:circle, :sphere][d-1], as=[     1,],
                                interior=true, nΩ=2, mode=:metis, nl=0), ]
                tcs = [ScatteringTC(;d=d, medium=medium, θ0=π/2, ϕ0=0, bcs=[RobinBC,]),]
                modes=[:explicit, :implicit,];
                solvers = [Jacobi_S(;tol=1e-1, maxit=1000, r=0.5, light_mode=true),
                           GMRES_S(;tol=1e-3, maxit=1000, light_mode=true), ]
                @testset "Exchange type : standard" begin
                    tps = [DespresTP(;z=1), SndOrderTP(;z=1,α=1/(2*k^2)),
                           DtN_neighbours_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin),]
                    Nits_test = andiamo(geos=geos, Nλs=Nλs, tcs=tcs, tps=tps,
                                        solvers=solvers, exchange_types=[:basic for _ in tps],
                                        modes=modes);
                    @show Nits_test
                    @test Nits[(d, typeof(medium), :basic)] == Nits_test
                end
                @testset "Exchange type : projection" begin
                    tps = [DespresTP(;z=1), SndOrderTP(;z=1,α=1/(2*k^2)),
                           DtN_TP(;z=1,medium=dissipative_medium(medium),fbc=:robin),]
                    Nits_test = andiamo(geos=geos, Nλs=Nλs, tcs=tcs, tps=tps,
                                        solvers=solvers, exchange_types=[:xpts for _ in tps],
                                        modes=modes);
                    @show Nits_test
                    @test Nits[(d, typeof(medium), :xpts)] == Nits_test
                end
            end
        end
    end
end

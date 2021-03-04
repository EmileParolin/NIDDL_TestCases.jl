using Revise, Pkg, Test
Pkg.activate("./")
using LinearAlgebra, SparseArrays, SuiteSparse, SharedArrays, Distributed
using LinearMaps, IterativeSolvers, TimerOutputs
using NIDDL_FEM, NIDDL, NIDDL_TestCases
using LaTeXStrings, Colors, PGFPlotsX
prefix = pwd() * "/data/"


function pgfplotsx_spy(M; separators=Int64[], tol=1.e-12)
    # Preparing data
    @assert size(M,1) == size(M,2)
    N = size(M,1)
    S = sparse(abs.(M))
    i, j, v = findnz(S)
    b = v .> tol
    x = j[b]
    y = N+1 .- i[b]
    c = v[b]
    # Adding down left and top right elements to please PGFPlots
    x = vcat([1, N], x)
    y = vcat([1, N], y)
    c = vcat([tol, tol], c)
    # Plotting set up
    td = TikzDocument()
    cmapname = "MyColorMap"
    cmap = Colors.sequential_palette(255, 100; c=1)
    push_preamble!(td, (cmapname, cmap))
    tp = @pgf TikzPicture()
    push!(td, tp)
    axis = @pgf Axis({
            "axis equal" = true,
            "hide axis" = true,
            "enlargelimits" = true,
            "colorbar" = false,
            "colormap name" = cmapname,
        }, )
    push!(tp, axis)
    # Plotting elements
    p = @pgf Plot(
        {
            scatter,
            "only marks",
            "mark" = "square*",
            "mark size" = 50 / N,
            "scatter/use mapped color" = "{draw opacity=0,fill=mapped color}",
            "point meta" = {"ln(abs(\\thisrow{c}))/ln(10)"},
        },
        Table(x = x, y = y, c = c,)
    )
    push!(axis, p)
    # Plotting separators
    sl = [0.5, N+0.5]
    for s in separators
        @pgf push!(axis, Plot({thick, solid, red},
                              Coordinates(sl, (N+1 .- s - 0.5) .* [1, 1])))
        @pgf push!(axis, Plot({thick, solid, red},
                              Coordinates((s + 0.5) .* [1, 1], sl)))
    end
    return td
end


function plot_sparsity_pattern(;d=2, k=1, Nλ=10, nΩ=2, non_local=true)
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d-1], as=[1,],
                interior=true, nΩ=nΩ, mode=:metis)
    h = 2π/abs(k) / max(5, Nλ)
    m, Ωs, Γs = get_mesh_and_domains(g, h, name=prefix*"eraseme");
    save_partition(m, union(Ωs...), prefix*"eraseme")
    medium = AcousticMedium(;k0=k)
    tc = ScatteringTC(;d=d, medium=medium, bcs=[RobinBC,])
    tps =  [Idl2TP(;z=1),
            DespresTP(;z=1),
            SndOrderTP(;z=1,α=h/(2*k^2)),
            ]
    if non_local
        push!(tps, 
              HS_TP(;z=1,ν=im*k),
              DtN_TP(;z=1,medium=dissipative_medium(medium)),
              #invS_TP(;z=1,ν=im*k),
              #LsL_H2D_TP(;z=1,ν=im*k,β=1,δ=(-1.0,0.0)),
              #DtN_neighbours_TP(;z=1,medium=dissipative_medium(medium)),
             )
    end
    for tp in tps
        r = Run(g, tc, m, Ωs, Γs, tp, GMRES_S(light_mode=false); Nλ=Nλ,
                exchange_type=:xpts, mode=:explicit);
        # Exchange operator matrix
        Πmat = vcat([ld.Πi.Πi for ld in r.ddm.lds]...);
        # Block separators
        separators = vcat(0, [maximum(findnz(ld.Πi.MΣitoΣmt)[1]) for ld in r.ddm.lds]...)
        # Plot and saving
        name = "manuscript_sparsity_pattern_$(d)D_J$(nΩ)_k$(k)_Nl$(Nλ)_$(typeof(tp))"
        a = pgfplotsx_spy(Πmat; separators=separators)
        pgfsave(prefix*replace(name, "."=>"d")*".pdf", a)
    end
end

plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=2)
plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=3)
plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=4)
plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=6)
plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=8)
plot_sparsity_pattern(;d=2, k=1, Nλ=20, nΩ=10)

plot_sparsity_pattern(;d=2, k=10, Nλ=10, nΩ=10, non_local=true)
plot_sparsity_pattern(;d=2, k=20, Nλ=5, nΩ=10, non_local=true)
plot_sparsity_pattern(;d=2, k=40, Nλ=2, nΩ=10, non_local=true)
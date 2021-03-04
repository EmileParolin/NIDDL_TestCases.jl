using Revise, Pkg, Test
Pkg.activate("./")
using LinearAlgebra, SparseArrays, SuiteSparse, SharedArrays, Distributed
using LinearMaps, IterativeSolvers, TimerOutputs, WriteVTK
using NIDDL_FEM, NIDDL, NIDDL_TestCases
using Colors, PGFPlotsX
prefix = pwd() * "/data/"

## Single to Multi trace and reverse
function st2mt(ddm, x, y)
    for ld in ddm.lds
        y += (ld.Πi.MΣitoΣmt * (transpose(ld.Πi.MΣitoΣst) * x))
    end
    return y
end
function mt2st(gid, ddm, x, y)
    ind_Ω = indices_full_domain(gid)
    ind_Σ = indices_skeleton(gid)
    MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
    weights = MΩtoΣ * (1 ./ dof_weights(gid))
    for ld in ddm.lds
        y += weights .* (ld.Πi.MΣitoΣst * (transpose(ld.Πi.MΣitoΣmt) * x))
    end
    return y
end
function mt2vtx(gid, m, Ω, Ωs, ddm, i)
    Nst = number_of_elements(m, skeleton(Ω), 0)
    Nmt = sum([number_of_elements(m, boundary(ω), 0) for ω in Ωs])
    ind_Ω = indices_full_domain(gid)
    ind_Σ = indices_skeleton(gid)
    MΩtoΣ = mapping_from_global_indices(ind_Ω, ind_Σ)
    x = zeros(Nmt)
    x[i] = 1.0
    y = zeros(Nst)
    y = mt2st(gid, ddm, x, y)
    j = argmax(y)
    return m.vtx[:,m.nod2vtx[:,j]]
end

## Indices in domain boundaries
function inds_boundaries(nΩ, ddm, Nst, Nmt)
    inds = Vector{Int64}[]
    for i in 1:Nst
        xst = zeros(Nst)
        xst[i] = 1.
        xmt = zeros(Nmt)
        xmt = st2mt(ddm, xst, xmt)
        xmts = [Int.(transpose(ld.MΣitoΣmt) * xmt) for ld in ddm.lds]
        ind = argmax.(xmts)
        for j in 1:nΩ
            if ind[j] == 1
                if xmts[j][ind[j]] == 0 
                    ind[j] = 0
                end
            end
        end
        push!(inds, ind)
    end
    return inds
end

## Saving function
function vtk_save_on_skeleton(m, Ωs, xmts, label, name)
    vtmfile = vtk_multiblock(name)
    # Points
    points = m.vtx
    # Loop on sub-domains
    for (xi, Ω) in zip(xmts, Ωs)
        Γ = boundary(Ω)
        # Cells (lines)
        cells = MeshCell[]
        for it in element_indices(m, Γ, 1)
            push!(cells, MeshCell(VTKCellTypes.VTK_LINE, m.edg2vtx[:,it]))
        end
        # File
        vtkfile = vtk_grid(vtmfile, points, cells)
        # Data
        pdata = zeros(size(points, 2))
        pdata[m.nod2vtx[1, element_indices(m, Γ, 0)]] = xi
        vtkfile[label, VTKPointData()] = pdata
    end
    vtk_save(vtmfile)
end

## Plotting
function draw_error_vs_dissipation_plot(dists, errs, diss)
    # Plotting set up
    td = TikzDocument()
    cmapname = "MyColorMap"
    cmap = Colors.sequential_palette(255, 100; c=1)
    cmap = reverse(Colors.colormap("RdBu"; logscale=false))
    push_preamble!(td, (cmapname, cmap))
    tp = @pgf TikzPicture()
    push!(td, tp)
    axis = @pgf Axis({
            "xlabel" = "Dissipation parameter \$\\nu\$",
            "ylabel" = "Error",
            "axis equal" = false,
            "hide axis" = false,
            "grid" = "both",
            "enlargelimits" = {abs = 2},
            "colorbar" = false,
            xmode = "log",
            ymode = "log",
            ymin = 1.e-16,
            ymax = 1.e0,
        }, )
    push!(tp, axis)
    # Plotting elements
    p = @pgf Plot(
        {
            scatter,
            "only marks",
            "mark" = "square*",
            "mark size" = 2,
            "scatter/use mapped color" = "{draw opacity=0,fill=mapped color}",
            "point meta" = {"ln(abs(\\thisrow{c}))/ln(10)"},
        },
        Table(x = diss, y = errs, c = dists,)
    )
    push!(axis, p)
    return td
end
function draw_error_vs_distance_plot(dists, errs, diss)
    # Plotting set up
    td = TikzDocument()
    cmapname = "MyColorMap"
    cmap = Colors.sequential_palette(255, 100; c=1)
    cmap = reverse(Colors.colormap("RdBu"; logscale=false))
    push_preamble!(td, (cmapname, cmap))
    tp = @pgf TikzPicture()
    push!(td, tp)
    axis = @pgf Axis({
            "xlabel" = "Distance to source",
            "ylabel" = "Error",
            "axis equal" = false,
            "hide axis" = false,
            "grid" = "both",
            "enlargelimits" = {abs = 0.1},
            "colorbar" = false,
            ymode = "log",
            ymin = 1.e-16,
            ymax = 1.e0,
        }, )
    push!(tp, axis)
    # Plotting elements
    p = @pgf Plot(
        {
            scatter,
            "only marks",
            "mark" = "square*",
            "mark size" = 2,
            "scatter/use mapped color" = "{draw opacity=0,fill=mapped color}",
            "point meta" = {"ln(abs(\\thisrow{c}))/ln(10)"},
        },
        Table(x = dists, y = errs, c = diss,)
    )
    push!(axis, p)
    return td
end

## Helper function
function run(k, Nλ, as, nΩ, νs, name)
    d = 2
    g = LayersGeo(;d=d, shape=[:circle,:sphere,][d - 1], as=as,
                  interior=true, nΩ=nΩ, mode=:metis)
    h = 2π / abs(k) / max(5, Nλ);
    m, Ωs, Γs = get_mesh_and_domains(g, h; name=prefix * name);
    Ω = union(Ωs...);
    save_partition(m, [Domain(ω) for ω in Ω], prefix * name * "_partition");
    # Test case
    medium = AcousticMedium(;k0=k)
    pb_type = HelmholtzPb
    tc = ScatteringTC(;d=d, pb_type=pb_type, medium=medium, bcs=[RobinBC,])
    # DDM type
    dd = JunctionsDDM(;implicit=true, precond=true)
    ## Reference
    # Problems
    tp = Idl2TP(;z=1)
    fullpb_ref, pbs_ref = get_problems(g, tc, Ωs, Γs, tp, dd);
    # DDM
    gid_ref = InputData(m, fullpb_ref, pbs_ref);
    ddm_ref = DDM(pbs_ref, gid_ref, dd);
    Π_ref = ddm_ref.A.Π;
    ## Ramp in dissipation parameter
    Πs = GlobalExchangeOp{JunctionsGlobalData,JunctionsLocalData}[]
    for ν in νs
        # Problems
        diss_medium = dissipative_medium(AcousticMedium(;k0=ν))
        tp = DtN_TP(;z=1,pb_type=pb_type,medium=diss_medium,fbc=:robin)
        fullpb, pbs = get_problems(g, tc, Ωs, Γs, tp, dd);
        # DDM
        gid = InputData(m, fullpb, pbs);
        ddm = DDM(pbs, gid, dd);
        push!(Πs, ddm.A.Π);
    end
    return m, Ω, Ωs, gid_ref, ddm_ref, Π_ref, Πs
end

## General parameters
k = 1
νs = 2. .^ (-2:0.5:6)
Nλ = 200
as = [1,]
nΩ = 3
name = "eraseme"
m, Ω, Ωs, gid_ref, ddm_ref, Π_ref, Πs = run(k, Nλ, as, nΩ, νs, name);
# Size of simple trace and multi trace
Nst = number_of_elements(m, skeleton(Ω), 0)
Nmt = sum([number_of_elements(m, boundary(ω), 0) for ω in Ωs])
# weights
ixpt = argmax(dof_weights(gid_ref))
# Indices of domain `iΩ` boundary
iΩ = 1
inds = inds_boundaries(nΩ, ddm_ref, Nst, Nmt)
f_xpt = i -> prod([inds[i][j] > 0 for j in 1:nΩ])
f_tra = i -> inds[i][iΩ] > 0 && (sum([inds[i][j] > 0 for j in 1:nΩ]) == 2)
f_ext = i -> inds[i][iΩ] > 0 && (sum([inds[i][j] > 0 for j in 1:nΩ]) == 1)
inds_xpt = [inds[i][iΩ] for i in 1:Nst if f_xpt(i)]
inds_tra = [inds[i][iΩ] for i in 1:Nst if f_tra(i)]
inds_ext = [inds[i][iΩ] for i in 1:Nst if f_ext(i)]

## Error: no real convergence for a random input
# input
xmt = randn(Nmt);
# output
err = Float64[]
ymt_ref = real.(Π_ref(xmt));
for Π in Πs
    ymt = real.(Π(xmt));
    push!(err, norm(ymt .- ymt_ref))
end
err

## Vizu (export Paraview)
# input
ip_source = inds_tra[3]
xmt = zeros(Nmt);
xmt[ip_source] = 1.;
xmts = [transpose(ld.MΣitoΣmt) * xmt for ld in ddm_ref.lds]
# output
ymt = real.(Π_ref(xmt));
ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
ymt = real.(Πs[5](xmt));
ymts = [transpose(ld.MΣitoΣmt) * ymt for ld in ddm_ref.lds]
# saving
vtk_save_on_skeleton(m, Ωs, xmts, "input", prefix*"exchange_input");
vtk_save_on_skeleton(m, Ωs, ymts, "output", prefix*"exchange_output");

## Error: convergence if input is a dirac, far enough from the dirac
# input
ip_source = inds_tra[3]
xmt = zeros(Nmt);
xmt[ip_source] = 1.;
xmts = [transpose(ld.MΣitoΣmt) * xmt for ld in ddm_ref.lds]
x_source = mt2vtx(gid_ref, m, Ω, Ωs, ddm_ref, ip_source)
# output (ref)
ymt_ref = real.(Π_ref(xmt));
# Computing error
diss = Float64[]
errs = Float64[]
dists = Float64[]
#for (ν, Π) in zip(νs[[1,5,9,13,17]], Πs[[1,5,9,13,17]])
for (ν, Π) in zip(νs, Πs)
    ymt = real.(Π(xmt));
    err = abs.(ymt .- ymt_ref)
    for ip in 1:Nmt
        x_ip = mt2vtx(gid_ref, m, Ω, Ωs, ddm_ref, ip)
        dist = norm(x_ip .- x_source)
        push!(diss, ν)
        push!(errs, err[ip])
        push!(dists, dist)
    end
end
p = draw_error_vs_dissipation_plot(dists, errs, diss);
pgfsave(prefix*"xpts-exchange_err_vs_dissipation_k$(k)_J$(nΩ)_Nl$(Nλ).pdf", p)
p = draw_error_vs_distance_plot(dists, errs, diss);
pgfsave(prefix*"xpts-exchange_err_vs_distance_k$(k)_J$(nΩ)_Nl$(Nλ).pdf", p)
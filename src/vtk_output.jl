using WriteVTK


function get_points_cells(m::Mesh,Ω::Domain; write_boundary=false)
	cells = MeshCell[]
    if dim(Ω) == 3
        for it in element_indices(m,Ω,3)
            push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, m.tet2vtx[:,it]))
        end
        if write_boundary
            for it in element_indices(m,Ω,2)
                push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, m.tri2vtx[:,it]))
            end
            for it in element_indices(m,boundary(Ω),2)
                push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, m.tri2vtx[:,it]))
            end
        end
    elseif dim(Ω) == 2
        for it in element_indices(m,Ω,2)
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, m.tri2vtx[:,it]))
        end
    end
    points = m.vtx
	return points, cells
end


function get_vtkfile(m::Mesh,Ω::Domain,name::String;write_boundary=false)
    return vtk_grid(name*".vtu", get_points_cells(m, Ω;
                                                  write_boundary=write_boundary)...)
end


function save_mesh(m::Mesh,Ω::Domain,name::String;write_boundary=false)
    vtkfile = get_vtkfile(m,Ω,name*".vtu",write_boundary=write_boundary)
	vtk_save(vtkfile)
end


"""Scalar fields"""
function set_data(vtkfile, m::Mesh, Ω::Domain,
                  ufields::Array{Tuple{String,Array{Complex{Float64},1}},1})
    # Extension matrix to full mesh points
    EΩ = transpose(restriction(m,Ω,0))
    for ufield in ufields
        # Vector extended to full domain
        u = EΩ*(ufield[2])
        # Mapping nodes (DOFs) to vtx
        u[m.nod2vtx[1,:]] = u
        # Writing data
        vtk_point_data(vtkfile,real.(u),ufield[1]*"_real")
        vtk_point_data(vtkfile,imag.(u),ufield[1]*"_imag")
        vtk_point_data(vtkfile,abs.(u),ufield[1]*"_abs")
    end
    return vtkfile
end


"""Vector fields"""
function set_data(vtkfile, m::Mesh, Ω::Domain,
                  ufields::Array{Tuple{String,Array{Complex{Float64},2}},1})
    # Extension matrix to full mesh points
    EΩ = transpose(restriction(m,Ω,0))
    for ufield in ufields
        # Vector extended to full domain
        ux = EΩ*(ufield[2][1,:])
        uy = EΩ*(ufield[2][2,:])
        uz = EΩ*(ufield[2][3,:])
        # Mapping nodes (DOFs) to vtx
        ux[m.nod2vtx[1,:]] = ux
        uy[m.nod2vtx[1,:]] = uy
        uz[m.nod2vtx[1,:]] = uz
        uxyz = vcat(ux',uy',uz')
        # Initialisation abs
        uabs = Array{Float64,1}(undef,size(uxyz,2))
        for iu in 1:size(uxyz,2)
            uabs[iu] = norm(uxyz[:,iu])
        end
        # Writing data
        vtk_point_data(vtkfile,real.(uxyz),ufield[1]*"_real")
        vtk_point_data(vtkfile,imag.(uxyz),ufield[1]*"_imag")
        vtk_point_data(vtkfile,uabs,ufield[1]*"_abs")
    end
    return vtkfile
end


function save_vector(m::Mesh, Ω::Domain, ufields, name::String)
    vtkfile = get_vtkfile(m, Ω, name)
    vtkfile = set_data(vtkfile, m, Ω, ufields)
	vtk_save(vtkfile)
end


function save_vector_partition(m::Mesh, Ωs::Vector{Domain}, ufieldss,
                               name::String)
    vtmfile = vtk_multiblock(name)
    local vtkfile
    for (Ω, ufields) in zip(Ωs, ufieldss)
        vtkfile = vtk_grid(vtmfile, get_points_cells(m, Ω)...)
        vtkfile = set_data(vtkfile, m, Ω, ufields)
    end
	vtk_save(vtmfile)
end


"""
Save colors of partition.
"""
function save_partition(m::Mesh, Ω::Domain, name::String)
    vtkfile = get_vtkfile(m,Ω,name)
    # Create colors vector
    colors = vcat([tag(ω)[2]*ones(number_of_elements(m,Domain(ω),dim(ω)))
                   for ω in Ω]...)
    vtk_cell_data(vtkfile, colors, "color")
	vtk_save(vtkfile)
end

function save_partition(m::Mesh, Ωs::Vector{Domain}, name::String)
    vtmfile = vtk_multiblock(name)
    local vtkfile
    for Ω in Ωs
        vtkfile = vtk_grid(vtmfile, get_points_cells(m, Ω)...)
        # Create colors vector
        colors = vcat([tag(ω)[2]*ones(number_of_elements(m,Domain(ω),dim(ω)))
                       for ω in Ω]...)
        vtk_cell_data(vtkfile, colors, "color")
    end
	vtk_save(vtmfile)
end


"""
Save medium coefficients (Helmholtz).
"""
function save_medium(m::Mesh, Ω::Domain, medium::AcousticMedium, name::String)
    vtkfile = get_vtkfile(m,Ω,name*"_"*medium.name)
    # Getting correct elements on which the loop is performed
    d = dim(Ω)
    if d == 2 elt2vtx = m.tri2vtx
    elseif d == 3 elt2vtx = m.tet2vtx
    end
    # Initialisation
    ρ = Array{Complex{Float64},1}(undef,size(elt2vtx,2))
    κ = Array{Complex{Float64},1}(undef,size(elt2vtx,2))
    # Loop on elements on which the integration is performed
    for ielt in 1:size(elt2vtx,2)
        s = m.vtx[:,elt2vtx[:,ielt]] # element vertices
        g = sum(s, dims=2) ./ size(s,2) # barycenter
        ρ[ielt] = medium.ρr(g)
        κ[ielt] = medium.κr(g)
    end
    vtk_cell_data(vtkfile, real.(ρ), "real rho")
    vtk_cell_data(vtkfile, imag.(ρ), "imag rho")
    vtk_cell_data(vtkfile, real.(κ), "real kappa")
    vtk_cell_data(vtkfile, imag.(κ), "imag kappa")
	vtk_save(vtkfile)
end


"""
Save medium coefficients (Maxwell).
"""
function save_medium(m::Mesh, Ω::Domain, medium::ElectromagneticMedium, name::String)
    vtkfile = get_vtkfile(m,Ω,name*"_"*medium.name)
    # Getting correct elements on which the loop is performed
    elt2vtx = m.tet2vtx
    # Initialisation
    μ = Array{Complex{Float64},1}(undef,size(elt2vtx,2))
    ϵ = Array{Complex{Float64},1}(undef,size(elt2vtx,2))
    # Loop on elements on which the integration is performed
    for ielt in 1:size(elt2vtx,2)
        s = m.vtx[:,elt2vtx[:,ielt]] # element vertices
        g = sum(s, dims=2) ./ size(s,2) # barycenter
        μ[ielt] = medium.μr(g)
        ϵ[ielt] = medium.ϵr(g)
    end
    vtk_cell_data(vtkfile, real.(μ), "real mu")
    vtk_cell_data(vtkfile, imag.(μ), "imag mu")
    vtk_cell_data(vtkfile, real.(ϵ), "real epsilon")
    vtk_cell_data(vtkfile, imag.(ϵ), "imag epsilon")
	vtk_save(vtkfile)
end


"""
Saving DDM solution on disk, and if applicable exact discrete solution and error.
Output is one (or three) file(s). The solution is averaged at interfaces (and
cross points) between sub-domains.
"""
function save_solutions(m, fullpb, solver, u, uexact, prefix, name)
    save_vector(m,fullpb.Ω,[("u",toP1(fullpb,m,fullpb.Ω,u))],prefix*"u_"*name)
    if !(solver.light_mode)
        save_vector(m,fullpb.Ω,[("u",toP1(fullpb,m,fullpb.Ω,uexact))],prefix*"uexact_"*name)
        save_vector(m,fullpb.Ω,[("e",toP1(fullpb,m,fullpb.Ω,uexact.-u))],prefix*"e_"*name)
    end
end

"""
Saving DDM solution on disk, and if applicable exact discrete solution and error.
Warning: output is one (or three) file(s) per sub-domain (with mesh points
duplicated in each file). The output contains jump at interfaces (and cross points).
"""
function save_solutions_partition(m, fullpb, pbs, ddm, solver, u, uexact, prefix, name)
    uis = [ld.Li.ui + ld.Fi for ld in ddm.lds]
    save_vector_partition(m, [pb.Ω for pb in pbs],
                          [[("u",toP1(pb,m,pb.Ω,ui))]
                           for (pb, ui) in zip(pbs, uis)],
                          prefix*"u_"*name)
    if !(solver.light_mode)
        uexactis = [transpose(ld.MΩitoΩ) * uexact for ld in ddm.lds]
        save_vector_partition(m, [pb.Ω for pb in pbs],
                            [[("u",toP1(pb,m,pb.Ω,ui))]
                            for (pb, ui) in zip(pbs, uis)],
                            prefix*"uexact_"*name)
        save_vector_partition(m, [pb.Ω for pb in pbs],
                            [[("e",toP1(pb,m,pb.Ω,uexacti.-ui))]
                            for (pb, ui, uexacti) in zip(pbs, uis, uexactis)],
                            prefix*"e_"*name)
    end
end

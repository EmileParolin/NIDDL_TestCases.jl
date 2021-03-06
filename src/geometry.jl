mutable struct LayersGeo <: Geometry
    d::Integer           # embbeding domain dimension
    shape::Symbol        # geometry of physical boundaries
    as::Vector{Real}     # vector of characteristic size of physical boundaries
    interior::Bool       # whether or not to mesh inside
    nΩ::Integer          # number of partition domains
    mode::Symbol         # type of partitionning (:cad, :geo, :metis)
    nl::Integer          # number of layers of cell for interior layering
    layer_from_PBC::Bool # whether or not layering from physical boundaries
end
function LayersGeo(;d=2, shape=:circle, as=[1,], interior=false, nΩ=2,
                   mode=:metis, nl=0, layer_from_PBC=true,)
    @assert mode in [:cad, :geo, :pie, :metis]
    LayersGeo(d, shape, as, interior, nΩ, mode, nl, layer_from_PBC)
end


"""
Return the object `m::Mesh`, the domain `Ω` of the problem and an array `Γs`
which corresponds to the list of physical boundaries which can be either
[`Γ_int`, `Γ_ext`] for a scattering problem or [`Γ_ext`,] for a transmission
problem.
"""
function get_mesh_and_domains(g::LayersGeo, h; name="")
    @assert h > 0
    # Constructing mesh via GMSH API
    info = construct_mesh(d=g.d, shape=g.shape, as=g.as, interior=g.interior,
                            h=h, nΩ=g.nΩ, mode=g.mode, name=name)
    vtx, eltdoms, elt2vtx, elttags, geobnd2vtx, geobnd_tags = info
    # Constructing Mesh
    m = Mesh(vtx, elt2vtx..., elttags)
    @info "Total number of vertices     $(number_of_elements(m,0))"
    @info "                edges        $(number_of_elements(m,1))"
    @info "                triangles    $(number_of_elements(m,2))"
    @info "                tetrahedrons $(number_of_elements(m,3))"
    # Constructing Domains
    Ω, Γs = construct_domains(g.d, m, eltdoms, geobnd2vtx, geobnd_tags,
                                length(g.as); interior=g.interior,
                                mode=g.mode)
    Ωs = [Domain(ω) for ω in Ω]
    # Interior layering
    if g.nl > 0
        info = interior_layering(m, Ω, Γs, g.d, g.nl, g.layer_from_PBC)
        vtx, eltdoms, elt2vtx, elttags, grps, new2old_bndtags = info
        # Re-Constructing Mesh
        # Re-Constructing Domains
        noddoms, edgdoms, tridoms, tetdoms = eltdoms
        @info "Creating domains"
        ωs = Vector{Vector{SingleDomain}}(undef,4)
        ωs[1] = detect_boundaries(m, noddoms, SingleDomain[], SingleDomain[])
        ωs[2] = detect_boundaries(m, edgdoms, noddoms, ωs[1])
        ωs[3] = detect_boundaries(m, tridoms, edgdoms, ωs[2])
        ωs[4] = detect_boundaries(m, tetdoms, tridoms, ωs[3])
        Ω = Domain(ωs[g.d+1])
        # Sanity checks
        msg = "Incompatibility between Domains and Mesh - dimension "
        @assert number_of_elements(m,Ω,3) == size(m.tet2vtx,2) msg*"3"
        @assert number_of_elements(m,Ω,2) == size(m.tri2vtx,2) msg*"2"
        @assert number_of_elements(m,Ω,1) == size(m.edg2vtx,2) msg*"1"
        @assert number_of_elements(m,Ω,0) == size(m.nod2vtx,2) msg*"0"
        # Grouping sub-domains
        Ωs = Domain[]
        ig = 1
        for grp in grps push!(Ωs, Domain(Ω[ig:ig+grp-1])); ig += grp end
        # Creating new boundary domains
        newΓs = Domain[]
        for γ in Γs
            newγ = Vector{SingleDomain}()
            for (tk, tv) in new2old_bndtags
                if (g.d-1, tv) in tags(γ)
                    for γi in ωs[g.d]
                        if (g.d-1, tk) == tag(γi)
                            push!(newγ, γi)
                        end
                    end
                end
            end
            push!(newΓs, Domain(newγ))
        end
        Γs = newΓs
    end
    return m, Ωs, Γs
end


function get_problems(g::LayersGeo, tc::TestCase, Ωs::Vector{Domain},
                      Γs::Vector{Domain}, tp::TransmissionParameters,
                      dd::OnionDDM)
    # Full domain
    Ω = union(Ωs...)
    Γ = union(Γs...)
    @assert sort(tags(Γ)) == sort(tags(boundary(Ω)))
    # Source (boundary conditions)
    bcs = tc(Γs)
    # Full problem
    fullpb = tc.pb_type(tc.medium,Ω,bcs)
    # Local problems
    pbs = Array{tc.pb_type,1}(undef,0)
    for ω in Ωs
        γ = boundary(ω)
        # Physical boundary condition
        pbc = tc([intersect(γ, Γi) for Γi in Γs])
        filter!(bc -> !isempty(bc.Γ), pbc)
        # Transmission boundary conditions
        γΣ = setdiff(γ, intersect(γ, Γ)) # γ \ (γ ∩ Γ)
        Σs = [intersect(γΣ, boundary(oω)) for oω in Ωs if oω != ω]
        filter!(σ -> !isempty(σ), Σs)
        filter!(σ -> dim(σ) == dim(γ), Σs)
        tbcs = [tp(Σ; geo=g, Ωs=Ωs, Γs=Γs, tc=tc) for Σ in Σs]
        # Local problem
        pb = tc.pb_type(tc.medium,ω,vcat(pbc, tbcs))
        push!(pbs, pb)
    end
    return fullpb, pbs
end
function get_problems(g::LayersGeo, tc::TestCase, Ωs::Vector{Domain},
                      Γs::Vector{Domain}, tp::TransmissionParameters,
                      dd::JunctionsDDM)
    # Layering sanity check
    @assert g.nl == 0 || g.layer_from_PBC
    # Full domain
    Ω = union(Ωs...)
    Γ = union(Γs...)
    @assert sort(tags(Γ)) == sort(tags(boundary(Ω)))
    # Source (boundary conditions)
    bcs = tc(Γs)
    # Full problem
    fullpb = tc.pb_type(tc.medium,Ω,bcs)
    # Local problems
    pbs = Array{tc.pb_type,1}(undef,0)
    for ω in Ωs
        γ = boundary(ω)
        # Physical boundary condition
        pbc = tc([intersect(γ, Γi) for Γi in Γs])
        filter!(bc -> !isempty(bc.Γ), pbc)
        # Transmission boundary conditions
        tbc = tp(γ; geo=g, Ωs=Ωs, Γs=Γs, tc=tc)
        # Local problem
        pb = tc.pb_type(tc.medium,ω,vcat(pbc, tbc))
        push!(pbs, pb)
    end
    return fullpb, pbs
end

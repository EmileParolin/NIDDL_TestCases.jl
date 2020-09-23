"""
Construct a mesh of dimension `d` using GMSH API.

The geometry consists of concentric layers, of characteristic lengths given by
`as`. If `interior` is true, the most interior volume is meshed. The typical
element mesh size is given by `h`. If `nΩ` if bigger than 1, a partition of
`nΩ` volumes is created. The mesh is saved to disk if a `name` is provided,
using the GMSH mesh format 4.1.

This method outputs the mesh information in the form of:
- `vtx`, the coordinates of the vertices;
- `elt2vtx` a 4-tuple defining mesh elements of dimension from 0 to 3;
- `elttag` a dictionnary with keys (dim, tag) specifying the location (start,
length) of geometrical entities characterized by the key in elt2vtx;
- `bndgeo_elts` surface elements corresponding to the geometrical surfaces;
- `bndgeo_tags` geometrical surfaces tags;
- `partition` specifies the relation between all geometrical entities present
in the mesh.
"""
function construct_mesh(; d=2, shape=:circle, as=[1,], interior=true, h=0.1,
                        nΩ=1, mode=:metis, name="", gmsh_info=10, to=missing)
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    @assert d in [2,3] "Incorrect dimension d = $(d), should be 2 or 3."
    d == 2 && @assert shape in [:circle, :square]
    d == 3 && @assert shape in [:sphere, :box, :cylinder, :cone]
    nPhysicalBC = interior ? 1 : 2
    if !(mode == :metis)
        @assert length(as)-nPhysicalBC == nΩ-1
    else
        @assert length(as) == nPhysicalBC
    end
    @info "Dimension               d  = $(d)"
    @info "Shape                        $(shape)"
    @info "Characteristic lengths  as = $(as)"
    @info "Interior volume is meshed    $(interior)"
    @info "Number of partitions    nΩ = $(nΩ)"
    @info "Typical mesh size       h  = $(h)"
    @info "GMSH Initialisation"
    # Boundaries that should belong to the CAD model
    as_cad = as
    if mode == :geo
        as_cad = interior ? [as[end],] : [as[1], as[end],]
    end
    # Start of GMSH stuff
    @timeit to "GMSH" begin
        gmsh.initialize()
        gmsh.model.add("Model")
        gmsh.option.setNumber("General.Terminal", gmsh_info)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        # Adding temporary domains to later define actual domain by difference
        @info "Creating geometry"
        shell_tags = Int64[]
        for ai in as_cad
            if d == 2
                if shape==:circle
                    push!(shell_tags, gmsh.model.occ.addDisk(0, 0, 0, ai, ai))
                elseif shape==:square
                    push!(shell_tags, gmsh.model.occ.addRectangle(-ai/2, -ai/2, 0, ai, ai))
                end
            elseif d == 3
                if shape==:sphere
                    push!(shell_tags, gmsh.model.occ.addSphere(0, 0, 0, ai))
                elseif shape==:box
                    push!(shell_tags, gmsh.model.occ.addBox(-ai/2, -ai/2, -ai/2, ai, ai, ai))
                elseif shape==:cylinder
                    push!(shell_tags, gmsh.model.occ.addCylinder(0, 0, -ai/2, 0, 0, ai, ai))
                elseif shape==:cone
                    push!(shell_tags, gmsh.model.occ.addCone(0, 0, -ai/2, 0, 0, ai, ai, 0))
                end
            end
        end
        # Construction of main domain by difference between the temporary domains
        # The first tag define a "volume" the following are holes in this volume
        layers = interior ? Vector{Int64}[Int64[shell_tags[1],]] : Vector{Int64}[]
        for ias in 1:length(as_cad)-1
            push!(layers, reverse(shell_tags[ias:ias+1]))
        end
        for layer in layers
            if d == 2
                tag = gmsh.model.occ.addPlaneSurface(layer)
            elseif d == 3
                tag = gmsh.model.occ.addVolume(layer)
            end
        end
        # Removing the temporary domains
        gmsh.model.occ.remove([(d,st) for st in shell_tags])
        # Meshing
        @info "Meshing"
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        # Boundary elements of geometrically constructed surfaces/boundaries
        geobnd2vtx, geobnd_tags = extract_elements(d-1)
        # METIS partitioning
        if mode == :metis
            @info "Mesh partitioning"
            gmsh.model.mesh.partition(nΩ)
        end
        # Extracting geometrical entities information
        tetdoms = [Ω for Ω in gmsh.model.getEntities(3)
                    if length(gmsh.model.mesh.getElements(Ω...)[3]) > 0]
        tridoms = [Ω for Ω in gmsh.model.getEntities(2)
                    if length(gmsh.model.mesh.getElements(Ω...)[3]) > 0]
        edgdoms = [Ω for Ω in gmsh.model.getEntities(1)
                    if length(gmsh.model.mesh.getElements(Ω...)[3]) > 0]
        noddoms = [Ω for Ω in gmsh.model.getEntities(0)
                    if length(gmsh.model.mesh.getElements(Ω...)[3]) > 0]
        # Extracting elements information
        @info "Extracting mesh data"
        tet2vtx, tettag = extract_elements(3)
        tri2vtx, tritag = extract_elements(2)
        edg2vtx, edgtag = extract_elements(1)
        nod2vtx, nodtag = extract_elements(0)
        # Extract nodes (and re-number according to tag)
        nodetags, coors, _ = gmsh.model.mesh.getNodes()
        @assert unique(nodetags) == nodetags
        vtx = reshape(coors,3,:)
        vtx[:,nodetags] = vtx # correct re-numbering
        # Writting mesh on disk
        if length(name) > 0
            @info "Mesh saved on disk as $(name).msh4"
            gmsh.write(name*".msh4")
        end
        # done...
        gmsh.finalize()
        @info "GMSH finalized"
    end
    # Renaming
    eltdoms = [noddoms, edgdoms, tridoms, tetdoms]
    elt2vtx = [nod2vtx, edg2vtx, tri2vtx, tet2vtx]
    elttags = [nodtag,  edgtag,  tritag,  tettag ]
    # Geometrical partitionning
    if mode == :geo || mode ==:pie
        @timeit to "Geometrical partitionning" begin
            if mode == :geo
                new_big2vtxs = geometrical_partitioning(vtx, elt2vtx[d+1], as, interior)
            elseif mode == :pie
                @assert interior && length(as) == 1
                new_big2vtxs = pie_partitioning(vtx, elt2vtx[d+1])
            end
            new_domains = create_new_domains(eltdoms[d+1], elt2vtx[d+1],
                                             eltdoms[d], elt2vtx[d], elttags[d],
                                             vtx, new_big2vtxs, d)
            # Unpacking
            eltdoms[d+1], elt2vtx[d+1], elttags[d+1] = new_domains[1]
            eltdoms[d],   elt2vtx[d],   elttags[d]   = new_domains[2]
        end
    end
    # Clean detection of junction points
    if d == 3
        edginfo = detect_junction_points(elt2vtx[3], elttags[3], 1)
        eltdoms[2], elt2vtx[2], elttags[2] = edginfo
    end
    nodinfo = detect_junction_points(elt2vtx[2], elttags[2], 0)
    eltdoms[1], elt2vtx[1], elttags[1] = nodinfo
    # Reformating
    elttags = merge(elttags...)
    return vtx, eltdoms, elt2vtx, elttags, geobnd2vtx, geobnd_tags
end


"""
Method to extract GMSH elements of dimension `d`.
"""
function extract_elements(d::Integer)
    # Initialisation
    elt2vtx = Array{Int64,2}(undef,d+1,0)
    elttag = Dict{Tuple{Int64,Tuple{Int64,Int64}},Tuple{Int64,Int64}}()
    # Loop on domains
    for Ω in gmsh.model.getEntities(d)
        elts = gmsh.model.mesh.getElements(Ω...)[3]
        # Treatment of entities that have not dissapeared due to partitioning
        if length(elts) > 0
            # Record size before adding elements
            previous_size = size(elt2vtx,2)
            # Extract, reshape and concatenate new elements to elt2vtx
            newshape = (d+1, Int(length(elts[1]) / (d+1)))
            elt2vtx = hcat(elt2vtx, Int64.(reshape(elts[1], newshape)))
            # Update elttag based on previously recorded size
            new_size = size(elt2vtx,2)
            elttag[(d,Ω)] = (previous_size+1, new_size-previous_size)
        end
    end
    return elt2vtx, elttag
end


"""
Perform a geometrical partitioning on a pre-existing mesh. The partition is
defined by concentric layers specified by `as`.
"""
function geometrical_partitioning(vtx, big2vtx, as::Vector{<:Number}, interior=true)
    as_nocad = interior ? as[1:end-1] : as[2:end-1]
    # Indices of mesh elements that will be split into domains
    ielts = collect(1:size(big2vtx,2))
    # To store domain elements
    domains = Vector{typeof(big2vtx)}(undef,length(as_nocad)+1)
    # Loop on concentric spheres
    for (ia,a) in enumerate(as_nocad)
        # To remove for future searches
        visited_ielts = typeof(ielts)(undef,0)
        # Initialisation of domain elements
        domains[ia] = typeof(big2vtx)(undef,size(big2vtx,1),0)
        for ie in ielts
            e = big2vtx[:,ie] # element
            # Geometric test
            if minimum(mapslices(norm, vtx[:,e], dims=1)) <= a
                domains[ia] = hcat(domains[ia], e)
                push!(visited_ielts, ie)
            end
        end
        setdiff!(ielts, visited_ielts)
    end
    # Remaining elements
    domains[end] = big2vtx[:,ielts]
    return domains
end


"""
Perform a geometrical partitioning on a pre-existing mesh. The partition is
defined by angular sectors starting from the node closest to the origin.
"""
function pie_partitioning(vtx, tri2vtx)
    # Looking for center
    iv0 = 0
    p0 = zeros(3) .+ Inf
    for iv in 1:size(vtx,2)
        piv = sum(vtx[:,iv],dims=2)
        if norm(piv) < norm(p0) iv0 = iv; p0 = piv end
    end
    # Looking for triangles containing center
    ntric = 0
    trics = Set{Int64}()
    for it in 1:size(tri2vtx,2)
        tri = tri2vtx[:,it]
        if iv0 in tri
            ntric += 1
            push!(trics, it)
        end
    end
    # Loop through triangles to define angular sectors
    sectors = [Vector{Int64}(undef, 0) for _ in 1:ntric]
    icandidates = Set(collect(1:size(tri2vtx,2)))
    for (ia, it) in enumerate(trics)
        tric = tri2vtx[:,it]
        # Edges
        edgs = getindex.(Ref(tric), [[1,2], [2,3], [3,1]])
        start_arm = p0 # Initialisation of edge leaving center
        end_arm = p0   # Initialisation of edge joining center
        for edg in edgs
            if iv0 == edg[1]
                start_arm = vtx[:,edg[2]]-p0
            elseif iv0 == edg[2]
                end_arm = vtx[:,edg[1]]-p0
            end
        end
        # Looping through triangles to define sectors
        for ic in icandidates
            tri = tri2vtx[:,ic]
            # Barycenter
            gb = sum(vtx[:,tri], dims=2)[:]
            # Is anti-clockwise to start-arm?
            bs = cross(start_arm, gb)[3] > 0
            # Is clockwise to end-arm?
            be = cross(end_arm, gb)[3] < 0
            # If in sector add it, and remove from candidates
            if bs && be
                push!(sectors[ia], ic)
                pop!(icandidates, ic)
            end
        end
    end
    domains = [tri2vtx[:,s] for s in sectors]
    return domains
end

"""
Detect boundary elements of a domain given only by its elements `big2vtx`. The
boundary elements `sml2vtx` are the boundary elements that are already known.
Two steps:
    1. Loop on big elements (triangles or tetrahedrons) to determine small
    elements (edges or triangles) which belong to the boundary
    2. Loop on small elements that belong to the boundary to determine their
    orientation with respect to the big elements
        a. For small elements that were already present, we need to preserve
        the orientation. Note that they may belong to different boundaries
        parts, so there may be different tags
        b. For small elements that are new, we need to compute a correct
        orientation
"""
function detect_boundary_elements(vtx, big2vtx, sml2vtx, smltag, d::Integer)
    @assert d in [2,3]
    @assert size(big2vtx,1)-1 == d
    @assert size(big2vtx,1)-1 == size(sml2vtx,1)
    # Dimension of big/small elements: nod=1, edg=2, tri=3, tet=4
    # Combinations of vertices
    if d == 2  # Edges in triangle
        comb = [[1,2], [2,3], [3,1]]
    elseif d == 3  # Triangles in tetrahedron
        comb = [[2,3,4], [1,4,3], [1,2,4], [1,3,2]]
    end
    # Dict using bnd element as keys
    seen = Dict{Array{Int64,1},Int64}()
    # Loop on elements
    for ib in 1:size(big2vtx,2)
        big = big2vtx[:,ib] # big element
        # Loop on small elements contained in big one
        for ind in comb
            sml = sort(big[ind]) # small element
            # testing if sml elt has already been seen
            if !(sml in keys(seen))
                seen[sml] = ib
            else
                seen[sml] = 0
            end
        end
    end
    # Elements that already belong to a boundary marked with tag of boundary domain
    for (k,v) in smltag
        for isml in v[1]:v[1]+v[2]-1
            sml = sort(sml2vtx[:,isml])
            # testing if sml elt has been seen
            if sml in keys(seen) && seen[sml] == 0
                error("Something went wrong! $(sml) seen twice and should belong to boundary")
            elseif sml in keys(seen) && seen[sml] > 0
                seen[sml] = -k[2][2]
            end
        end
    end
    # Initialisation new boundary elements
    Nbnd_new = sum(0 .< values(seen))
    sml2vtx_new = typeof(big2vtx)(undef,size(big2vtx,1)-1,Nbnd_new)
    # Initialisation old boundary elements
    tgs = unique([v for v in values(seen) if v < 0])
    sml2vtxs_old = Dict{Int64,typeof(big2vtx)}()
    for tg in unique(tgs)
        Nbnd_old = sum(values(seen) .== tg)
        sml2vtxs_old[tg] = typeof(big2vtx)(undef,size(big2vtx,1)-1,Nbnd_old)
    end
    # Current sml bnd element index
    is = Dict{Int64,Int64}([(1=>0),])
    for tg in tgs is[tg] = 0 end
    # Reformating
    for (sml, ibig) in seen
        if ibig < 0 # Negative tags are bnd elements that were already present
            # /!\ ibig is here the negative tag of the physical boundary
            is[ibig] += 1
            # Add element to array, but we need the unsorted one, so we look for it
            rg = smltag[(d-1, (d-1, -ibig))]
            sml_unsort = zero(sml)
            for isml in rg[1]:rg[1]+rg[2]-1
                if sml == sort(sml2vtx[:,isml])
                    sml_unsort = sml2vtx[:,isml]
                    break
                end
            end
            if sml_unsort == zero(sml) error("Element should have been found.") end
            sml2vtxs_old[ibig][:,is[ibig]] = sml_unsort
        elseif ibig > 0 # Positive tags are bnd elements that belong to new boundary
            is[1] += 1
            big = big2vtx[:,ibig]
            nde = setdiff(big,sml) # last/opposite node
            se = vtx[:,big] # vertices of element
            sf = vtx[:,sml] # vertices of face
            sn = vtx[:,nde] # vertex of last/opposite node
            # Computing orientation
            if d == 2
                # Normal vector (non-normalized)
                nrm = cross(se[:,2]-se[:,1], se[:,3]-se[:,1])
                # Tangent vector (non-normalized)
                tau = sf[:,2]-sf[:,1]
                # Sign
                sgn = sign(dot(sn-sf[:,1], cross(nrm, tau)))
            elseif d == 3
                # Normal vector (non-normalized)
                nrm = cross(sf[:,2]-sf[:,1], sf[:,3]-sf[:,1])
                # Sign
                sgn = sign(dot(sf[:,1]-sn, nrm))
            end
            # Re-orientation if necessary
            if sgn < 0 sml[1:2] = sml[[2,1]] end
            # Add element to array
            sml2vtx_new[:,is[1]] = sml
        end
    end
    return sml2vtxs_old, sml2vtx_new
end


"""
Detect junction points/lines.

`d` is either 0 to detect point, or 1 to detect lines.
"""
function detect_junction_points(big2vtx, bigtag, d)
    # For each small element, detect to which big element it belongs
    indoms = Dict{Vector{Int64}, Set{Int64}}()
    for (tk, tv) in bigtag
        for iv in tv[1]:tv[1]+tv[2]-1
            vtxs = big2vtx[:,iv]
            inds = d == 0 ? [[1,], [2,]] : [[1,2], [2,3], [3,1]]
            smls = sort.(getindex.(Ref(vtxs), inds))
            for sml in smls
                if sml in keys(indoms)
                    push!(indoms[sml], tk[2][2])
                else
                    indoms[sml] = Set(tk[2][2])
                end
            end
        end
    end
    # Extracting only when it belongs to more than one domain
    jctpts = Dict([kv for kv in indoms if length(kv[2]) > 1])
    # Creating output
    smldoms = Vector{Tuple{Int64,Int64}}(undef,0)
    sml2vtx = Matrix{Int64}(undef,d+1,0)
    smltag = Dict{Tuple{Int64,Tuple{Int64,Int64}},Tuple{Int64,Int64}}()
    iv = 0
    for (ijct, doms) in enumerate(unique(values(jctpts)))
        # New domain
        push!(smldoms, (d,ijct))
        # Appending small elements
        niv = 0
        for (sml, idoms) in jctpts
            if idoms == doms
                iv += 1
                niv += 1
                sml2vtx = hcat(sml2vtx, sml)
            end
        end
        # New tag
        smltag[(d,(d,ijct))] = (iv-niv+1, niv)
    end
    return smldoms, sml2vtx, smltag
end


"""
This routine creates all the information required to build the data structure
associated to domains.
It takes as input a pre-existing data structure and an array `new_big2vtxs`
which is a new partition of the mesh.
The output is the new data structure for this new partition.

The main challenge is to detect boundaries and deal with pre-existing
boundaries which we want to preserve.
"""
function create_new_domains(bigdoms, big2vtx, smldoms, sml2vtx, smltag,
                            vtx, new_big2vtxs, d)
    # Initialisation of big stuff
    new_bigdoms = typeof(bigdoms)(undef,0)
    new_big2vtx = typeof(big2vtx)(undef,size(big2vtx,1),0)
    new_bigtag = Dict{Tuple{Int64,Tuple{Int64,Int64}},Tuple{Int64,Int64}}()
    bigid = 1
    # Initialisation of small stuff
    new_smldoms = typeof(smldoms)(undef,0)
    new_sml2vtx = typeof(sml2vtx)(undef,size(sml2vtx,1),0)
    new_smltag = Dict{Tuple{Int64,Tuple{Int64,Int64}},Tuple{Int64,Int64}}()
    smlid = 1
    # Used to keep track of how existing boundaries are split in new ones
    new2old_bndtags = Dict{Int64, Int64}()
    new22old_bndtags = Dict{Int64, Int64}()
    # Loop on domains
    for new_big2vtx_i in new_big2vtxs
        old_sml2vtx_is, new_sml2vtx_i = detect_boundary_elements(vtx,
                                                                 new_big2vtx_i,
                                                                 sml2vtx,
                                                                 smltag, d)
        # 1. Taking care of big elements
        push!(new_bigdoms, (d,bigid))
        new_bigtag[(d,(d,bigid))] = (size(new_big2vtx,2)+1, size(new_big2vtx_i,2))
        new_big2vtx = hcat(new_big2vtx, new_big2vtx_i)
        bigid += 1
        # 2. Taking care of small elements
        # 2a. Newly created boundary of this domain
        push!(new_smldoms, (d-1,smlid))
        new_smltag[(d-1,(d-1,smlid))] = (size(new_sml2vtx,2)+1, size(new_sml2vtx_i,2))
        new_sml2vtx = hcat(new_sml2vtx, new_sml2vtx_i)
        smlid += 1
        # 2b. Old (pre-existing before new partitionning) boundaries of this domain
        for (old_sml2vtx_ik, old_sml2vtx_iv) in old_sml2vtx_is
            # Keep track of how old boundaries are split into new ones
            new2old_bndtags[smlid] = abs(old_sml2vtx_ik)
            # Add info
            push!(new_smldoms, (d-1,smlid))
            new_smltag[(d-1,(d-1,smlid))] = (size(new_sml2vtx,2)+1, size(old_sml2vtx_iv,2))
            new_sml2vtx = hcat(new_sml2vtx, old_sml2vtx_iv)
            smlid += 1
        end
    end
    # Post-treatment of small elements, because new boundaries are necessarly
    # created twice
    # Re-Initialisation of small stuff
    new2_smldoms = typeof(smldoms)(undef,0)
    new2_sml2vtx = typeof(sml2vtx)(undef,size(sml2vtx,1),0)
    new2_smltag = Dict{Tuple{Int64,Tuple{Int64,Int64}},Tuple{Int64,Int64}}()
    sml2id = 1
    for (tgk, tgv) in sort(new_smltag)
        # Dict using bnd element as keys
        seen = Dict{Array{Int64,1},Tuple{Int64,Int64}}()
        # Default marking is 0
        for is in tgv[1]:tgv[1]+tgv[2]-1
            seen[sort(new_sml2vtx[:,is])] = (0,is)
        end
        # Loop on other tags
        for (otgk, otgv) in sort(new_smltag)
            if tgk != otgk
                # If already present, mark as 1
                for is in otgv[1]:otgv[1]+otgv[2]-1
                    sml = sort(new_sml2vtx[:,is])
                    if sml in keys(seen)
                        seen[sml] = (1,is)
                    end
                end
                # Creation of intersection, if not done already and not empty
                ind = [v[2] for v in values(seen) if v[1] == 1]
                if tgk < otgk && length(ind) > 0
                    new2_sml2vtx_i = new_sml2vtx[:,ind]
                    push!(new2_smldoms, (d-1,sml2id))
                    new2_smltag[(d-1,(d-1,sml2id))] = (size(new2_sml2vtx,2)+1,
                                                       size(new2_sml2vtx_i,2))
                    new2_sml2vtx = hcat(new2_sml2vtx, new2_sml2vtx_i)
                    sml2id += 1
                end
                # Now that is created (or not), mark as -1
                for is in otgv[1]:otgv[1]+otgv[2]-1
                    sml = sort(new_sml2vtx[:,is])
                    if sml in keys(seen)
                        seen[sml] = (-1,is)
                    end
                end
            end
        end
        # Creation of remaining, if not empty
        ind = [v[2] for v in values(seen) if v[1] == 0]
        if length(ind) > 0
            # Keep track of how old boundaries are split into new ones
            new22old_bndtags[sml2id] = new2old_bndtags[tgk[2][2]]
            # Add info
            new2_sml2vtx_i = new_sml2vtx[:,ind]
            push!(new2_smldoms, (d-1,sml2id))
            new2_smltag[(d-1,(d-1,sml2id))] = (size(new2_sml2vtx,2)+1,
                                               size(new2_sml2vtx_i,2))
            new2_sml2vtx = hcat(new2_sml2vtx, new2_sml2vtx_i)
            sml2id += 1
        end
    end
    biginfo = new_bigdoms, new_big2vtx, new_bigtag
    smlinfo =  new2_smldoms, new2_sml2vtx, new2_smltag
    return biginfo, smlinfo, new22old_bndtags 
end


"""
Construction domains as defined by the package Kumquat.

Outputs Ωs, vector of domains, containing all the domains in the partition,
and Γs, vector of domains, containing all the physical boundaries.

The determination of the physical boundaries rests on the fact that each layer
is constructed using a fixed number of geometrical entities of dimension d-1,
which are tagged with increasing tags.
"""
function construct_domains(d::Integer, m::Mesh, eltdoms, geobnd2vtx,
                           geobnd_tags, nas; interior=true, mode=:metis, to=missing)
    if ismissing(to) to = TimerOutputs.get_defaulttimer() end
    noddoms, edgdoms, tridoms, tetdoms = eltdoms
    @info "Creating domains"
    @timeit to "Boundary detection" begin
        Ωs = Vector{Vector{SingleDomain}}(undef,4)
        @timeit to "d = 0" Ωs[1] = detect_boundaries(m, noddoms, SingleDomain[], SingleDomain[])
        @timeit to "d = 1" Ωs[2] = detect_boundaries(m, edgdoms, noddoms, Ωs[1])
        @timeit to "d = 2" Ωs[3] = detect_boundaries(m, tridoms, edgdoms, Ωs[2])
        @timeit to "d = 3" Ωs[4] = detect_boundaries(m, tetdoms, tridoms, Ωs[3])
        Ω = Domain(Ωs[d+1])
    end
    @info "Creating physical boundary domains"
    @timeit to "Physical boundary" begin
        Γs = Domain[]
        # Determination of number of geometrical entities in a boundary
        if mode == :geo
            nphys = interior ? 1 : 2
            nshell_tags = Int(length(geobnd_tags)/nphys)
        else
            nshell_tags = Int(length(geobnd_tags)/nas)
        end
        shell_tags = sort([t[2][2] for t in keys(geobnd_tags)])
        # Relying on sorting of geo tags to determine the boundaries
        Γext = [(d-1,(d-1,t)) for t in shell_tags[end-nshell_tags+1:end]]
        Γint = [(d-1,(d-1,t)) for t in shell_tags[1:nshell_tags]]
        # One or two physical boundaries
        Γs_tags = interior ? [Γext,] : [Γint, Γext]
        for Γtags in Γs_tags # Loop on physical boundaries
            Γ = Domain()
            for Γtag in Γtags # Loop on entities that make up the physical boundary
                # Loop on all entities of dim d-1
                for γ in skeleton(Ω)
                    # First element of γ
                    if dim(γ) == 1
                        bnd2vtx = m.edg2vtx
                    elseif dim(γ) == 2
                        bnd2vtx = m.tri2vtx
                    end
                    elt = bnd2vtx[:,tag2range(m,tag(γ),dim(γ))[1]]
                    # Loop on elements of physical boundary
                    istart, istop = geobnd_tags[Γtag]
                    for ie in istart:(istart+istop-1)
                        # If this element belongs to physical boundary
                        if norm(elt - geobnd2vtx[:,ie]) < 1.e-12
                            # Add γ as a subset of physical boundary
                            Γ = union(Domain(γ), Γ)
                            break
                        end
                    end
                end
            end
            push!(Γs, Γ)
        end
    end
    # Sanity checks
    msg = "Incompatibility between Domains and Mesh - dimension "
    @assert number_of_elements(m,Ω,3) == size(m.tet2vtx,2) msg*"3"
    @assert number_of_elements(m,Ω,2) == size(m.tri2vtx,2) msg*"2"
    @assert number_of_elements(m,Ω,1) == size(m.edg2vtx,2) msg*"1"
    @assert number_of_elements(m,Ω,0) == size(m.nod2vtx,2) msg*"0"
    return Ω, Γs
end


function detect_boundaries(m::Mesh, bigdoms, smldoms, Γs)
    Ωs = SingleDomain[]
    for Ωtag in bigdoms
        Ωoutwardbnd = SingleDomain[]
        Ωinwardbnd = SingleDomain[]
        Ωembeddedbnd = SingleDomain[]
        # Loop on small domains
        for Γtag in smldoms
            # Already created boundary domain
            Γ = Γs[[tag(γ)==Γtag for γ in Γs]][1]
            # Relation between Γ and Ω
            isbnd, sgn = detect_relation(m, Γtag, Ωtag)
            if isbnd
                if sgn > 0
                    push!(Ωoutwardbnd, Γ)
                elseif sgn < 0
                    push!(Ωinwardbnd, Γ)
                else
                    push!(Ωembeddedbnd, Γ)
                end
            end
        end
        # Creating big domain
        Ω = SingleDomain(Ωtag, Ωoutwardbnd, Ωinwardbnd, Ωembeddedbnd)
        push!(Ωs, Ω)
    end
    return Ωs
end


"""
Detect `nl` layers of elements inside the domains of `Ω` starting from its
boundary.  Boundary parts starting from the domains `Γs` are ignored.
"""
function element_layer(m::Mesh, Ω::Domain, Γs::Vector{Domain}, d::Integer,
                       nl::Integer)
    @assert d in [2,3]
    # Mesh elements that will be split into domains
    candidates = d == 2 ? m.tri2vtx : m.tet2vtx
    candidates2boundary = d == 2 ? m.tri2edg : m.tet2tri
    # To store domain elements
    domains = Vector{typeof(candidates)}(undef,0)
    domain_groups = Int64[]
    # Loop on pre-existing domains
    for sω in Ω
        ω = Domain(sω)
        # Admissible boundary
        γ = setdiff(boundary(ω), intersect(boundary(ω), union(Γs...)))
        # Set of sub-domains, themselves given by set of indices
        icandidates = Set{Set{Int64}}()
        push!(icandidates, Set(element_indices(m,ω,d)))
        # Loop on boundary parts
        for sγi in γ
            # Loop on existing sub-domains
            for icandidate in icandidates
                pop!(icandidates, icandidate)
                icandidate_layer = Set{Int64}()
                # Elements from which the domain is extruded
                γielts = element_indices(m,Domain(sγi),d-1)
                # Add nl layers
                for inl in 1:nl
                    # To store future boundary
                    future_γielts = typeof(γielts)(undef,0)
                    # Loop on elements in sub-domain
                    for ie in icandidate
                        # do we touch the boundary?
                        if any(in.(candidates2boundary[:,ie], Ref(γielts)))
                            pop!(icandidate, ie)
                            push!(icandidate_layer, ie)
                            push!(future_γielts, candidates2boundary[:,ie]...)
                        end
                    end
                    # New boundary
                    γielts = setdiff(future_γielts, γielts)
                end
                # Add two new sub-domains
                isempty(icandidate_layer) || push!(icandidates, icandidate_layer)
                isempty(icandidate      ) || push!(icandidates, icandidate      )
            end
        end
        # Creating actual domains
        for icandidate in icandidates
            push!(domains, candidates[:, sort(collect(icandidate))])
        end
        push!(domain_groups, length(icandidates))
    end
    return domains, domain_groups
end


"""
Create new domains by splitting each domain into an interior and a boundary
strip consisting of `nl` layers of elements inside the domains of `Ω` starting
from its boundary.  Boundary parts starting from the domains `Γs` are ignored
if the boolean `layer_from_PBC` is true.
"""
function interior_layering(m::Mesh, Ω::Domain, Γs::Vector{Domain}, d::Integer,
                           nl::Integer, layer_from_PBC::Bool)
    # Renaming
    eltdoms = [[k[2] for (k,v) in m.elttags if k[1] == id && k[2][1] == id]
               for id in 0:3]
    elt2vtx = [m.nod2vtx, m.edg2vtx, m.tri2vtx, m.tet2vtx]
    elttags = [Dict([(k,v) for (k,v) in m.elttags if k[1] == id && k[2][1] == id])
               for id in 0:3]
    # Creating layers and new domains
    new_big2vtxs, grps = element_layer(m, Ω, layer_from_PBC ? Domain[] : Γs, d, nl)
    new_domains = create_new_domains(eltdoms[d+1], elt2vtx[d+1],
                                     eltdoms[d], elt2vtx[d], elttags[d],
                                     m.vtx, new_big2vtxs, d)
    # Unpacking
    eltdoms[d+1], elt2vtx[d+1], elttags[d+1] = new_domains[1]
    eltdoms[d],   elt2vtx[d],   elttags[d]   = new_domains[2]
    new2old_bndtags = new_domains[3]
    # Removing interior (very small) elements creating during meshing
    for id in 1:d-1
        Nnew = 0
        for (_,v) in sort(elttags[id]) Nnew += v[2] end
        new_elt2vtx = typeof(elt2vtx[id])(undef, size(elt2vtx[id], 1), Nnew)
        for (k,v) in sort(elttags[id])
            @assert k[1] == k[2][1]
            ind = v[1]:v[1]+v[2]-1
            new_elt2vtx[:,ind] = elt2vtx[id][:,ind]
        end
        elt2vtx[id] = new_elt2vtx
    end
    # Clean detection of junction points
    if d == 3
        edginfo = detect_junction_points(elt2vtx[3], elttags[3], 1)
        eltdoms[2], elt2vtx[2], elttags[2] = edginfo
    end
    nodinfo = detect_junction_points(elt2vtx[2], elttags[2], 0)
    eltdoms[1], elt2vtx[1], elttags[1] = nodinfo
    # Reformating
    elttags = merge(elttags...)
    return m.vtx, eltdoms, elt2vtx, elttags, grps, new2old_bndtags
end

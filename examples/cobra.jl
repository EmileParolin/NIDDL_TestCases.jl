using Pkg
Pkg.activate("./")
using NIDDL


function build_cobra_mesh(;h=0.01,nΩ=1,name="", gmsh_info=10)
    gmsh.initialize()
    gmsh.model.add("Model")
    gmsh.option.setNumber("General.Terminal", gmsh_info)
    # Open cad file
    gmsh.open("/Users/emile/phd/Cobra/cobra.geo")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    # Meshing
    @info "Meshing"
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    # Boundary elements of geometrically constructed surfaces/boundaries
    geobnd2vtx, geobnd_tags = extract_elements(2)
    # METIS partitioning
    @info "Mesh partitioning"
    gmsh.model.mesh.partition(nΩ)
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
    # Renaming
    eltdoms = [noddoms, edgdoms, tridoms, tetdoms]
    elt2vtx = [nod2vtx, edg2vtx, tri2vtx, tet2vtx]
    elttags = [nodtag,  edgtag,  tritag,  tettag ]
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

build_cobra_mesh(;h=0.008,nΩ=16,name="/Users/emile/phd/Cobra/cobra");

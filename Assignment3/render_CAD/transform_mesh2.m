function mesh = transform_mesh2(mesh)

if isfield(mesh, 'transform')
   transform = mesh.transform;
   if ~iscell(transform)
       t{1} = transform;
       transform = t;
   end;
   for i = 1 : length(transform)
      mesh.vertices = transform{i}.sc*mesh.vertices*transform{i}.R + repmat(transform{i}.t, [size(mesh.vertices, 1), 1]); 
      if isfield(mesh, 'bbox')
         mesh.bbox.vertices = transform{i}.sc*mesh.bbox.vertices*transform{i}.R + repmat(transform{i}.t, [size(mesh.bbox.vertices, 1), 1]);
      end;
   end;
   if isfield(mesh, 'm') & numel(mesh.m)
       mesh.m = transform_m(mesh.m, transform);
   end;
end;
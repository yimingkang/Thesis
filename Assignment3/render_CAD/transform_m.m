function m = transform_m(m, transform)

for i = 1 : length(m.layers)
    if size(m.layers(i).vertices, 1) == 3
        if (0)
            mesh.vertices = m.layers(i).vertices';
            mesh.faces = double(m.layers(i).facetidx');
            try
            mesh = refine_mesh(mesh);
            m.layers(i).vertices = mesh.vertices';
            m.layers(i).facetidx = mesh.faces';
            catch
                aa=1;
            end;
        end;
        for j = 1 : length(transform)
            m.layers(i).vertices = (transform{j}.sc*m.layers(i).vertices'*transform{j}.R + repmat(transform{j}.t, [size(m.layers(i).vertices, 2), 1]))';
        end;
    end;
end;
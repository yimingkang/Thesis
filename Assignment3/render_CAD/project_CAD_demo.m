function project_CAD_demo(imname, cad_id, store)

if nargin < 1
    imname = 'um_000038';
end;
if nargin < 2
    cad_id = 1;
end;
if nargin < 3
    store = 0;
end;

im = imread([imname '.png']);
calib_file = [imname '.txt'];

cls = 'car';
if cad_id < 0
    cls = 'bed';
    cad_id = abs(cad_id);
end;
MODEL_DIR = pwd;

data = load(fullfile(MODEL_DIR, sprintf('%s_%03d_mesh.mat', cls, cad_id)));
[~,~,calib] = loadCalibration(calib_file);
caddata.mesh = data.mesh;
caddata.mesh = adddims(caddata.mesh, cls);

[K, R, t] = art(calib.P_rect{3});
%x=size(im,2)*0.5*ones(3,1);
%y = [size(im,1),size(im,1)*0.6]';
%y = [y; y(end)-5];
imshow(im);
[x,y] = getpts();
% the trajectory is not very good... probably Bezier would make it better
[x,y] = generate_traj(x,y);

%-------------------------------------
% you should return:
% p3d ... a nx3 matrix of points on the ground plane computed from 2D points
% (x,y), order matters
% ng ... a 3x1 normal to the ground plane
[p3d, ng] = YOURFUNCTION(x,y,??)  % project points to ground
%-------------------------------------


[p3d, d] = smooth_traj(p3d); % sample and smooth the trajectory

M = [];
if store
   outdir = fullfile(pwd, 'frames');
   if ~exist(outdir, 'dir')
       mkdir(outdir);
   end
end;
close all;
figure('position', [5,5,size(im,2), size(im, 1)]);
subplot('position',[0,0,1,1]);
imshow(im);
hold on;
h=[];
for i = 1 : size(d, 1)
   p = p3d(i, :)';
   dir3D = d(i, :)';
   vertices = generate_box3D(p, dir3D, ng, caddata.mesh.dims/1000);
   [mesh, ind] = findtransform(caddata, vertices);
   v = getfacedist(mesh);
   mesh.faces = mesh.faces(v, :);
   %close all;
   vertices = project_points(mesh.vertices, K);
   if ~isempty(h)
       delete(h);
   end;
   h=renderimage([], mesh.faces, vertices, mesh, 1);
   if store
   M{i} = getframe;
   else
       pause(0.05);
   end;
   %aviobj = addframe(aviobj,M);
end;
if store
    % extract frames, do:
    % ./ffmpeg -framerate 1/0.15 -i frames/%04d.png -c:v libx264 -r 30  CAD.mp4
    % to do a video
for i = 1 : length(M)
   imwrite(M{i}.cdata, fullfile(outdir, sprintf('%04d.png', i-1)));
end;
end;

end


function [P, gr_plane] = generate_traj3D(x, y, K)
   P = zeros(length(x), 3);
   for i = 1 : length(x)
       [~,~,p, gr_plane] = ground_dist_point_to_cam(K, [x(i),y(i)]);
       P(i, :) = p';
   end;
end

function vertices = project_points(vertices, K)
   vertices = (K * vertices')';
   vertices = vertices ./ repmat(vertices(:, 3), [1,3]);
end


function [x_out,y_out] = generate_traj(x,y)

n = 2;
x_out = [];
y_out = [];
for i = 1 : length(x)-1
    x_i = x(i) + [0:1/n:1-1/n]'*(x(i+1)-x(i));
    y_i = y(i) + [0:1/n:1-1/n]'*(y(i+1)-y(i));
    x_out = [x_out; x_i];
    y_out = [y_out; y_i];
end;

end

function [p_out, d] = smooth_traj(p)
% do a simple thing. longer trajectories will result in faster driving
% might have some problems in the clicked points...
n = 6;
p_out = [];
d = [];
for i = 1 : size(p, 1)-1
    p_i = repmat(p(i,:), [n, 1]) + [0:1/n:1-1/n]'*(p(i+1, :)-p(i, :));
    p_out = [p_out; p_i];
end;
% smooth the trajectory a bit
g = fspecial('gaussian', [5,1], 1);
p_smooth = [];
for j = [1,3]
    p_smooth(:, j) = conv(p_out(:, j), g, 'valid');
end;
p_smooth(:, 2) = p_out(1, 2); % floor always the same
d = diff(p_smooth);

end


function vertices = generate_box3D(p, dir3D, ng, dims)

dir3D = dir3D/norm(dir3D);
ng = ng/norm(ng);
nf = cross(dir3D, ng);
nf = nf/norm(nf);
vertices = zeros(8, 3);
vertices(6, :) = (p - nf*dims(1)/2)';
vertices(5, :) = (p + nf*dims(1)/2)';
vertices(7, :) = (vertices(6,:)' + ng*dims(2))';
vertices(8, :) = (vertices(5,:)' + ng*dims(2))';
vertices(1, :) = (vertices(5,:)' + dir3D*dims(3))';
vertices(2, :) = (vertices(6,:)' + dir3D*dims(3))';
vertices(3, :) = (vertices(2,:)' + ng*dims(2))';
vertices(4, :) = (vertices(1,:)' + ng*dims(2))';

end

function [vertices, vert_ind] = getboxpoints(vertices, faces)

faces_annot = faces_for_box();
vert_ind = zeros(size(vertices, 1), 1);

for i = 1 : size(vertices, 1)
    [y,x] = find(faces_annot == i);
    y = unique(y);
    
    vert = [1:size(vertices, 1)]';
    for j = 1 : length(y)
        vert = intersect(vert, faces(y(j), :));
    end;
    vert_ind(i) = vert;
end;

vertices = vertices(vert_ind, :);
end

function v = getfacedist(mesh)

vertices = mesh.vertices;
faces = mesh.faces;
d = sqrt(vertices(:, 1).^2 + vertices(:, 2).^2 + vertices(:, 3).^2);
ind = find(faces);
dfaces = zeros(size(faces));
dfaces(ind) = d(faces(ind));
dfaces = max(dfaces, [], 2);
%dfaces = mean(dfaces, 2);
[u, v] = sort(dfaces, 'descend');

end



function [mesh, ind] = findtransform(caddata, vertices)

    vertices_cad = caddata.mesh.bbox.vertices;
    [pointsCAD, ind] = getboxpoints(vertices_cad, caddata.mesh.bbox.faces);
    
    pointsdata = vertices;
    [d, p, transf] = procrustes(pointsdata,pointsCAD);
    transform = struct('t', transf.c(1, :), 'R', transf.T, 'sc', transf.b);
    %transform.t = point3D;
    %transform.sc = 1;
    %transform.R = pinv(pointsCAD(2:end,:)') * pointsdata(2:end,:)';
    mesh = caddata.mesh;
    mesh.transform{1} = transform;
    mesh = transform_mesh2(mesh);
    
end


function h=renderimage(im, faces, vertices, meshtransf, rendertexture)

if ~isempty(im)
figure('position', [5,5,size(im,2), size(im, 1)]);
subplot('position',[0,0,1,1]);
imshow(im);
hold on;
end;
if ~rendertexture
   h=trisurf(faces,vertices(:,1),vertices(:,2),ones(size(vertices, 1), 1),'EdgeColor','none','FaceColor',[1,1,1]);
else
   h=trisurf(faces,vertices(:,1),vertices(:,2),ones(size(vertices, 1), 1),'facevertexcdata',meshtransf.colors,'EdgeColor','none');
end;
end


function mesh = adddims(mesh, cls)
   if isfield(mesh, 'dims')
       return;
   else
       [dx, dy, dz, ~] = getboxdimensions(mesh.bbox.vertices);
       if strcmp(cls, 'car')
           dims = [dx*4.5/dz, dy*4.5/dx, 4.5];
       else
           dims = [dx*2/dz, dy*2/dx, 2];
       end;
       mesh.dims = dims*1000;
   end;
end
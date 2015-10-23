function [dx, dy, dz, aspect] = getboxdimensions(vertices)

if size(vertices, 1) < size(vertices, 2), vertices = vertices'; end;
dx = norm(vertices(2, :) - vertices(1, :));
dy = norm(vertices(4, :) - vertices(1, :));
dz = norm(vertices(5, :) - vertices(1, :));

aspect = [dy / dx, dz / dx];
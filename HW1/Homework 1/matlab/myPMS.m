function [N, rho] = myPMS(data, m)
% myPMS  Perform standard photometric stereo
% INPUT:
%   data.s        : nimages x 3 light source directions
%   data.L        : nimages x 3 light source intensities
%   data.imgs{i}  : H x W (or H x W x 3) image under light i
%   data.mask     : H x W mask
%   m             : linear indices of valid pixels (mask == 1)
%
% OUTPUT:
%   N : H x W x 3 surface normal map, values in [-1, 1]

    %% Extract information
    directions = data.s;            % n x 3 light directions
    imgs = data.imgs;
    nImages = size(directions, 1);

    [H, W] = size(data.mask);
    p = length(m);

    I = zeros(nImages, p, 3);
    for i = 1 : nImages
        img = double(imgs{i});
        img = reshape(img, [], 3);
        I(i, :, :) = img(m, :);
    end

    %% Solve for normals and albedo using least squares
    % For each pixel j:  I(:, j) = directions * g(:, j)
    % g = rho * N  (albedo * normal)
    gR = directions \ I(:, :, 1);
    gG = directions \ I(:, :, 2);
    gB = directions \ I(:, :, 3);
    G = (gR + gG + gB) / 3;

    %% Normalize to get rho and normals
    N_pixels = G ./ vecnorm(G, 2, 1);
    rho_pixels = cat(1, vecnorm(gR, 2, 1), vecnorm(gG, 2, 1), vecnorm(gB, 2, 1));

    %% Reshape into H × W × 3 normal map
    rho = zeros(H, W, 3);
    rho_reshaped = zeros(H * W, 3);
    rho_reshaped(m, :) = rho_pixels';
    rho(:, :, 1) = reshape(rho_reshaped(:, 1), H, W);
    rho(:, :, 2) = reshape(rho_reshaped(:, 2), H, W);
    rho(:, :, 3) = reshape(rho_reshaped(:, 3), H, W);

    N = zeros(H, W, 3);
    N_reshaped = zeros(H * W, 3);
    N_reshaped(m, :) = N_pixels';
    N(:, :, 1) = reshape(N_reshaped(:, 1), H, W);
    N(:, :, 2) = reshape(N_reshaped(:, 2), H, W);
    N(:, :, 3) = reshape(N_reshaped(:, 3), H, W);
end

function N = myPMS(data, m)
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
    intensities = data.L;           % n x 3 light directions
    nImages = size(directions, 1);

    [H, W] = size(data.mask);
    p = length(m);

    I = zeros(nImages, p, 3);
    for i = 1 : nImages
        img = double(data.imgs{i});
        img = reshape(img, [], 3);

        I(i, :, :) = img(m, :) ./ intensities(i, :);
    end

    %% Solve for normals and albedo using least squares
    % For each pixel j:  I(:, j) = directions * g(:, j)
    % g = rho * N  (albedo * normal)
    gR = directions \ I(:, :, 1);
    gG = directions \ I(:, :, 2);
    gB = directions \ I(:, :, 3);
    G = (gR + gG + gB) / 3;

    %% Normalize to get only normals
    N_pixels = G ./ vecnorm(G, 2, 1);

    %% Reshape into H × W × 3 normal map
    N = zeros(H, W, 3);
    N_reshaped = zeros(H * W, 3);
    N_reshaped(m, :) = N_pixels';
    N(:, :, 1) = reshape(N_reshaped(:, 1), H, W);
    N(:, :, 2) = reshape(N_reshaped(:, 2), H, W);
    N(:, :, 3) = reshape(N_reshaped(:, 3), H, W);
end

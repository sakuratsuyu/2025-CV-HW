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
    intensities = data.L;           % n x 3 light directions
    imgs = data.imgs;
    nImages = size(directions, 1);

    [H, W] = size(data.mask);
    p = length(m);

    dark_ratio = 0.2;
    bright_ratio = 0.0;

    I = zeros(nImages, p, 3);
    for i = 1 : nImages
        img = double(imgs{i});
        img = reshape(img, [], 3);

        I(i, :, :) = img(m, :) ./ intensities(i, :);
    end

    I_magnitude = sqrt(squeeze(sum(I .^ 2, 3)));  % n x p

    [~, sorted_idx] = sort(I_magnitude, 1, 'ascend');

    low_cut = ceil(dark_ratio * nImages) + 1;
    high_cut = floor((1 - bright_ratio) * nImages);
    kcount = high_cut - low_cut + 1;
    W_mask = false(nImages, p);
    rowsToFill = low_cut:high_cut; % length = kcount
    for r = 1:kcount
        idx_rows = sorted_idx(rowsToFill(r), :); % 1 × p, image indices to keep for each pixel
        % linear indexing trick: set W(idx_rows(j), j) = true for all j
        % We'll compute linear indices:
        linIdx = idx_rows + (0:(p-1))*nImages; % 1×p vector of linear indices into W(:)
        W_mask(linIdx) = true;
    end

    gR = zeros(3, p);
    gG = zeros(3, p);
    gB = zeros(3, p);
    for j = 1 : p
        mask = W_mask(:,j) ~= 0;
        gR(:,j) = directions(mask, :) \ I(mask, j, 1);
        gG(:,j) = directions(mask, :) \ I(mask, j, 2);
        gB(:,j) = directions(mask, :) \ I(mask, j, 3);
    end
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

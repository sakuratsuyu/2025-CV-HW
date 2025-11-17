function [N, rho] = myPMS_robust(data, m)
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

    N = zeros(H, W, 3);
    rho = zeros(H, W, 3);
    for k = 1 : p
        idx = m(k);

        [y, x] = ind2sub([H, W], idx);

        % Collect observations for this pixel
        I = zeros(nImages, 3);
        for i = 1 : nImages
            pixel = double(imgs{i}(y, x, :));
            pixel = reshape(pixel, [1, 3]);
            I(i, :) = pixel ./ intensities(i, :);
        end

        % Compute intensity magnitude to do trimming
        I_magnitude = sqrt(sum(I .^ 2, 2));  % per-image magnitude

        % Sort intensities
        [~, order] = sort(I_magnitude);

        % Compute trimming ranges
        low_cut = ceil(dark_ratio * nImages) + 1;
        high_cut = floor((1 - bright_ratio) * nImages);
        keep_idx = order(low_cut : high_cut);  % indices to keep

        I_clean = I(keep_idx, :);
        directions_clean = directions(keep_idx, :);

        gR = directions_clean \ I_clean(:, 1);
        gG = directions_clean \ I_clean(:, 2);
        gB = directions_clean \ I_clean(:, 3);
        G = (gR + gG + gB) / 3;
        n = G ./ vecnorm(G, 2, 1);
        reflect = cat(1, vecnorm(gR, 2, 1), vecnorm(gG, 2, 1), vecnorm(gB, 2, 1));

        % Save normal
        rho(y, x, :) = reflect;
        N(y, x, :) = n;
    end
end

function N = myPMS(data, m)
% myPMS  Perform standard photometric stereo
% INPUT:
%   data.s        : nLights x 3 light direction matrix
%   data.imgs{i}  : H x W (or H x W x 3) image under light i
%   data.mask     : H x W mask
%   m             : linear indices of valid pixels (mask == 1)
%
% OUTPUT:
%   N : H x W x 3 surface normal map, values in [-1, 1]

    %% Extract information
    L = data.s;            % n × 3 light directions
    nLights = size(L, 1);

    % Image resolution
    [H, W] = size(data.mask);

    %% Convert images into a big matrix I: nLights × p
    % p = number of valid pixels
    p = length(m);
    I = zeros(nLights, p);

    for i = 1 : nLights
        img = data.imgs{i};

        % If RGB -> convert to grayscale (photometric stereo requires 1 ch)
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % Convert to double
        img = double(img);

        % Extract pixels in mask using indices m
        I(i, :) = img(m);
    end

    %% Solve for normals and albedo using least squares
    % For each pixel j:  I(:, j) = L * g(:, j)
    % g = rho * N   (albedo * normal)
    G = L \ I;      % Solve L * G = I, result is 3 × p

    %% Normalize to get only normals
    N_pixels = bsxfun(@rdivide, G, sqrt(sum(G.^2, 1)) + eps);  % 3 × p

    %% Reshape into H × W × 3 normal map
    N = zeros(H, W, 3);
    N1 = N_pixels';  % Convert to p × 3

    % Fill into normal map
    N_reshaped = zeros(H * W, 3);
    N_reshaped(m, :) = N1;
    N(:, :, 1) = reshape(N_reshaped(:, 1), H, W);
    N(:, :, 2) = reshape(N_reshaped(:, 2), H, W);
    N(:, :, 3) = reshape(N_reshaped(:, 3), H, W);
end

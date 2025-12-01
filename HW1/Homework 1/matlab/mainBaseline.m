clc;
close all;
clear all;

dataFormat = 'PNG'; 

%==========01=========%
dataNameStack{1} = 'bear';
%==========02=========%
dataNameStack{2} = 'cat';
%==========03=========%
dataNameStack{3} = 'pot';
%==========04=========%
dataNameStack{4} = 'buddha';

for testId = 1 : 4
    dataName = [dataNameStack{testId}, dataFormat];
    datadir = ['..\pmsData\', dataName];
    bitDepth = 16;
    gamma = 1;
    resize = 1;  
    data = load_datadir_re(datadir, bitDepth, resize, gamma); 

    L = data.s;
    f = size(L, 1);
    [height, width, color] = size(data.mask);
    if color == 1
        mask1 = double(data.mask./255);
    else
        mask1 = double(rgb2gray(data.mask)./255);
    end
    mask3 = repmat(mask1, [1, 1, 3]);
    m = find(mask1 == 1);
    p = length(m);

    %% Standard photometric stereo
    [normal, rho] = myPMS_robust_accelerate(data, m);
    
    t = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
    fprintf('[%s] Iteration %d completed\n', string(t), testId);

    %% Save results "png"
    % imwrite(uint16((normal + 1) * (2 ^ (bitDepth - 1))) .* uint16(mask3), strcat(dataName, '_normal.png'));
    % imwrite(uint16(rho * (2 ^ bitDepth - 1)) .* uint16(mask3), strcat(dataName, '_rho.png'));

    nImages = size(data.s, 1);
    psnr_avg = 0.0;
    ssim_avg = 0.0;
    for i = 1 : nImages
        direction = data.s(i, :);
        intensity = data.L(i, :);
        direction = reshape(direction, [1 size(direction)]);
        intensity = reshape(intensity, [1 size(intensity)]);

        reflectance = rho .* sum(normal .* direction, 3);
        reflectance = reshape(reflectance, [size(reflectance) 1]);
        rendered = intensity .* reflectance;
        color = uint16(rendered * (2 ^ bitDepth - 1)) .* uint16(mask3);

        % idx = sprintf('%03d', i);
        % imwrite(color, strcat(dataName, idx, '_color.png'));
    
        % diff = abs(double(rendered) - data.imgs{1});
        % max_diff = max(diff, [], 'all');
        % fprintf('max diff: %.6f\n', max_diff)

        psnr_value = psnr(double(color), data.imgs{i}, 2 ^ bitDepth - 1);
        psnr_avg = psnr_avg + psnr_value;
        % fprintf('PSNR: %.6f\n', psnr_value);

        ssim_value = ssim(double(color), data.imgs{i});
        ssim_avg = ssim_avg + ssim_value;
    end

    fprintf('%s avg PSNR: %.6f\n', dataName, psnr_avg / nImages);
    fprintf('%s avg SSIM: %.6f\n', dataName, ssim_avg / nImages);

    %% Save results "mat"
    % save(strcat(dataName, '_Normal.mat'), 'Normal');
end
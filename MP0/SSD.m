clear all
clc
figure(1)
img = imread('data_hires/01861a.tif');
imagesc(img);
colormap('gray');
truesize;
img = im2double(img);
orig_height = floor(size(img, 1) / 3);
orig_width = size(img, 2);
DISPLACE = 20;

orig_B = img(1:orig_height, :);             % B
orig_G = img(orig_height+1: 2*orig_height, :);   % G
orig_R = img(2*orig_height+1:3*orig_height, :);  % R    

CROP = 150;
B = orig_B(CROP: orig_height-CROP, CROP: orig_width-CROP);
G = orig_G(CROP: orig_height-CROP, CROP: orig_width-CROP);
R = orig_R(CROP: orig_height-CROP, CROP: orig_width-CROP);

height = size(B, 1);
width = size(B, 2);

figure(2)
imagesc(B);
colormap('gray');
figure(3)
imagesc(G);
colormap('gray');
figure(4)
imagesc(R);
colormap('gray');

%% Align B, G to R
tic
[x1, y1, x2, y2] = SSD_displacement(B, G, R, DISPLACE);    % align ch1, ch2 to ch3
toc
output1 = zeros(height, width, 3);
output1(:, :, 1) = R;
output1(:, :, 2) = circshift(G, [y2, x2]);
output1(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output1);
title1 = strcat('SSD: align B, G to R; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title1);
truesize;

%% Align B, R to G
[x1, y1, x2, y2] = SSD_displacement(B, R, G, DISPLACE);    % align ch1, ch2 to ch3
output2 = zeros(height, width, 3);
output2(:, :, 1) = circshift(R, [y2, x2]);
output2(:, :, 2) = G;
output2(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output2);
title2 = strcat('SSD: align B, R to G; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title2);
truesize;

%% Align G, R to B
[x1, y1, x2, y2] = SSD_displacement(G, R, B, DISPLACE);    % align ch1, ch2 to ch3
output3 = zeros(height, width, 3);
output3(:, :, 1) = circshift(R, [y2, x2]);
output3(:, :, 2) = circshift(G, [y1, x1]);
output3(:, :, 3) = B;
imagesc(output3);
title3 = strcat('SSD: align G, R to B; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title3);
truesize;

%% Multiscale Align B, G to R
tic
[x1, y1, x2, y2] = SSD_multiscale(B, G, R);    % align ch1, ch2 to ch3
toc
output1 = zeros(height, width, 3);
output1(:, :, 1) = R;
output1(:, :, 2) = circshift(G, [y2, x2]);
output1(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output1);
title1 = strcat('SSD Multiscale: align B, G to R; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title1);
%truesize;
imwrite(output1, '1407.tif');

%% Multiscale Align B, R to G
[x1, y1, x2, y2] = SSD_multiscale(B, R, G);    % align ch1, ch2 to ch3
output2 = zeros(height, width, 3);
output2(:, :, 1) = circshift(R, [y2, x2]);
output2(:, :, 2) = G;
output2(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output2);
title2 = strcat('SSD Multiscale: align B, R to G; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title2);
truesize;

%% Multiscale Align G, R to B
[x1, y1, x2, y2] = SSD_multiscale(G, R, B);    % align ch1, ch2 to ch3
output3 = zeros(height, width, 3);
output3(:, :, 1) = circshift(R, [y2, x1]);
output3(:, :, 2) = circshift(G, [y1, x1]);
output3(:, :, 3) = B;
imagesc(output3);
title3 = strcat('SSD Multiscale: align G, R to B; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title3);
truesize;



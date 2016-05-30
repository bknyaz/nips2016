% This is Matlab demo code for paper "Autoconvolution for unsupervised feature learning".
% It implements the recursive autconvolution operator which can be applied to arbitrary images.
% The code below is intended for the demo purposes only, so it is not optimized and some bad coding practice may occur.
% You should use it as following:
% recursive_autoconv_demo() or recursive_autoconv_demo(path_to_your_image_file) or recursive_autoconv_demo(I),
% where 'path_to_your_image_file' must be the path Matlab is able to find, e.g., recursive_autoconv_demo('stl10_sample1.png'),
% and I - the matrix containing an image, e.g., 96x96x3 for STL-10.
% You can feed MNIST, CIFAR-10 or other images as well.
% In result, you will see a figure with 5 patches and 5 autoconvolution orders (n) from 0 to 4.
% After each run you can observe different results because of randomized cropping and subsampling

function recursive_autoconv_demo(varargin)
% The main demo function

% read the image
if (nargin == 0)
    I = 'stl10_sample1.png';
elseif (nargin == 1)
    I = varargin{1};
else
    error('only one argument can be passed')
end

if (ischar(I))
    I = imread(I);
end
I = single(I);

% let's extract several (random) patches from the image
opts.cropSize = [size(I,1), round(size(I,1)/2), min(size(I,1),[13,13,13])]; % the first one will be as big as the input image
opts.conv_orders = repmat({1:5}, 1, length(opts.cropSize));  

patches = extract_patches_2d(I, opts); % extract patches

% visualize extracted patches
close all
figure('units','normalized','outerposition',[0 0 1 1])
sz = size(patches);
for it = 1:numel(patches)
    subplot(sz(1),sz(1),it);
    imshow(mat2gray(imresize(patches{it},[30,30]))); % resize and scale for visualization purposes
    [a,b] = ind2sub(size(patches),it);
    title(sprintf('patch %d, n = %d', a, b-1), 'FontSize', 10);
end
end

function patches = extract_patches_2d(featureMaps, opts)
% We used this function to collect data points for k-means or ICA

[m,n,~,~] = size(featureMaps);
patches = {};
% take random crops from featureMaps
for crop_iter = 1:length(opts.cropSize)
    rows = randi([1, 1+m-opts.cropSize(crop_iter)]);
    cols = randi([1, 1+n-opts.cropSize(crop_iter)]);
    X_n = autoconv_recursive_2d(featureMaps(rows:rows+opts.cropSize(crop_iter)-1,cols:cols+opts.cropSize(crop_iter)-1,:),...
        max(opts.conv_orders{crop_iter})-1, repmat(opts.cropSize(crop_iter),1,2));
    patches = cat(1,patches,X_n(opts.conv_orders{crop_iter}));
end
end

function X_n = autoconv_recursive_2d(X, n_MAX, filterSize)
% This function implements expression (2) in the paper
% X - a random image (patch) in the spatial domain
% n_MAX - the last autoconvolution order (we use n_MAX <= 4)
% filterSize - desired size of returned patches
% X_n - a collection of autoconvolutional patches of orders n=0,1,...,n_MAX
% Patches are normalized in the range [0,1]

X_n = cell(1,n_MAX+1); % initialize the collection
for n = 0:n_MAX
    if (n > 0)
        X = autoconv_2d(X); % a single iteration of two-dimensional autoconvolution applied to X
        if (n == 1 || rand > 0.5) % for n > 1 decide randomly what to do next 
            X = imresize(X, 0.5); % resize (i.e. take the central part in the frequency domain)
        else
            X = downsample(X, 2, 'space'); % take the central part in the spatial domain
        end
        X = single(mat2gray(real(X))); % normalize in the range [0,1]
    end
    X_n{n+1} = X; % in case n = 0, we just take the input image patch 
    X_n{n+1} = single(mat2gray(imresize(X_n{n+1}, filterSize))); % resize and normalize in the range [0,1]
end
end

function X = autoconv_2d(X)
% This function implements expression (1) in the paper
% input X - an input image in the spatial domain
% output X - a result in the spatial domain of convolving X with itself

X = X-mean(X(:)); % subtract image's mean
% It might be better to subtract channel's mean
% m = mean(mean(X,1),2); 
% X = bsxfun(@minus,X,m);

sz = size(X);
X = padarray(X, sz(1:2)-1, 'post'); % zero-padding to compute linear convolution
X = ifft2(fft2(X).^2); % autoconvolution in the frequency domain 
end

function f = downsample(f, dwn_coef, type, varargin)
% This is a quite general function to take a central part of some signal f with some downsampling coefficient dwn_coef.
% type can be 'freq', otherwise assumed 'spatial'
% varargin can be used to specify the number of dimensions along which downsampling is performed
% the size of output f is defined as size(f)/dwn_coef

if (nargin <= 3)
    n_dimensions = 2;
else
    n_dimensions = varargin{1};
end

if (n_dimensions > 3)
    error('maximum 3 dimensions is supported')
end

if (length(dwn_coef) == 1)
    dwn_coef = repmat(dwn_coef,1,n_dimensions);
elseif (length(dwn_coef) == 2)
    dwn_coef = [dwn_coef,1];
end
if (isequal(lower(type),'freq'))
    f = fftshift(f);
end
sz = size(f);
sz = sz(1:n_dimensions);
sz_new = round(sz./dwn_coef(1:n_dimensions));
d = repmat((sz-sz_new)./2,2,1);
for i=1:n_dimensions
    if (abs(d(1,i)-floor(d(1,i))) > eps)
        d(1,i) = ceil(d(1,i));
        d(2,i) = floor(d(2,i));
    end
end
f = f(d(1,1)+1:end-d(2,1), d(1,2)+1:end-d(2,2), :, :, :);
if (n_dimensions >= 3)
    f = f(:,:,d(1,3)+1:end-d(2,3),:,:);
end
if (isequal(lower(type),'freq'))
    f = ifftshift(f);
end
end

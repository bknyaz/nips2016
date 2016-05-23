% This is Matlab demo code for the paper "Autoconvolution for unsupervised feature learning"
% It implements the recursive autconvolution operator which can be applied to arbitrary images
% The code below is intended for the demo purposes only, so it is not optimized and some bad coding practice may occur
% You should use it as following:
% recursive_autoconv_demo() or recursive_autoconv_demo(path_to_your_image_file) or recursive_autoconv_demo(I),
% where 'path_to_your_image_file' must be the path Matlab is able to find, e.g., recursive_autoconv_demo('stl10_sample2.png'),
% and I - a matrix containing image, e.g., 96x96x3 for STL-10.
% You can feed MNIST, CIFAR-10 or other images as well
% In result, you will see a figure with 5 patches and 5 autoconvolution orders (n) from 0 to 4.

function recursive_autoconv_demo(varargin)

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

% let's extract several (random) patches from the image
opts.cropSize = [size(I,1), round(size(I,1)/2), min(size(I,1),[13,13,13])]; % the first one will be as big as the input image
opts.conv_orders = repmat({1:5}, 1, length(opts.cropSize));  

filters = extract_2D(I, opts);

% visualize extracted patches
close all
figure('units','normalized','outerposition',[0 0 1 1])
sz = size(filters);
for it = 1:numel(filters)
    subplot(sz(1),sz(1),it);
    imshow(mat2gray(imresize(filters{it},[30,30]))); % resize for visualization purposes
    [a,b] = ind2sub(size(filters),it);
    title(sprintf('patch %d, n = %d', a, b), 'FontSize', 10);
end

end

function filters = extract_2D(featureMaps, opts)
% We used this function to collect data points for k-means (or ICA)

[m,n,~,~] = size(featureMaps);
filters = {};
for crop_iter = 1:length(opts.cropSize)
    rows = randi([1, 1+m-opts.cropSize(crop_iter)], 1, 1);
    cols = randi([1, 1+n-opts.cropSize(crop_iter)], 1, 1);
    f1 = featureMaps(rows:rows+opts.cropSize(crop_iter)-1,cols:cols+opts.cropSize(crop_iter)-1,:);
    y_conv = autoconv_recursive_2d(f1, max(opts.conv_orders{crop_iter})-1, repmat(opts.cropSize(crop_iter),1,2));
    filters = cat(1,filters,y_conv(1,opts.conv_orders{crop_iter}));
end
end

function y_conv = autoconv_recursive_2d(x, n_MAX, filterSize)
% This is a function implementing expression (2) in the paper

% x - a random image (patch) in the spatial domain
% n_MAX - the last autoconvolution order (we use n_MAX <= 4)
% filterSize - desired size of produced patches
% y_conv - a collection of autoconvolutional patches for orders n=0,1,...,n_MAX
% patches are normalized in the range [0,1], which is important for k-means in our experience

y_conv = cell(2,n_MAX+1); % initialize the collection

for n = 0:n_MAX
    if (n > 0)
        Y = autoconv_2d(x); % single iteration of autoconvolution applied to x
        x = real(ifft2(Y).* numel(Y)); % inverse Fourier transform
        if (n == 3)
            x = imresize(x, 0.5); % resize to make the image smaller for the next iteration
        elseif (n == 2 || n == 4 || n > 4)
            if (rand > 0.5) % decide randomly what to do next
                x = imresize(x, 0.5); % resize (take the central part in the frequency domain)
            else
                x = downsample(x, 2, 'space'); % take the central part in the spatial domain
            end
        end
        x = mat2gray(real(x)); % normalize in the range [0,1]
        y_conv{1,n+1} = mat2gray(imresize(x,filterSize)); % resize the result to filterSize
    else
        % in case n = 0 we just save an input image patch 
        if (size(x,1) ~= filterSize(1))
            y_conv{1,n+1} = single(mat2gray(imresize(x, filterSize)));
        else
            y_conv{1,n+1} = single(mat2gray(x));
        end
    end
end
end

function Y = autoconv_2d(x)
% This is a function implementing expression (1) in the paper
% x - an input image in the spatial domain
% Y - a result in the frequency domain of convolving x with itself

x = x-mean(x(:)); % subtract mean
sz_org = size(x);
vl = false; % we prefer to use simple Matlab implementation
if (vl) % if Matconvnet
    kernel = x;
    x = padarray(x, sz_org(1:2), 'both');
    Y = fft2(vl_nnconv(single(x),single(kernel),[]));
else
    sz = sz_org(1:2).*2-1; 
    x = padarray(x, sz(1:2)-sz_org(1:2), 'post'); % zero padding to compute linear convolution
    Y = (fft2(x)./numel(x)).^2; % autoconvolution in the frequency domain 

    % to compute autocorrelation:
    % f1 = fft2(x1)./numel(x1);
    % Y = f1.*conj(f1); 
end
end

function f = downsample(f, dwn_coef, type, varargin)
% this a quite general function to take central parts of some signal f with some downsampling coefficient dwn_coef
% for our purposes we could rewrite it in a much simpler way, but this implementation works fine
% type can be 'freq', otherwise assumed 'spatial'
% varargin can be used to specify dimensions along which perform this downsampling
% the size of output f is defined as size(f)/dwn_coef

if (nargin <= 3)
    n_dimensions = 2;
else
    n_dimensions = varargin{1};
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

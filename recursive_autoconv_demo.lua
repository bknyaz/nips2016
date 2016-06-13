-- This is a very basic Lua script alternative to Matlab script recursive_autoconv_demo.m
-- Run it as following: qlua recursive_autoconv_demo.lua 'path_to_your_image_file' or simply qlua recursive_autoconv_demo.lua
-- qlua is used instead of lua to display images
-- In result, you will see 5 windows (one window per patche) with 5 autoconvolution orders (n) from 0 to n_max = 4.
-- After each run you can observe different results because of randomized cropping and subsampling

require 'image'
require 'nn'

local function autoconv_2d(X)
-- This function implements expression (1) in the paper
-- input X - an input image in the spatial domain
-- output X - a result in the spatial domain of convolving X with itself 

X = X - torch.mean(torch.mean(X,2),3):expandAs(X) -- subtract the mean of channels
X = torch.cdiv(X,torch.std(torch.std(X,2),3):expandAs(X)) -- divide by standard variation of channels
s1 = X:size(2)
s2 = X:size(3)
X1 = {}
for c=1,X:size(1) do -- indpendently for channels 
	layer = nn.SpatialConvolution(1,1,s1,s2,1,1,s1,s2)  -- zero-padding to compute linear convolution
	x_tmp = torch.reshape(X:select(1,c), torch.LongStorage{1, s1, s2}) 
	layer.weight = x_tmp;
	table.insert(X1, layer:forward(x_tmp)) -- autoconvolution in the spatial domain 
end
X = torch.cat(X1,1)
return X
end

local function autoconv_recursive_2d(X, n_max, filter_size)
-- This function implements expression (2) in the paper
-- X - a random image (patch) in the spatial domain
-- n_max - the last autoconvolution order (we use n_max <= 4)
-- filter_size - desired size of returned patches
-- X_n - a collection of autoconvolutional patches of orders n=0,1,...,n_max
-- Patches are normalized in the range [0,1]

X = X-torch.min(X)
X = X/torch.max(X)
X_n = {}
for n=0,n_max do
  if n>0 then
	  X = autoconv_2d(X) -- a single iteration of two-dimensional autoconvolution applied to X
	  s1 = X:size(2)
	  s2 = X:size(3)
    -- resize or take a central crop to prevent size increase of X
	  if n == 1 or math.random() > 0.5 then -- for n > 1 decide randomly what to do next 
	    X = image.scale(X, s1/2) 
	  else
      X = image.crop(X, math.floor(s1/4), math.floor(s2/4), math.floor(0.75*s1), math.floor(0.75*s2))
	  end
  end
  -- in case n = 0, we just take the input image patch 
  -- normalize in the range [0,1]
  X = X-torch.min(X)
  X = X/torch.max(X)
  table.insert(X_n, image.scale(X, filter_size))
end
return X_n
end

-- main
-- read the image
local X
if (arg[1]) then
  X = image.load(arg[1])
else
  X = image.load('stl10_sample1.png')
end
local X = image.scale(X, math.min(X:size(2),32)) -- decrease size for images larger than 32x32
local s1 = X:size(2)
crop_size = {s1, math.floor(s1/2), 13, 13, 13} -- extract several (random) patches from the image
n_max = 4
math.randomseed(os.time())
for patch=1,table.getn(crop_size) do
   -- take random crops from image X
   id1 = math.random(0, s1-crop_size[patch]);
   id2 = math.random(0, s1-crop_size[patch]);
   -- resize results to 64 pixels for better visualization
   X_n = autoconv_recursive_2d(image.crop(X,id1,id2,id1+crop_size[patch],id2+crop_size[patch]), n_max, 64) 
   -- visualize extracted patches
   image.display{image=X_n, zoom=1, legend=string.format('patch %d',patch) } -- new window for each patch
end





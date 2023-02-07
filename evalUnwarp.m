function [ms, ld] = evalUnwarp(A, ref, ref_msk)

x = A;
y = ref;
z = ref_msk;

im1=imresize(imfilter(x,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
im2=imresize(imfilter(y,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
im3=imresize(imfilter(z,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');

im1=im2double(im1);
im2=im2double(im2);
im3=im2double(im3);

cellsize=3;
gridspacing=1;

sift1 = mexDenseSIFT(im1,cellsize,gridspacing);
sift2 = mexDenseSIFT(im2,cellsize,gridspacing);

SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;


[vx,vy,~]=SIFTflowc2f(sift1,sift2,SIFTflowpara);

d = sqrt(vx.^2 + vy.^2);
mskk = (im3==0);
ld = mean(d(~mskk));

wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
for s = 1 : 5
    ss(s) = ssim(x, z);
    x = impyramid(x, 'reduce');
    z = impyramid(z, 'reduce');
end
ms = wt * ss;

end

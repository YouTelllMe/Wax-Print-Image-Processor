function imgFilt=guifilter(img,exp1,exp2)

[mImg,nImg]=size(img);
[x,y]=meshgrid(1:nImg,1:mImg);
%returns 2-D grid coordinates based on the coordinates contained in vectors x and y

mask1=1-exp(-exp1*((x-nImg/2).^2+(y-mImg/2).^2)); %    for high pass filtering
mask2=exp(-exp2*((x-nImg/2).^2+(y-mImg/2).^2));   %    for low pass filtering

imgFilt=real(ifft2(ifftshift( fftshift(fft2(img)).*mask1.*mask2 )));

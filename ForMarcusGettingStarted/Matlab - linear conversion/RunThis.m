
clear 
close all

%img = imread('LG_233_U_may 2019_0001.jpg');
%img = rgb2gray(imread('LG_244_U_0.8x001 oct 18 2019.jpg'));
%img = rgb2gray(imread('LG_244_U_0.8x aug 16 2019.jpg'));
%img = rgb2gray(imread('invertTest.jpg'));
img = imread('Dye Images/sep_15_2017 LG_219_U_0.8x.jpg');
exp1=exp(-7);
exp2=exp(-12);
imgFilt=guifilter(img,exp1,exp2);
axis on
imgThresh=(imgFilt>10).*imgFilt;
figure(3);
imagesc(imgThresh);%displays filtered image

FastPeakFind(imgThresh,6);%finds peaks from teeth centers
Fit;

exp1=exp(-5);
exp2=exp(-20);
imgFilt=guifilter(img,exp1,exp2);

Interp

FindTeeth1D

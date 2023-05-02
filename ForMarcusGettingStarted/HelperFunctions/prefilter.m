directory = '/media/alex/Windows/Users/April/Documents/School/Gecko Project/Geck Teeth Image Analysis/TrainingSet/';
%filename = "LG _244_U_0.8x001 july 20 2019.jpg";
%filename = "LG_244_U_0.8x001 nov 29 2019.jpg";
filename = "LG_244_U_0.8x001 oct 18 2019.jpg";
filePath = strcat(directory, filename);

img = rgb2gray(imread(filePath));
exp1=exp(-7);
exp2=exp(-12);
imgFilt=guifilter(img,exp1,exp2);
axis on
imgThresh=(imgFilt>0).*imgFilt;
figure(3);
imagesc(imgThresh);%displays filtered image

cd(directory);
if exist("Filtered_Images", 'dir')
    cd Filtered_Images
    imwrite(imgThresh, filename);
else
    mkdir("Filtered_Images");
    cd Filtered_Images;
    imwrite(imgThresh, filename);
end
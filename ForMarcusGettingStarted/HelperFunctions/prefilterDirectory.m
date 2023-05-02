
function prefilterDirectory(directory, doThresh)

    %cd ../%this line may need editing, sometimes it seems to go above the gecko project directory
    %geckoDirectory = pwd;
    %imageDirectory = fullfile(geckoDirectory, "TrainingSet");
    imageDirectory = directory;
    
    imageFolder = fullfile(imageDirectory,'*.jpg');
    imageDS = imageDatastore(imageFolder);

    for j = 1:length(imageDS.Files)
        %Read image with name
        img = rgb2gray(readimage(imageDS, j));
        imageName = remFolderStruct(imageDS.Files{j});
        %filename = strcat("FILTERED_", imageName);
        filename = imageName;

        %Filter
        %cd fullfile(projectDirectory, "Matlab - linear conversion");
        exp1=exp(-7);
        exp2=exp(-12);
        imgFilt=guifilter(img,exp1,exp2);
        if doThresh
            %axis on
            imgThresh=(imgFilt>0).*imgFilt;
        else
            imgThresh = imgFilt;
        end
        
        %figure(3);
        %imagesc(imgThresh);%displays filtered image

        %Write
        cd(imageDirectory);
        if exist("Filtered_Images", 'dir')
            cd Filtered_Images
            imwrite(imgThresh, filename);
        else
            mkdir("Filtered_Images");
            cd Filtered_Images;
            imwrite(imgThresh, filename);
        end

    end
end

% This function takes an image and trained rcnn location as input, 
% finds the gecko teeth, and returns an array of x,y coordinates of the 
% teeth centres as an m x 2 struct of type double, where m is the number
% of teeth found.
%
% Authour: Alex Fraser
% Project: Gecko teeth dental analysis for Joy Richman's UBC lab
%
% Example function calls:
%
% To call and plot centroids on the image:
%   xyCoords = createToothArray(img, "toothNet_10Image.mat", true); 
% 
% To call not show results:
%   xyCoords = createToothArray(img, "toothNet_10Image.mat"); 
% 
% To call and not show results with superfluous input:
%   xyCoords = createToothArray(img, "toothNet_10Image.mat", false); 
% 

function points = createToothArray(varargin)
    if nargin == 2
        testImage = varargin{1};
        rcnn = varargin{2};
        plotImage = false;
    elseif nargin == 3
        testImage = varargin{1};
        rcnn = varargin{2};
        plotImage = varargin{3};
    end
    
    %Detect teeth
    [bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128);

    %creates centroids to return from bounding boxes
    points(:,1) = bboxes(:,1) + bboxes(:,3) / 2;
    points(:,2) = bboxes(:,2) + bboxes(:,4) / 2;
    
    %Code for visualization
    if plotImage
        outputImage = testImage;
        i = 1;
        for b = bboxes'
            annot = sprintf('%s: (Confidence = %f)', label(i), score(i));
            outputImage = insertObjectAnnotation(outputImage, 'rectangle', b', annot);
            i = i+1;
        end

        %plot results
        imshow(outputImage);
        hold on;
        axis on;
        scatter(points(:,1), points(:,2));
    end
    
end


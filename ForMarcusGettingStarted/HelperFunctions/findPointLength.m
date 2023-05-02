function [pointLength, pointDist] = findPointLength(inputPoints)
%findPointLength() Takes an array of xy coordinates along a hyperbola and
%converts them to a 1D continuous representation along a line. New linear coordinate is centered
%around the zero y coordinate of input data. This method uses point to
%point summation instead of arclength integration.


    %split points into upper and lower to calculate point-to-point
    %teeth positions along jawline
    inputUpper = inputPoints(inputPoints(:,2) >= 0, :);
    inputLower = inputPoints(inputPoints(:,2) < 0, :);
    
    inputUpperPrev = [inputUpper(2:end,:) ; inputLower(1, :)];
    upperSquare = (inputUpper - inputUpperPrev).^2;
    upperDist = sqrt(upperSquare(:,1) + upperSquare(:,2));
    upperDist(end, :) = upperDist(end, :) / 2;
    upperLength = upperDist;
    for n = length(upperLength)-1 : -1 : 1
        upperLength(n) = upperLength(n) + upperLength(n+1);
    end
    
    inputLowerPrev = [inputUpper(end, :) ; inputLower(1:end - 1,:)];
    lowerSquare = (inputLower - inputLowerPrev).^2;
    lowerDist = sqrt(lowerSquare(:,1) + lowerSquare(:,2));
    lowerDist(1, :) = lowerDist(1, :) / 2;
    lowerLength = lowerDist;
    for n = 2:length(lowerLength)
        lowerLength(n) = lowerLength(n) + lowerLength(n-1);
    end
    lowerLength = lowerLength * -1;
    
    pointLength = [upperLength ; lowerLength] * -1;%invert length array to match left/right orientation of jaw images
    pointDist = [upperDist(1:end-1); sqrt(sum((inputUpper(end,:)-inputLower(1,:)).^2)) ; lowerDist(2:end)];
    
    %{
    %alternate form to calculate width distribution and histogram, *MUST MATCH RESULTS ABOVE IN pointDist!!!*
    %This seems to match for the tested inputs. It might run faster in
    matlab, need to check and run the faster code. OPTIMIZE HERE
    pointSquare = (shiftPoints(1:end-1,:) - shiftPoints(2:end,:)).^2;
    pointDistAlt = sqrt(pointSquare(:,1) + pointSquare(:,2));
    %}

end


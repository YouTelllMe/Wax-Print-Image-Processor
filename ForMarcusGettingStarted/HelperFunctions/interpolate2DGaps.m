function [toothGapPoints, gapPoints] = interpolate2DGaps(toothPoints)
    toothMaxWidth = 70;
    gapPoints = [];
    
    i = 1;
    while i < length(toothPoints)-1
        if sqrt((toothPoints(i+1,1)-toothPoints(i,1))^2 + (toothPoints(i+1,2)-toothPoints(i,2))^2) > toothMaxWidth
            %interpolate gap coordinates
            newGap = [(toothPoints(i+1,1)+toothPoints(i,1))/2 (toothPoints(i+1,2)+toothPoints(i,2))/2];
            toothPoints = [toothPoints(1:i, :); newGap; toothPoints(i+1:end, :)];
            gapPoints = [gapPoints; newGap];
            i = i+1;
        end
        i = i+1;
    end
    
    toothGapPoints = toothPoints;
end
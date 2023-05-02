

function showToothPairs(img, toothGapPoints, centerToothGapIndex, majorAxis)
    leftIndex = centerToothGapIndex - 1;
    rightIndex = centerToothGapIndex + 1;
    majorAxis = [majorAxis(1)/norm(majorAxis); majorAxis(2)/norm(majorAxis)];%converts to unit vector
    
    figure
    hold on
    scatter(toothGapPoints(:,1), toothGapPoints(:,2))
    line([toothGapPoints(centerToothGapIndex,1); toothGapPoints(centerToothGapIndex,1) + majorAxis(1)*size(img, 1)], [0; majorAxis(2)*size(img, 1)])
    while leftIndex > 0 && rightIndex <= length(toothGapPoints)
        line([toothGapPoints(leftIndex,1); toothGapPoints(rightIndex,1)], [toothGapPoints(leftIndex,2); toothGapPoints(rightIndex,2)])
        
        leftIndex = leftIndex - 1;
        rightIndex = rightIndex +1;
    end
    
    hold off
end
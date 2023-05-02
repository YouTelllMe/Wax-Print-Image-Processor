function dotSum = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex)
    
    leftIndex = centerIndex - 1;
    rightIndex = centerIndex + 1;
    dotSum = 0;
    
    while leftIndex > 0 && rightIndex <= length(toothGapPoints)
        pairVec = [toothGapPoints(rightIndex, 1) - toothGapPoints(leftIndex, 1), toothGapPoints(rightIndex, 2) - toothGapPoints(leftIndex, 2)];
        pairVec = pairVec / norm(pairVec); %scales vector to unit length
        dotSum = dotSum + dot(pairVec, majorJawAxis)^2;
        
        leftIndex = leftIndex - 1;
        rightIndex = rightIndex +1;
    end
    
end
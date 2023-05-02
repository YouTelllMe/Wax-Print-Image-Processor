

function [newIndex, linearShift] = centerJaw(toothGapLengths, toothGapPoints, majorJawAxis, img)
    assert(length(toothGapLengths) == length(toothGapPoints), "1D and 2D tooth and gap positions must have same count!");
    show = false;
    [~, centerIndex] = min(abs(toothGapLengths));
    
    initialScore = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex);
    shiftLeftOneScore = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex - 1);
    shiftLeftTwoScore = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex - 2);
    shiftRightOneScore = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex + 1);
    shiftRightTwoScore = scoreCenterTooth(toothGapPoints, majorJawAxis, centerIndex + 2);
    
    if show
        showToothPairs(img, toothGapPoints, centerIndex - 2, majorJawAxis)
        showToothPairs(img, toothGapPoints, centerIndex - 1, majorJawAxis)
        showToothPairs(img, toothGapPoints, centerIndex, majorJawAxis)
        showToothPairs(img, toothGapPoints, centerIndex + 1, majorJawAxis)
        showToothPairs(img, toothGapPoints, centerIndex + 2, majorJawAxis)
    end
    
    [~, shift] = min([shiftLeftTwoScore; shiftLeftOneScore; initialScore; shiftRightOneScore; shiftRightTwoScore]);
    shift = shift - 3;
    
    newIndex = centerIndex + shift;
    
    linearShift = toothGapLengths(newIndex);
    
end
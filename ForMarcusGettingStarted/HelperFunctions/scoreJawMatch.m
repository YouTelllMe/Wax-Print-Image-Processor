%Score function used to align 1 dimensional vectors of continuous
%datapoints. Turned out to not work very well for tooth data.
function score = scoreJawMatch(prevVec, nextVec)
    exponent = 2;%2 seems to work best, lower and higher are both worse
    matrixPrev = repmat(prevVec', 1, length(nextVec));
    matrixNext = repmat(nextVec, length(prevVec), 1); 

    diffMatrix = abs(matrixNext - matrixPrev).^exponent;
    columnMins = min(diffMatrix);
    
    matrixPrev2 = repmat(prevVec, length(nextVec), 1);
    matrixNext2 = repmat(nextVec', 1, length(prevVec));

    diffMatrix2 = abs(matrixPrev2 - matrixNext2).^exponent;
    columnMins2 = min(diffMatrix2);
    
    score = sum(columnMins) + sum(columnMins2);
end
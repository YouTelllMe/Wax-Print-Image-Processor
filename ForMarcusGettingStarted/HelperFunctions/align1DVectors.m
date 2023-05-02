%This alignment function is deprecated. centerJaw has superior jaw
%alignments. The scoring matrix function was clever though!
function diffSol = align1DVectors(prevVec, nextVec)
    fun = @(diff) scoreJawMatch(prevVec, nextVec-diff);
    diffSol = fminsearch(fun, 0);
    
    %alignedNextVector = nextVec - diffSol;
end
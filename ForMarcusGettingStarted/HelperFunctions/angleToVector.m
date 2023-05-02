%Returns the angle from second vector to first in radians, in full range [-pi, pi]. Only works for 2D vectors. 
%Useful for determining how to rotate vector2 onto vector1.
%
%Uses a slightly different formula using cross product for angle 
%determination that is more accurate in matlab than using acos(). The low precision in the
%acos() function makes the standard solution unable to detect small angle differences.

function theta2 = angleToVector(refVector, testVector)
    if length(refVector) ==2 && length(testVector) == 2 
        refVector = [refVector ; 0];
        testVector = [testVector ; 0];
    end
    theta = atan2(cross(testVector, refVector),dot(refVector,testVector));
    theta2 = theta(3);
end

%NOTE: might be a better way to get the sign for transformation that
%generalizes to 3D, but 3D vector transformation is not needed in current
%project
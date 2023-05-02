%Returns the angle between two vectors in radians, range [0, pi]. 3D vectors untested, but should work?
%Uses a different formula than
%normal that is more accurate in matlab. The low precision in the
%acos() function makes the standard solution unable to detect small angle differences.
function theta = angleBetween(vector1, vector2)
    if length(vector1) ==2 && length(vector2) == 2
        vector1 = [vector1 ; 0];
        vector2 = [vector2 ; 0];
    end
    theta = atan2(norm(cross(vector1,vector2)),dot(vector1,vector2));
end
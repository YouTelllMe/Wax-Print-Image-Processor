% Returns a rotation matrix to rotate a 2d vector counterclockwise by the
% theta argument in radians.

function rotMat = rotateMat(theta)
     rotMat = [cos(theta), -sin(theta); sin(theta), cos(theta)];
end
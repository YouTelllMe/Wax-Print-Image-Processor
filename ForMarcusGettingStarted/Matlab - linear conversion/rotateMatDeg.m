% Returns a rotation matrix to rotate a 2d vector, which rotates by the
% theta argument in degrees.

function rotMat = rotateMatDeg(theta)
     rotMat = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
end
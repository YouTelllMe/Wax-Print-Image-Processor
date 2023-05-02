% seems to be finding the vertex and coefficients of a standard parabola
% fitted to the points in the x,y 

function [vertex,theta, a] = myfit(x,y)
    xy=[x(:),y(:)].';
    theta= fminsearch(@(theta) cost(theta,xy), 45);%minimizes an angle, and stores angle in theta variable?
    [~,coeffs]=cost(theta,xy);
    [a,b,c]=deal(coeffs(1),coeffs(2), coeffs(3));
    xv=-b/2/a;
    vertex=rotateMatDeg(-theta)*[xv;polyval(coeffs,xv)];
end
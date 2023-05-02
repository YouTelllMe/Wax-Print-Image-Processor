%finds the best fit parabola?
% 

function [Cost,coeffs,xx,yy] = cost(theta,xy)
    rotatedxy=rotateMatDeg(theta)*xy;
    [xx,idx]=sort(rotatedxy(1,:));
    yy=rotatedxy(2,idx);
    [coeffs,S]=polyfit(xx,yy,2); %quadratic parabola function fit? polyfit returns coeffs for eqn. of form p(x) = ax^2 + bx + c
    Cost=S.normr;


function arclength = findArcLength(x1, y1, xv, yv, aa, bb) 

    %calculate arclength from vertex to point on conic
    syms x;
    if y1 >= 0%upper half of conic
        sign = 1;
    elseif y1 < 0%lower half of conic
        sign = -1;
    end
    
    fun = @(x) sqrt(1 + abs(((bb*x)/(aa*sqrt(bb*(1-x.^2/aa)))).^2));
    arclength = sign*integral(fun, xv, x1, 'ArrayValued', true);
    
end
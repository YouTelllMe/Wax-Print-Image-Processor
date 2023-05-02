%given a point, returns the nearest point that lies along the conic section
%defined by aa and bb

function point = findNearest(x1, y1, aa, bb) 
    warning('off', 'symbolic:solve:PossiblySpuriousSolutions');

    syms x;

    if y1 >= 0%upper half of ellipse/hyperbola
        x = solve((2*bb*x*(y1-sqrt(bb*(1-x^2/aa))))/(aa*sqrt(bb*(1-x^2/aa)))-2*(x1-x) == 0, x);
        j = 1;
        while j <= length(x)
        %This loop removes extraneous complex solutions
            if imag(x(j)) ~= 0
                x(j) = [];
                j = j-1;%To compensate for reduction in array size
            end
            j = j+1;
        end
        x = double(min(x));
        y = sqrt(bb*(1 - x^2/aa));
    elseif y1 < 0%lower half of ellipse/hyperbola
        x = solve(-(2*bb*x*(y1+sqrt(bb*(1-x^2/aa))))/(aa*sqrt(bb*(1-x^2/aa)))-2*(x1-x) == 0, x);
        j = 1;
        while j <= length(x)
        %This loop removes extraneous complex solutions
            if imag(x(j)) ~= 0
                x(j) = [];
                j = j-1;%To compensate for reduction in array size
            end
            j = j+1;
        end
        x = double(min(x));
        y = -sqrt(bb*(1 - x^2/aa));
    end
    
    warning('on', 'symbolic:solve:PossiblySpuriousSolutions');
    
    point = [x y];
end
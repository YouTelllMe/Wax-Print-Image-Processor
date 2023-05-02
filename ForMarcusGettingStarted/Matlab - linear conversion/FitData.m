% Finds a hyperbola fit by minimizing error, then returns coefficients
% Created by Roxanne

function v=FitData(x,y,v0,exponent)

M = [x.^2 x.*y y.^2 x y]; %Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 is the bivariate quadratic form
b = -ones(size(x));
fun = @(v) sum(abs(M*v-b).^exponent);%error function?
v = fminsearch(fun,v0);

end
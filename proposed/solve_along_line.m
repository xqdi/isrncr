function eta = solve_along_line(M, point, x, y, g, Hy, sigma)
% Minimize the function h(eta) = f(x + eta*y) where
%     f(s) = <s, H[s]> + <g, s> + sigma * ||s||^3.
%
% Inputs: A manifold M, a point on the manifold, vectors x, y, g, and H[y]
%         on the tangent space T_(point)M, and a constant sigma.
%
% Outputs: minimizer eta if eta is positive real; otherwise returns eta = 0

% This file is part of Manopt: www.manopt.org.
% Original authors: May 2, 2019,
%    Bryan Zhu, Nicolas Boumal.
% Contributors:
% Change log: 
    
    % Magnitude tolerance for imaginary part of roots.
    im_tol = 1e-05;

    xx = double(M.inner(point, x, x));
    xy = double(M.inner(point, x, y));
    yy = double(M.inner(point, y, y));
    yHy = double(M.inner(point, y, Hy));
    const = double(M.inner(point, x, Hy) + M.inner(point, g, y));
    
    func = @(a) a * const + 0.5 * a^2 * yHy + (sigma/3) * M.norm(point, M.lincomb(point, 1, x, a, y))^3;

    
    % calculate the gradient of func w.r.t a and make equal to zero
    s2 = sigma * sigma;
    s2_xy = s2 * xy;
    yy2 = yy * yy;
    A = s2 * yy2 * yy;
    B = 4 * s2_xy * yy2;
    C = 5 * s2_xy * xy * yy + s2 * xx * yy2 - yHy^2;
    D = 2 * s2_xy * (xy^2 + xx * yy) - 2 * yHy * const;
    E = s2_xy * xx * xy - const^2;
    coeffs = [A, B, C, D, E];
    poly_roots = roots(coeffs);
    eta = 0;
    min_val = func(0);
    for root = poly_roots.'
        if root < 0 || abs(imag(root)) > im_tol
            continue;
        end
        rroot = real(root);
        root_val = func(rroot);
        if root_val < min_val
            eta = rroot;
            min_val = root_val;
        end
    end
end
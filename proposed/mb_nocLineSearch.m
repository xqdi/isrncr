function [xc, fx, grad, oraclecalls_inner, alpha, flag] = mb_nocLineSearch(problem,x,grad0, dir, of, old_of0, samp_grad_idx, storedb, key, Hp_conj, sigma0, j)

% implementation of strong wolfe line search
% 
% original c++ code by m.siggel@dkfz.de
% import to matlab by m.bangert@dkfz.de
% reference: nocedal: numerical optimization 3.4 line search methods
%
% f       - function handle of objective function
% gradF   - function handle of gradient
% x       - current iterate
% dir     - search direction
% slope0  - gradient of function at x in direction dir
% alphaLo - alpha at lower objective function value for zoom
% alphaHi - alpha at higher objectvie function value for zoom
% of_0
% oflo

% parameter for sufficient decrease condition
c1 = 0.0001;
% parameter for curvature condition
c2 = 0.9;

if c1 > c2
  error('c1 > c2\n');
end

of_0 = of;
%n1 = problem.M.norm(x, dir)
%n2 = problem.M.norm(x, grad0)
%sign(sum(sum(dir)))
%sign(sum(sum(grad0)))
%sign(n1)
%sign(n2)
slope0 = problem.M.inner(x, grad0, dir);

alphaMax = 100.0;
alpha_0  = 0;
fprintf("slope0: %.9f\n", slope0);
if (old_of0 ~= Inf) && (slope0 ~= 0)
  alpha_1 = min(1.0, 2.02*(of_0 - old_of0) / slope0);
  if (alpha_1 < 0.0)
    alpha_1 = 1.0;
  end
  alpha_1 = min(alpha_1, alphaMax);
else
  alpha_1 = 1.0;
end

oraclecalls_inner = 0;
flag = 1;
xc  = problem.M.exp(x, dir, 1e-8);
of_a1  = getCost(problem, xc, storedb);
if (of_a1 >= of_0)
  alpha = 0;
  xc = x;
  fx = of_0;
  grad = grad0;
  flag = 0;
  return;
end

xc  = problem.M.exp(x, dir, alpha_1);
of_a1  = getCost(problem, xc, storedb);

of_a0 = of_0;
slope_a0 = slope0;

iter = 0;
samp_grad_size = length(samp_grad_idx);

while (iter < 11)
  if (alpha_1 == 0) || (alpha_0 == alphaMax)
    fx = of_1;
    grad = Inf;
    if (alpha_1 == 0) 
      alpha = 0;
      fx = of_0;
    end
    break;
  end

  %fprintf("ofa1 of0: %f %f %f %f\n", of_a1, of_0, c1*alpha_1*slope0, of_0 + c1*alpha_1*slope0)
  % check if current iterate violates sufficient decrease
  if (of_a1 > of_0 + c1*alpha_1*slope0) || ((of_a1 >= of_a0) && iter > 0)
    fprintf("Entry1\n");
    [alpha, n_oraclecalls, fx, grad, xc, flag] = nocZoom(problem, x,grad0,dir, alpha_0, alpha_1, of_a0,of_a1,slope_a0, of_0, slope0 ,c1,c2,samp_grad_idx, storedb, key, Hp_conj, sigma0, j, 1);
    oraclecalls_inner = oraclecalls_inner + n_oraclecalls;
    break;
  end

  if (0 == samp_grad_size)
    grad = getGradient(problem, xc, storedb);
    oraclecalls_inner = oraclecalls_inner + problem.ncostterms;
  else
    %samp_grad_idx = randsample(problem.ncostterms, samp_grad_size);
    grad = getPartialGradient(problem, xc, samp_grad_idx, storedb);
    oraclecalls_inner = oraclecalls_inner + samp_grad_size;
  end

  %[U, S, V] = svd(dir, "econ");
  %Tdir = [x*V U]*[-sin(S*alpha_1);cos(S*alpha_1)]*(S*V');

  trans_dir = problem.M.transp(xc, dir);
  slope_a1 = problem.M.inner(xc, grad, trans_dir);
  
  %fprintf("slope_a1 c2s0: %f %f\n", slope_a1, -c2*slope0)
  % current iterate has sufficient decrease, but are we too close?
  if(abs(slope_a1) <= -c2*slope0)
    % strong wolfe fullfilled, quit
    fprintf("Entry2\n");
    alpha = alpha_1;
    fx = of_a1;
    break;
  end
  % are we behind the minimum?
  if (slope_a1 > 0.0)
    fprintf("Entry3\n");
    % there has to be an acceptable point between alpha_0 and alpha_1
    [alpha, n_oraclecalls, fx, grad, xc, flag] = nocZoom(problem, x,grad0,dir,alpha_1, alpha_0, of_a1,of_a0,slope_a1, of_0, slope0 ,c1,c2,samp_grad_idx, storedb, key, Hp_conj, sigma0, j, 0);
    oraclecalls_inner = oraclecalls_inner + n_oraclecalls;
    break;
  end

  alpha_2 = min(2 * alpha_1, alphaMax);
  alpha_0 = alpha_1;
  alpha_1 = alpha_2;
  of_a0 = of_a1;

  xc  = problem.M.exp(x, dir, alpha_1);
  of_a1  = getCost(problem, xc, storedb);
  slope_a0 = slope_a1;
  iter = iter + 1;
  fprintf("Entry4\n");
end

if (iter == 11) 
  alpha = alpha_1;
  fx = of_a1;
  grad = Inf;
end

end

function [xmin] = _cubicmin(a, fa, fpa, b, fb, c, fc)
  C = fpa;
  db = b - a;
  dc = c - a;
  denom = (db * dc)^2 * (db - dc);
  v1 = fb - fa - C*db;
  v2 = fc - fa - C*dc;
  A = dc^2 * v1 - db^2 * v2;
  B = -dc^3 * v1 + db^3 * v2;
  if (abs(denom) < 1e-18)
    xmin = Inf;
    return;
  end
  A = A / denom;
  B = B / denom;
  radical = B^2 - 3*A*C;
  if (abs(A) < 1e-18)
    xmin = Inf;
    return;
  end
  xmin = a + (-B + sqrt(radical)) / (3*A);
end

function [xmin] = _quadmin(a, fa, fpa, b, fb)
  D = fa;
  C = fpa;
  db = b - a;
  if (abs(db) < 1e-18)
    xmin = Inf;
    return;
  end
  B = (fb - D - C*db) / (db^2);
  if (abs(B) < 1e-18)
    xmin = Inf;
    return;
  end
  xmin = a - C / (2*B);
end

function [alpha, n_oraclecalls, fx, grad, xc, flag] = nocZoom(problem, x,grad0,dir,alphaLo,alphaHi,ofLo,ofHi,slopeLo,of_0,slope0,c1,c2,samp_grad_idx, storedb, key, Hp_conj, sigma0, j, ch)
% this function is only called by mb_nocLineSearch - everything else does
% not make sense!
n_oraclecalls = 0;
samp_grad_size = length(samp_grad_idx);

i = 0;
delta1 = 0.2;
delta2 = 0.1;
f_rec = of_0;
a_rec = 0;
a_j = Inf;
flag = 1;
if (j == 0)
  maxiter = 10;
else
  maxiter = 10;
end
while (i < maxiter)
  dalpha = alphaHi - alphaLo;
  if (dalpha < 0)
    a = alphaHi;
    b = alphaLo;
  else
    a = alphaLo;
    b = alphaHi;
  end

  if (i > 0) 
    cchk = delta1 * dalpha;
    a_j = _cubicmin(alphaLo, ofLo, slopeLo, alphaHi, ofHi, a_rec, f_rec);
  end

  if (i == 0) || (a_j == Inf) || (a_j > b - cchk) || (a_j < a + cchk)
    qchk = delta2 * dalpha;
    a_j = _quadmin(alphaLo, ofLo, slopeLo, alphaHi, ofHi);
    if (a_j == Inf) || (a_j > b - qchk) || (a_j < a + qchk)
      a_j = alphaLo + 0.5*dalpha;
    end
  end

  xc = problem.M.exp(x, dir, a_j);
  of = getCost(problem, xc, storedb);
  %Halpconj = problem.M.lincomb(x, 0, problem.M.zerovec(x), a_j, Hp_conj);
  %alpha_conj = problem.M.lincomb(x, 0, problem.M.zerovec(x), a_j, dir);
  %alpconj_norm = problem.M.norm(x, alpha_conj);
  %of = of_0 + problem.M.inner(x, grad0, alpha_conj) + 0.5 * problem.M.inner(x, alpha_conj, Halpconj) + (sigma0/3) * alpconj_norm^3;
  n_oraclecalls = n_oraclecalls + problem.ncostterms;

  if (j > 0) && (ch == 1) && (of < of_0)
    alpha = a_j;
    fx = of;
    grad = Inf;
    break;
  end
  fprintf("D: %.9f %.9f %.4f %.4f %.9f %.9f\n", of, of_0, c1*a_j*slope0, ofLo, alphaHi, a_j);
  if (of > of_0 + c1*a_j*slope0) || (of >= ofLo)
    % if we do not observe sufficient decrease in point alpha, we set
    % the maximum of the feasible interval to alpha
    f_rec = ofHi;
    a_rec = alphaHi;
    alphaHi = a_j;
    ofHi = of;
  else
    if (0 == samp_grad_size)
      grad = getGradient(problem, xc, storedb);
      n_oraclecalls = n_oraclecalls + problem.ncostterms;
    else
      %samp_grad_idx = randsample(problem.ncostterms, samp_grad_size);
      grad = getPartialGradient(problem, xc, samp_grad_idx, storedb);
      n_oraclecalls = n_oraclecalls + samp_grad_size;
    end
    
    %[U, S, V] = svd(dir, "econ");
    %Tdir = [x*V U]*[-sin(S*a_j);cos(S*a_j)]*(S*V');
    trans_dir = problem.M.transp(xc, dir);
    slopec = problem.M.inner(xc, grad, trans_dir);

    fprintf("E: %.4f %.4f\n", abs(slopec), -c2*slope0);
    if (abs(slopec) <= -c2*slope0)
      alpha = a_j;
      fx = of;
      break;
    end

    fprintf("F: %.4f %.4f %.4f\n", slopec, alphaHi-alphaLo, alphaLo);
    if (slopec*(alphaHi-alphaLo) >= 0) % if slope positive and alphaHi > alphaLo  
      f_rec = ofHi;
      a_rec = alphaHi;  
      alphaHi = alphaLo;
      ofHi = ofLo;
    else
      f_rec = ofLo;
      a_rec = alphaLo;
    end
    alphaLo = a_j;
    ofLo  = of;
    slopeLo = slopec;
  end
  i = i + 1;
end

if (i >= maxiter)
  if (j > 0) && (ch == 1)
    alpha = 0;
    fx = of_0;
    xc = x;
    grad = grad0;
    flag = 0;
  else
    alpha = a_j;
    fx = of;
    grad = Inf;
  end
end

end



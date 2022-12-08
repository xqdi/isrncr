function [alpha, fx_new, Tdir] = exact_line_search(problem,x, fx, grad0, dir, storedb, key)
  lr = 0.000001;
  alpha = 0.0;
  iter = 0;
  dir = getGradient(problem, x, storedb);
  dir = -dir;
  [U, S, V] = svd(dir, "econ");
  while (iter < 1000)
    
    %T_dir = ([x*V U]*[-sin(S);cos(S)]*U'+(eye(nrows, nrows)-U*U'))*prev_mgrad;
    Tdir = [x*V U]*[-sin(S*alpha);cos(S*alpha)]*(S*V');
    %Tdir = problem.M.tangent(xc, Tdir);
    %grad_m0 = problem.M.inner(x, grad0, dir);

    xc = problem.M.exp(x, dir, alpha);
    grad = getGradient(problem, xc, storedb);
    MM = problem.M.norm(xc, grad);
    fprintf("MM %f\n", MM);
    %trans_dir = problem.M.transp(xc, dir);
    %trans_grad = problem.M.transp(x, grad);
    %M1 = problem.M.norm(x, dir);
    %M2 = problem.M.norm(xc, Tdir);
    grad_m = problem.M.inner(xc, grad, Tdir);
    sign(grad(1,1))
    sign(Tdir(1,1))
    sign(xc(1,1))
    fx_new  = getCost(problem, xc, storedb);
    %sign(grad_m)
    fprintf("%d: norm grad %f %f %f\n", iter, alpha, grad_m, fx_new);
    if (abs(grad_m) < 1e-5)
      break;
    end
    alpha = alpha - lr * grad_m;
    %fprintf("alpha %f\n", alpha);
    iter = iter + 1;
  end
  xc = problem.M.exp(x, dir, alpha);
  fx_new  = getCost(problem, xc, storedb);
  Tdir = [x*V U]*[-sin(S*alpha);cos(S*alpha)]*(S*V');
  %Tdir = problem.M.tangent(xc, Tdir);
  fprintf("X XNEW %f %f\n", fx, fx_new);
end

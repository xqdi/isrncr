function [eta, Heta, hesscalls, stop_str] = conjugte_solver(problem, x, grad, gradnorm, sigma, samp_hess_idx, samp_hess_p, options, storedb, key)
% Subproblem solver for ARC based on a nonlinear conjugate gradient method.
%
% [eta, Heta, hesscalls, stop_str, stats] = 
%     arc_conjugate_gradient(problem, x, grad, gradnorm, sigma, options, storedb, key)
%
% This routine approximately solves the following problem:
%
%   min_{eta in T_x M}  m(eta),  where
%
%       m(eta) = <eta, g> + .5 <eta, H[eta]> + (sigma/3) ||eta||^3
%
% where eta is a tangent vector at x on the manifold given by problem.M,
% g = grad is a tangent vector at x, H[eta] is the result of applying the
% Hessian of the problem at x along eta, and the inner product and norm
% are those from the Riemannian structure on the tangent space T_x M.
%
% The solve is approximate in the sense that the returned eta only ought
% to satisfy the following conditions:
%
%   ||gradient of m at eta|| <= theta*||eta||^2   and   m(eta) <= m(0),
%
% where theta is specified in options.theta (see below for default value.)
% Since the gradient of the model at 0 is g, if it is zero, then eta = 0
% is returned. This is the only scenario where eta = 0 is returned.
%
% The approximation proccess can be compared with that in 
% <Trust-Region Methods on Riemannian Manifolds>
%
% In the subproblem with cubic regularization, the tCG removed the stop criteria 
% \conj*H*\conj < 0 and ||\eta|| >= radius, and replaced the computation of 
% alpha = <r,r>/(conj*H*conj) with solving a polynomial
%
% Numerical errors can perturb the described expected behavior.
%
% Inputs:
%     problem: Manopt optimization problem structure
%     x: point on the manifold problem.M
%     grad: gradient of the cost function of the problem at x
%     gradnorm: norm of the gradient, often available to the caller
%     sigma: cubic regularization parameter (positive scalar)
%     options: structure containing options for the subproblem solver
%     storedb, key: caching data for problem at x
%
% Options specific to this subproblem solver:
%   theta (0.25)
%     Stopping criterion parameter for subproblem solver: the gradient of
%     the model at the returned step should have norm no more than theta
%     times the squared norm of the step.
%   maxinner (the manifold's dimension)
%     Maximum number of iterations of the conjugate gradient algorithm.
%   beta_type ('P-R')
%     The update rule for calculating beta:
%     'F-R' for Fletcher-Reeves, 'P-R' for Polak-Ribiere, and 'H-S' for
%     Hestenes-Stiefel.
%
% Outputs:
%     eta: approximate solution to the cubic regularized subproblem at x
%     Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%           is often naturally available to the subproblem solver at the
%           end of execution, so that it may be cheaper to return it here.
%     hesscalls: number of Hessian calls during execution
%     stop_str: string describing why the subsolver stopped
%     stats: a structure specifying some statistics about inner work - 
%            we record the model cost value and model gradient norm at each
%            inner iteration.

% This file is part of Manopt: www.manopt.org.
% Original authors: May 2, 2019,
%    Bryan Zhu, Nicolas Boumal.
% Contributors:
% Change log: 
%
%   Aug. 19, 2019 (NB):
%       Option maxiter_cg renamed to maxinner to match trustregions.

% TODO: Support preconditioning through getPrecon().

    % Some shortcuts
    M = problem.M;
    n = M.dim();
    % Counter for Hessian calls issued
    hesscalls = 0;
    samp_hess_size = length(samp_hess_idx);
    eta = M.zerovec(x);
    Heta = M.zerovec(x);
    nrows = size(x, 1);
    is_first_alpha = false;
    alpha0 = -1;
    
    % If the gradient has norm zero, return a zero step
    if gradnorm == 0
        stop_str = 1; % 'Cost gradient is zero';
        stats = struct('gradnorms', 0, 'func_values', 0);
        return;
    end

    % Set local defaults here
    localdefaults.theta = 1e-2;
    localdefaults.maxinner = n;
    localdefaults.beta_type = 'P-R';
    localdefaults.subproblemstop = 'sqrule';
    
    % Merge local defaults with user options, if any
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    

    % Initialize variables needed for calculation of conjugate direction
    prev_mgrad = grad;  % Because Delta(m(0)) = grad, and eta=0 actually in this case
    norm_r0 = M.norm(x, prev_mgrad);
    prev_conj = M.lincomb(x, -1, prev_mgrad);
    
    % Main conjugate gradients iteration
    maxiter = min(options.maxinner, 50);
    gradnorms = zeros(maxiter, 1);
    func_values = zeros(maxiter, 1);
    gradnorm_reached = false;
    j = 1;
    norm_prev = norm_r0;
    
    while j < maxiter
        % Calculate the Cauchy point as our initial step
        if samp_hess_size >= problem.ncostterms
            Hp_conj = getHessian(problem, x, prev_conj, storedb, key);
        else
            Hp_conj = getPartialHessian(problem, x, prev_conj, samp_hess_idx, samp_hess_p, storedb, key); 
        end
        hesscalls = hesscalls + 1;
        
        % Find the optimal step in the conjugate direction
        alpha = solve_along_line(M, x, eta, prev_conj, grad, Hp_conj, sigma);
        
        eta = M.lincomb(x, 1, eta, alpha, prev_conj);
        eta_norm = M.norm(x, eta);
        Heta = M.lincomb(x, 1, Heta, alpha, Hp_conj);
        
	      % Calculate the gradient of the model
        %new_mgrad = M.lincomb(x, 1, Heta, 1, grad);
        %new_mgrad = M.lincomb(x, 1, new_mgrad, sigma*eta_norm, eta);
        %x_next = M.exp(x, eta);
        %new_mgrad = M.tangent(x_next, new_mgrad);
        new_mgrad = M.lincomb(x, 1, prev_mgrad, alpha, Hp_conj);
        new_mgrad = M.tangent(x, new_mgrad);
        norm_r = M.norm(x, new_mgrad);
	
        if j >= options.mininner && norm_r <= norm_r0*min(norm_r0^options.theta,options.kappa)
            stop_str = 6;
            gradnorm_reached = true;
            break;
        end
        
        if alpha == 0
            stop_str = 3; % 'Unable to make further progress in search direction';
            gradnorm_reached = true;
            break;
        end

        %if ~is_first_alpha
        %    alpha0 = alpha;
        %    is_first_alpha = true;
        %elseif (j >= options.mininner && alpha <= alpha0*min(alpha0^options.theta,options.kappa))

        if alpha < 1e-8
            stop_str = 3; % 'Unable to make further progress in search direction';
            gradnorm_reached = true;
            break;
        end

                              
        % Compute some statistics
        gradnorms(j) = norm_r;
        func_values(j) = M.inner(x, grad, eta) + 0.5 * M.inner(x, eta, Heta) + (sigma/3) * eta_norm^3;
        
        %if func_values(j) < M.inner(x, grad, eta) * 0.9
        %    stop_str = 2;
        %    break;
        %end

        if options.verbosity >= 4
            fprintf('\nModel grad norm: %.16e, Iterate norm: %.16e', gradnorms(j), eta_norm);
        end

        % Check termination condition
        % TODO -- factor this out, as it is the same for all subsolvers.
        % TODO -- I haven't found a scenario where sqrule doens't win.
        % TODO -- 1e-4 might be too small (g, s, ss seem equivalent).
        %{
        switch lower(options.subproblemstop)
            case 'sqrule'
                stop = (gradnorms(j) <= options.theta * eta_norm^2);
            case 'grule'
                stop = (gradnorms(j) <= min(1e-4, sqrt(gradnorms(1)))*gradnorms(1));
            case 'srule'
                stop = (gradnorms(j) <= min(1e-4, eta_norm)*gradnorms(1));
            case 'ssrule'
                stop = (gradnorms(j) <= min(1e-4, eta_norm/max(1, sigma))*gradnorms(1));
	          case 'my'
                %gradnorms(j) = grad + Heta + sigma * eta_norm * eta;
                stop = gradnorms(j) <= options.kappa * min(1, eta) * gradnorm;
            otherwise
                error(['Unknown value for options.subproblemstop.\n' ...
                  'Possible values: ''sqrule'', ''grule'', ' ...
                  '''srule'', ''ssrule''.\n']); % ...
                  % 'Current value: ', options.subproblemstop, '\n']);
        end
        
        if stop
            stop_str = 2; % 'Model grad norm condition satisfied';
            gradnorm_reached = true;
            break;
        end
        %}
        
        %if j > 1 && func_values(j) > func_values(j-1)
        %    stop_str = 5; % numerical errors stop
        %    break;
        %end
        


        switch upper(options.beta_type)
            case 'F-R'
                beta = M.inner(x, new_mgrad, new_mgrad) / M.inner(x, prev_mgrad, prev_mgrad);
            case 'P-R'
	    	        delta = M.lincomb(x, 1, new_mgrad, -1, prev_mgrad);
                beta = max(0, M.inner(x, new_mgrad, delta) / M.inner(x, prev_mgrad, prev_mgrad));
            case 'H-S'
	    	        delta = M.lincomb(x, 1, new_mgrad, -1, prev_mgrad);
                beta = max(0, -M.inner(x, new_mgrad, delta) / M.inner(x, prev_conj, delta));
            case 'my'
                delta = M.lincomb(x, 1, new_mgrad, -norm_r/norm_prev, prev_mgrad);
                beta = max(0, M.inner(x, new_mgrad, delta) / M.inner(x, prev_mgrad, prev_mgrad)) / 2;
            otherwise
                error('Unknown options.beta_type. Should be F-R, P-R, or H-S.');
        end
        
        prev_conj = M.lincomb(x, -1, new_mgrad, beta, prev_conj);
        prev_mgrad = new_mgrad;
        norm_prev = norm_r;
        j = j + 1;
        
    end
    
    % Check why we stopped iterating
    if ~gradnorm_reached
        stop_str = 4; %sprintf(['Reached max number of conjugate gradient iterations ' ...
              % '(options.maxinner = %d)'], options.maxinner);
        j = j - 1;
    end
    
    % Return the point we ended on
    eta = M.tangent(x, eta);
    if options.verbosity >= 4
        fprintf('\n');
    end
end

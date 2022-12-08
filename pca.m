function  pca()
    
    clc; 
    close all; 
    clear
    
    addpath(pwd);

    cd data/;
    addpath(genpath(pwd));
    cd ..;
    
    cd proposed/;
    addpath(genpath(pwd));
    cd ..;
    
    cd tools/;
    addpath(genpath(pwd));
    cd ..;
    
    cd benchmarks/;
    addpath(genpath(pwd));
    cd ..;
    
    cd manopt-4/;
    addpath(genpath(pwd));
    cd ..;
    
    cd manopt_mod_solvers/;
    addpath(genpath(pwd));
    cd ..;

    pkg load statistics
    
    %% Select case
    case_num = 2;

    %% Generate data
    fprintf('generating data ... ');

    if case_num == 1
        N = 500000; d = 1000; r = 5; 
        x_sample = randn(d, N);
        x_sample = diag(exprnd(2, d , 1))*x_sample;    
        x_sample = x_sample - repmat(mean(x_sample,2),1,size(x_sample,2));
        sigma_init = 100;
    elseif case_num == 2
        N = 60000; d = 784; r =10;
        x_sample = double(read_mnist_images('./data/train-images.idx3-ubyte'));
        x_sample = x_sample - repmat(mean(x_sample,2),1,size(x_sample,2));
        x_sample = x_sample / 255;
        sigma_init = 1;
    elseif case_num == 3
        load('./data/covtype.scale.mat');
        x_sample = x_sample';
        N = 581012; d = 54; r = 10;
        sigma_init = 1;
    end
    
    
    %cond(x_sample)
    fprintf('done.\n');
    
    %% Define parameters
    maxepoch = 100;
    tolgradnorm = 1e-8;
    
    % Iput data as cell
    data.x = mat2cell(x_sample, d, ones(N, 1)); %     
    
    %{
    %% Obtain solution
    % coeff = pca(x_sample');
    % x_star = coeff(:,1:r);
    xx = x_sample * x_sample';
    [U,S,V] = svd(xx, 'econ');
    x_star = U(:,1:r);
    f_sol = -0.5/N*norm(x_star'*x_sample, 'fro')^2;
    fprintf("f_sol %f\n", f_sol);
    %}
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
    problem.data = data;
    
    %% Define problem definitions
    
    function f = cost(U)
        f = -0.5*sum(sum((x_sample'*U).^2)); % norm(x_sample'*U, 'fro')^2;
        f = f/N;
    end
    problem.cost = @(U) cost(U);
    
    
    function g = egrad(U)
        g = - x_sample*(x_sample'*U);        
        g = g/N;
    end
    problem.egrad = @(U) egrad(U);
    
    
    function g = partialegrad(U, indices)
        len = length(indices);
        x_sample_batchsize = x_sample(:,indices);        
        g = - x_sample_batchsize*(x_sample_batchsize'*U);
        g = g/len;
    end        
    problem.partialegrad = @(U, indices) partialegrad(U, indices);
    
    
    function gdot = ehess(U, Udot)
        gdot = - x_sample*(x_sample'*Udot);
        gdot = gdot/N;
    end 
    problem.ehess = @(U, Udot) ehess(U, Udot);
    
    
    function gdot = partialehess(U, Udot, indices, square_hess_diag)
        len = length(indices);

        x_sub_sample = x_sample(:, indices);
        gdot = - x_sub_sample * (x_sub_sample' * Udot);

        gdot = gdot/len;               
    end
    problem.partialehess = @(U, Udot, indices, square_hess_diag) partialehess(U, Udot, indices, square_hess_diag);
    

    %% Run algorithms    
    
    % Initialize
    Uinit = problem.M.rand();
    
    
    fprintf('RSGD\n');
    clear options;
    options.maxiter = 10000;
    options.tolgradnorm = tolgradnorm;
    options.batchsize = floor(N/100); 
    [~, infos_sgd, ~, best_x] = stochasticgradient(problem, Uinit, options);
    

    fprintf('RSD\n');
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_sd, ~, best_x] = steepestdescent_mod(problem, Uinit, options); 
    

    fprintf('RCG\n');
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_cg, ~, best_x] = conjugategradient_mod(problem, Uinit, options); 
    

    fprintf('RLBFGS\n');
    % Run RLBFGS
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_lbfgs, best_x] = lbfgs_mod(problem, Uinit, options); 
    

    fprintf('RSVRG\n');
    clear options;    
    inner_repeat = 5;
    options.verbosity = 1;
    options.batchsize = floor(N/100);
    options.update_type = 'svrg';
    options.stepsize = 0.01;
    options.stepsize_type = 'fix';
    options.stepsize_lambda = 0;
    options.tolgradnorm = tolgradnorm; 
    options.boost = 0;
    options.svrg_type = 2; % effective only for R-SVRG variants
    options.maxinneriter = inner_repeat * N;
    options.transport = 'ret_vector';
    options.maxepoch = 30%floor(maxepoch / (1 + inner_repeat)) * 2;
    [~, ~, infos_svrg, ~, best_x] = Riemannian_svrg(problem, Uinit, options);  
    

    fprintf('RSRG\n');
    clear options;
    inner_repeat = 5;
    srg_varpi = 0.5;
    options.verbosity = 1;
    options.batchsize = floor(N/100);
    options.maxepoch = 30;%int32(maxepoch / (1 + inner_repeat) * 2);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 0.01;       
    options.gamma = srg_varpi;
    options.transport = 'ret_vector_locking';   
    options.store_innerinfo = false; 
    options.maxinneriter = inner_repeat*N;    
    [~, ~, infos_srg_plus, ~, best_x] = Riemannian_srg(problem, Uinit, options);
    

    fprintf('RTR\n');
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;     
    options.samp_hess_scheme = 'full';
    options.samp_grad_scheme = 'full';
    options.useExp = true;    
    [~, ~, infos_tr, ~, best_x] = subsampled_rtr(problem, Uinit, options); 
    

    fprintf('Inexct RTR\n');
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;     
    options.samp_hess_scheme = 'fix';
    %options.samp_hess_init_size = floor(N/100);
    options.samp_grad_scheme = 'fix';
    options.samp_grad_init_size = floor(N/10);
    options.useExp = true;    
    [~, ~, infos_tr, ~, best_x] = subsampled_rtr(problem, Uinit, options); 
    

    fprintf('Inexct Sub-RN-CR\n');
    clear options;
    options.maxiter = 100;
    options.tolgradnorm = tolgradnorm;
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100);
    options.samp_grad_scheme = 'full';
    %options.samp_grad_init_size = floor(N/10);
    options.maxinner = 5;  % P1 20  MNIST 5-10 covtype 5-10
    options.useExp = true;
    options.tol_newton = 1e-8; % 1e-8 ~ 1e-2
    options.kappa = 0.15;
    
    %sigma_init = mean(abs(x_sample(:)))^2 * sqrt(problem.M.dim() * d / std(x_sample(:)))
    [~, ~, infos_isrncr, ~, best_x] = isrncr_lanczos(problem, Uinit, options, sigma_init); 
    [~, ~, infos_isrncr, ~, best_x] = isrncr_tCG(problem, Uinit, options, sigma_init); 
    
end


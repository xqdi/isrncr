function mc()

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
    
    cd manopt/;
    addpath(genpath(pwd));
    cd ..;
    
    cd manopt_mod_solvers/;
    addpath(genpath(pwd));
    cd ..;

    pkg load statistics
    
    %% Select case
    case_num = 3;   % small instance
    
    if case_num == 1
        N = 100000; d = 100; r = 5;
        condition_number = 5;
        sigma_init = 10;
    elseif case_num == 2
        N = 100000; d = 100; r = 5;
        condition_number = 20;
        sigma_init = 10;
    elseif case_num == 3
        N = 30000; d = 100; r = 5;
        condition_number = 20;        
        sigma_init = 5;
    elseif case_num == 4
        N = 24983; d = 100; r = 5;
        condition_number = 0;
        sigma_init = 200;
    elseif case_num == 5      
        N = -1; d = -1; r = 20; 
        condition_number = 5; 
        sigma_init = 8000000;
    end 
    

    %% Define parameters
    maxepoch = 100;
    tolgradnorm = 1e-8;    
    over_sampling = 4;
    noiseFac = 1e-10;
    %sqn_mem_size = 0;
    
    NumEntries_train = over_sampling*r*(N + d -r);      
    NumEntries_test = over_sampling*r*(N + d -r);      

    %% Generate data
    fprintf('generating data ... \n');    
    if case_num == 4
        load('./data/jester_data.mat') 
    elseif case_num == 5
        img = imread('image_recovery.jpg');
        img = double(img(:,:,1));
        [d, N] = size(img);
        img = img(:);

        rand("seed",1313)

        perm = randperm(d*N);
        B = logical(zeros(d*N, 1));
        B(perm(1:int32(0.6*d*N))) = true;

        A_train = img(B);
        A_test = img(~B);

        NumEntries_train = length(A_train);
        NumEntries_test = length(A_test);

        indices = find(B);
        S = A_train;
        noise = noiseFac*max(S)*randn(size(S));
        S = S + noise;
        
        [I, J] = ind2sub([d, N], indices);
        [J, indI] = sort(J,'ascend');
        I = I(indI);
        I = I(:);
        J = J(:);

        values = sparse(I, J, S, d, N);
        indicator = sparse(I, J, 1, d, N);

        % Creat the cells
        samples(N).colnumber = []; % Preallocate memory.
        for k = 1 : N
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator(:, k)); % find known row indices
            values_col = values(idx, k); % the non-zero entries of the column

            samples(k).indicator = idx;
            samples(k).values = values_col;
            samples(k).colnumber = k;
        end 

        B_test = ~B;
        indices_test = find(B_test);
        S_test = A_test;

        [I_test, J_test] = ind2sub([d, N], indices_test);
        [J_test, indI] = sort(J_test,'ascend');
        I_test = I_test(indI);
        I_test = I_test(:);
        J_test = J_test(:);

        values_test = sparse(I_test, J_test, S_test, d, N);
        indicator_test = sparse(I_test, J_test, 1, d, N);

        samples_test(N).colnumber = [];
        for k = 1 : N
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator_test(:, k)); % find known row indices
            values_col = values_test(idx, k); % the non-zero entries of the column

            samples_test(k).indicator = idx;
            samples_test(k).values = values_col;
            samples_test(k).colnumber = k;
        end

        % for grouse
        data_ls.rows = I;
        data_ls.cols = J;
        data_ls.entries = S;
        data_ls.nentries = length(data_ls.entries);

        data_test.rows = I_test;
        data_test.cols = J_test;
        data_test.entries = S_test;
        data_test.nentries = length(data_test.entries);
    else
        % Generate well-conditioned or ill-conditioned data
        M = over_sampling*r*(N + d -r); % total entries

        % The left and right factors which make up our true data matrix Y.
        YL = randn(d, r);
        YR = randn(N, r);

        % Condition number
        if condition_number > 0
            %YLQ = orth(YL);
            %YRQ = orth(YR);
            [YRQ,S,V] = svd(YR,'econ'); %S is always square.
            s = diag(S);
            tol = max(size(YR)) * eps(max(s));
            r = sum(s > tol);
            YRQ(:, r+1:end) = [];
            
            [YLQ,S,V] = svd(YL,'econ'); %S is always square.
            s = diag(S);
            tol = max(size(YL)) * eps(max(s));
            r = sum(s > tol);
            YLQ(:, r+1:end) = [];

            s1 = 1000;
            %     step = 1000; S0 = diag([s1:step:s1+(r-1)*step]*1); % Linear decay
            S0 = s1*diag(logspace(-log10(condition_number),0,r)); % Exponential decay

            YL = YLQ*S0;
            YR = YRQ;

            fprintf('Creating a matrix with singular values...\n')
            for kk = 1: length(diag(S0))
                fprintf('%s \n', num2str(S0(kk, kk), '%10.5e') );
            end
            singular_vals = svd(YL'*YL);
            condition_number = sqrt(max(singular_vals)/min(singular_vals));
            fprintf('Condition number is %f \n', condition_number);
        end

        % Select a random set of M entries of Y = YL YR'.
        idx = unique(ceil(N*d*rand(1,(10*M))));
        idx = idx(randperm(length(idx)));

        [I, J] = ind2sub([d, N],idx(1:M));
        [J, inxs] = sort(J); I=I(inxs)';

        % Values of Y at the locations indexed by I and J.
        S = sum(YL(I,:).*YR(J,:), 2);
        S_noiseFree = S;

        % Add noise.
        noise = noiseFac*max(S)*randn(size(S));
        S = S + noise;

        values = sparse(I, J, S, d, N);
        indicator = sparse(I, J, 1, d, N);


        % Creat the cells
        samples(N).colnumber = []; % Preallocate memory.
        for k = 1 : N
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator(:, k)); % find known row indices
            values_col = values(idx, k); % the non-zero entries of the column

            samples(k).indicator = idx;
            samples(k).values = values_col;
            samples(k).colnumber = k;
        end 

        % Test data
        idx_test = unique(ceil(N*d*rand(1,(10*M))));
        idx_test = idx_test(randperm(length(idx_test)));
        [I_test, J_test] = ind2sub([d, N],idx_test(1:M));
        [J_test, inxs] = sort(J_test); I_test=I_test(inxs)';

        % Values of Y at the locations indexed by I and J.
        S_test = sum(YL(I_test,:).*YR(J_test,:), 2);
        values_test = sparse(I_test, J_test, S_test, d, N);
        indicator_test = sparse(I_test, J_test, 1, d, N);

        samples_test(N).colnumber = [];
        for k = 1 : N
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator_test(:, k)); % find known row indices
            values_col = values_test(idx, k); % the non-zero entries of the column

            samples_test(k).indicator = idx;
            samples_test(k).values = values_col;
            samples_test(k).colnumber = k;
        end
        
        % for grouse
        data_ls.rows = I;
        data_ls.cols = J';
        data_ls.entries = S;
        data_ls.nentries = length(data_ls.entries);

        data_test.rows = I_test;
        data_test.cols = J_test';
        data_test.entries = S_test;
        data_test.nentries = length(data_test.entries);
    end
    cn = floor(condition_number); 
    fprintf('done.\n');    
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
   
    % Define problem definitions
    
    function f = cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/N;
    end
    problem.cost = @(U) cost(U);
    
    
    function g = egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
    end
    problem.egrad = @(U) egrad(U);
    
    
    function gdot = ehess(U, Udot) 
        [W, Wdot] = mylsqfitdot(U, Udot, samples);
        gdot = (indicator.*(U*W') - values)*Wdot +  (indicator.*(U*Wdot' + Udot*W'))*W;
        gdot = gdot/N;
    end
    problem.ehess = @(U, Udot) ehess(U, Udot);
    
    
    function g = partialegrad(U, idx_batch)
        g = zeros(d, r);
        m_batchsize = length(idx_batch);
        for ii = 1 : m_batchsize
            colnum = idx_batch(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
    end
    problem.partialegrad = @(U, idx_batch) partialegrad(U, idx_batch);
   
    
	  function gdot = partialehess(U, Udot, idx_batch, square_hess_diag) % We need Udot and the idx_batch.
      if 0
            gdot = zeros(d, r);
            m_batchsize = length(idx_batch);
            for ii = 1 : m_batchsize
                colnum = idx_batch(ii);
                [w, wdot] = mylsqfitdot(U, Udot, samples(colnum)); % we compute both w and wdot. This has some redundant computations because w is already computed in partialegrad
                indicator_vec = indicator(:, colnum);
                values_vec = values(:, colnum);
                gdot = gdot + (indicator_vec.*(U*w') - values_vec)*wdot ... % we need both w and wdot. w is obtained from partialegrad, but we compute here again.
                    + (indicator_vec.*(U*wdot' + Udot*w'))*w;
            end
            gdot = gdot/m_batchsize;

        else
            m_batchsize = length(idx_batch);

            sub_samples = samples(idx_batch);
            sub_values = values(:, idx_batch);
            sub_indicator = indicator(:, idx_batch);

            [W, Wdot] = mylsqfitdot(U, Udot, sub_samples);
            gdot = (sub_indicator.*(U*W') - sub_values)*Wdot +  (sub_indicator.*(U*Wdot' + Udot*W'))*W;
            gdot = gdot/m_batchsize;        
        end
    end
    problem.partialehess = @(U, Udot, idx_batch, square_hess_diag) partialehess(U, Udot, idx_batch, square_hess_diag);
    
    function stats = mc_mystatsfun(problem, U, stats)
        W = mylsqfit(U, samples_test);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/N;
        stats.cost_test = f_test;
    end


    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            
            % Solve a simple least squares problem to populate W.
            %OmegaUtUOmega = U_Omega'*U_Omega;
            OmegaUtUOmega = U_Omega'*U_Omega + 1e-10*eye(r);            
            W(ii,:) = (OmegaUtUOmega\(U_Omega'*values_Omega))';

        end
    end

    
    function [W, Wdot] = mylsqfitdot(U, Udot, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        Wdot = zeros(size(W));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            Udot_Omega = Udot(IDX,:);
            
            % Solve a simple least squares problem to populate W and Wdot
            OmegaUtUOmega = U_Omega'*U_Omega;
            W(ii,:) = (OmegaUtUOmega\(U_Omega'*values_Omega))';

            UOmegaW = U_Omega*(W(ii,:))';
            UdotOmegaW = Udot_Omega*(W(ii,:))';

            Wdot(ii,:) = (OmegaUtUOmega\(Udot_Omega'*values_Omega - U_Omega'* UdotOmegaW  - Udot_Omega'* UOmegaW ))';
        end
    end   

    
    % Consistency checks
%     checkgradient(problem)
%     pause;
%     
%     
%     checkhessian(problem);
%     pause;
    
    
%% Run algorithms    
    % Initialize
    x_init = problem.M.rand();

    %{
    fprintf('RSGD\n');
    clear options;
    options.maxiter = maxepoch*200;
    options.tolgradnorm = tolgradnorm;
    options.batchsize = floor(N/100);
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);                
    [~, infos_sgd, ~, best_x] = stochasticgradient(problem, x_init, options);
    
    
    fprintf('RSD\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);                
    [~, ~, infos_sd, ~, best_x] = steepestdescent_mod(problem, x_init, options);  
    

    fprintf('RCG\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);                
    [~, ~, infos_cg, ~, best_x] = conjugategradient_mod(problem, x_init, options);  
    

    fprintf('RLBFGS\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm; 
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);     
    [~, ~, infos_lbfgs, best_x] = lbfgs_mod(problem, x_init, options); 
    

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
    options.maxepoch = floor(maxepoch / (1 + inner_repeat)) * 2;
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);       
    [~, ~, infos_svrg, ~, best_x] = Riemannian_svrg(problem, x_init, options);  
    

    fprintf('RSRG\n');
    clear options;
    inner_repeat = 5;
    srg_varpi = 0.5;
    options.verbosity = 1;
    options.batchsize = floor(N/100);
    options.maxepoch = int32(maxepoch / (1 + inner_repeat) * 2);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 0.01;       
    options.gamma = srg_varpi;
    options.transport = 'ret_vector_locking';   
    options.store_innerinfo = false; 
    options.maxinneriter = inner_repeat*N;    
    [~, ~, infos_srg_plus, ~, best_x] = Riemannian_srg(problem, x_init, options);  
    

    fprintf('RTR\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);                
    [~, ~, infos_tr, ~, best_x] = subsampled_rtr(problem, x_init, options);  
    

    fprintf('RTRMC\n');
    clear options;
    options.maxiterations = maxepoch;
    [infos_rtrmc, best_x, W] = rtrmc_rapper(x_init, d, N, r, data_ls, data_test, options);  
    

    fprintf('Inexact RTR\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);     
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100);
    options.samp_scheme = 'uniform';
    options.samp_grad_scheme = 'fix';
    options.samp_grad_init_size = floor(N/10);     
    [~, ~, infos_irtr_fix, ~, best_x] = subsampled_rtr(problem, x_init, options);  
    %}

    fprintf('Inexact Sub-RN-CR\n');
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm; 
    options.statsfun = @(problem, U, stats) mc_mystatsfun(problem, U, stats);         
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100); 
    options.samp_scheme = 'uniform';
    options.samp_grad_scheme = 'full';
    %options.samp_grad_init_size = floor(N/10);       
    options.maxinner = 20;  % M1-M3 20-50 Jester 3-5 
    options.tol_newton = 1e-8; 
    options.kappa = 0.15;
    
    % x_sample = YR * YL';
    %sigma_init = mean(abs(x_sample(:)))^2 * sqrt(problem.M.dim() * d / std(x_sample(:)))
    %[~, ~, infos_isrncr, ~, best_x] = isrncr_lanczos(problem, x_init, options, sigma_init); 
    [~, ~, infos_isrncr, ~, best_x] = isrncr_tCG(problem, x_init, options, sigma_init); 
 
  
end


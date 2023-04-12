function output = Core_Noise_NTGNN(t, iter_data, R, D, Noise)

    gamma = 5;
    lambda = 10;
    param_p = 0.04;
    param_sigma = 1.2;

    [len ,wid] = size(D);
    
    H = R;
    J = -D';
    d = -ones(wid,1);
    
    x = iter_data(1:len);
    inte = iter_data(len+1:end-1);
    A = H;
    z = x;
    
    [R_len, R_wid] = size(d);
    p_alapha = zeros(R_wid, 1);

    for i = 1:wid
        p_alapha = p_alapha + exp(-param_sigma*(d(i,:)-J(i,:)*x))*J(i,:)';
    end
    p_alapha = param_p * param_sigma * p_alapha;
    g = -p_alapha;
    
    dyn = zeros(R_wid, 1);
    for i = 1:wid
        dyn = dyn + exp(-param_sigma*(d(i,:)-J(i,:)*x))*(J(i,:)'*J(i,:));
    end
    dyn = param_p * (param_sigma^2) * dyn;
    
    M = R + dyn;
    
    Err = A*z-g;
    norm_info = norm(Err);
    
    output = A'*(-gamma*(A*z-g)-lambda*inte) + Noise;
    
    output = [output;Err; norm_info];
    t
end
    
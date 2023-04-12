function output = Core_Noise_NBCZNN(t, iter_data, R, D, Noise)
    
    [len ,wid] = size(D);

    gamma = 5;
    lambda = 10;
    param_p = 0.02;
    param_sigma = 2;

    x = iter_data(1:len);
    inte = iter_data(len+1:end-1);
    
    H = R;
    J = -D';
    d = -ones(wid,1);
    
    A = H;
    z = x;
    
    [~, R_wid] = size(d);
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
    
    dotx = pinv(M)*(-gamma*Err-lambda*inte + Noise);
    output = [dotx;Err; norm_info];
    t
end
    
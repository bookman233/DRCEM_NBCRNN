function output = AF_Adaptive(X)
    bound = 100;
    output = zeros(length(X), 1);
    for index = 1:length(X)
        if X(index) > bound
            output(index) = bound;
        end
        if X(index) < -bound
            output(index) = -bound;
        end
        if X(index) <= bound && X(index) >= -bound
            output(index) = (((abs(X(index))+1).^5)+5) .* X(index);
        end
    end
end
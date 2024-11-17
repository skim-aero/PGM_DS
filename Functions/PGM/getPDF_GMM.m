function pdf_sum = getPDF_GMM(X, weights, means, covariances)
    k = length(weights);    % number of components
    n = size(X, 2);         % number of data points
    d = size(X, 1);         % dimension of the data
    pdf_vals = zeros(1, n);

    for j = 1:n
        x = X(:, j);
        for i = 1:k
            mu = means(i,:)';
            Sigma = covariances(:,:,i);
            coef = 1/((2*pi)^(d/2)*sqrt(det(Sigma)));
            exponent = -0.5*(x-mu)'/Sigma*(x-mu);
            pdf_vals(j) = pdf_vals(j)+weights(i)*coef*exp(exponent);
        end
    end

    pdf_sum = sum(pdf_vals);
end


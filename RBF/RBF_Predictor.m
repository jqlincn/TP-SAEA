function Surrogate_Obj=RBF_Predictor(X, Surrogate_model, M)

% This function is written by Jianqing Lin

    N  = size(X,1);
    Surrogate_Obj = zeros(N,M);
    for j = 1:N
        for i=1:M
            Surrogate_Obj(j,i) = RBFInterp(X(j,:), Surrogate_model{i});
        end
    end
end

function y = RBFInterp(x, para)
    ax = para.nodes;
    nx = size(x, 1);
    np = size(ax, 1);    % np: the size of data set

    xmin = para.xmin;
    xmax = para.xmax;
    ymin = para.ymin;
    ymax = para.ymax;

    % normalization
    x = 2./(repmat(xmax - xmin, nx, 1)) .* (x - repmat(xmin, nx, 1)) - 1;

    r = dist(x, ax');
    switch para.kernel
        case 'gaussian'
            Phi = radbas(sqrt(-log(.5))*r);
        case 'cubic'
            Phi = r.^3;
    end

    y = Phi * para.alpha + [ones(nx, 1), x] * para.beta;
    % renormalization
    y = repmat(ymax - ymin, nx, 1)./2 .* (y + 1) + repmat(ymin, nx, 1);

end

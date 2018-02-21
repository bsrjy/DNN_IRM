function [ pri_val ] = princ( phase )
%Returns the principal value of phase
% Input:
% phase: unwraped phase, could be any real number
% Output:
% pri_val: wrapped phase in (-pi, pi)

    pri_val = mod(phase,2*pi) - 2*pi*(mod(phase,2*pi)>pi);

end

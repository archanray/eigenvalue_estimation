clear all;
clc;
n=1000;
s=100;

A = 2*randi(2,n,n) - 3;
A = A - tril(A,-1) + triu(A,1)';

alpha=1000;
beta=-500;

p = sort(randperm(n,s));

[V,D]=eigs(A,n);
D_root=sqrtm(D);
B=V*D_root;
B_sample=B(p,:);
R=(alpha*(n/s)*(transpose(B_sample)*B_sample))+(beta*(D));
eigvals=eigs(R,n);

if (abs(eigvals-real(eigvals))<10^(-8))
    sprintf('real')
else
    sprintf('complex')
end
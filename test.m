n = 2000;
A = rand(n,n);
%sparse binary matrix
A = A > .99;
%symmetrize
A = triu(A) + triu(A)';
%hard case?
%A = zeros(n,n);
%A(1:600,1:600) = -1;
%top 40 magnitude eigenvalues. Safe to assume the smallest lies in this set.
lmin = eigs(A,40);
lmin = min(lmin);
sizes = 50:50:1000;
err = zeros(size(sizes));
for i = 1:length(sizes)
  for t = 1:10
    ind = randperm(n,sizes(i));
    lminS = eigs(A(ind,ind),40);
    err(i) = err(i) + abs(n/sizes(i)*min(lminS)-lmin);
  end
end
err = err/10;
plot(log(sizes),log(err))
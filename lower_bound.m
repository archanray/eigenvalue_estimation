n = 5000;
%n=100;
%hard case?
c=0.5; 
A = zeros(n,n);
%A(1:600,1:600) = -1;
%top 40 magnitude eigenvalues. Safe to assume the smallest lies in this set.
%lmin = eigs(A,40);
ind=randperm(n,n*c);
A(ind,ind)=-1;
lmin=-(c*n);
sizes = 50:50:2000;
err = zeros(size(sizes));
for i = 1:length(sizes)
  for t = 1:40
    ind = randperm(n,sizes(i));
%     c_avg=
%     lminS=-c_avg*n;
%    lminS = eigs(A(ind,ind),40);
    lminS = eigs(A(ind,ind),1,'smallestreal');
    err(i) = err(i) + abs(n/sizes(i)*min(lminS)-lmin);
  end
end
err = err/40;
plot(log(sizes),log(err))
clf;clc;clear;

n = 5000;
A = zeros(n,n);
A(1:2500,1:2500) = 1;
e1 = eigs(A,1);
sample_sizes = 50:10:1000;
errors = zeros(size(sample_sizes));
for i=1:length(sample_sizes)
  sample_sizes(i)
  for t = 1:100
    r = randperm(n); r = r(1:sample_sizes(i));
    etilde = eigs(A(r,r),1)*n/sample_sizes(i);
    etilde;
    errors(i) = errors(i) + abs(e1-etilde)/n;
  end
  errors(i) = errors(i)/100;
end
plot(log(sample_sizes/n),log(errors));
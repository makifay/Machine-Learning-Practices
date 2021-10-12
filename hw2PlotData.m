% BLG527E, HW2, Q1, Examining the dataset 
% Akif Aydogmus, ID: 702121013
% run on Ubuntu 14.04.5 LTS, GNU Octave, version 3.8.1

d = importdata("hw2.mat");

CountC0=0;
CountC1=0;

for i = 1:2000

  if(d(i,3)==0)
    CountC0++;
  elseif(d(i,3)==1)
    CountC1++;
  end

end

display("Observations for Class Label:0")
CountC0
display("Observations for Class Label:1")
CountC1

% ===========================
C0= d(1:1000,1:2);
C1= d(1001:2000,1:2);

Clmn1 =0;
Clmn2 =0;
for i = 1:1000
   Clmn1 += C0 (i,1);
   Clmn2 += C0 (i,2);
end

display("Mean Vector for Class Label:0")
Clmn1 /1000
Clmn2 /1000
% Alternative command : mean(C0)

Clmn1 =0;
Clmn2 =0;
for i = 1:1000
   Clmn1 += C1 (i,1);
   Clmn2 += C1 (i,2);
end

display("Mean Vector for Class Label:1")
Clmn1 /1000
Clmn2 /1000
% Alternative command : mean(C1)

% ============================
n=size(C0,1);
a = C0 - ones (n,n)*C0*(1/n); 
CM0 = a'*a * (1/n);
display("Covariance Matrix for Class Label:0")
CM0

n=size(C1,1);
a = C1 - ones (n,n)*C1*(1/n); 
CM1 = a'*a * (1/n);
display("Covariance Matrix for Class Label:1")
CM1

plot(C0(:,1), C0(:,2), 'b.', C1(:,1), C1(:,2), 'g.' );
legend ("Class Label=0", "Class Label=1");





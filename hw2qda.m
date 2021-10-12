% BLG527E, HW2, Q1, Quadratic Discriminant Analysis
% Akif Aydogmus, ID: 702121013
% run on Ubuntu 14.04.5 LTS, GNU Octave, version 3.8.1


function ret = hw2qda()

display('')
display('#####  Quadratic Discriminant Analysis  #####')

data = importdata("hw2.mat"); % read input data
rowCount=size(data,1);
clmnCount=size(data,2);
falsePrediction_Sum = 0 ; 
falsePrediction_Sum2 = 0 ; 

% >> Extracting feature datas and labels
X_data = data(:,1:clmnCount-1);
Y_data = data(:,clmnCount);

% >> Dividing data according to the their labels
% Label=0 variables : X_L0(feature values), mu_L0(mean), sigma_L0(covariance matrix)
% Label=1 variables : X_L1(feature values), mu_L1(mean), sigma_L1(covariance matrix)
X_L0_data= X_data(find(Y_data==0),:);
X_L1_data= X_data(find(Y_data==1),:);

% --------------------------------
for  i = 1:10
% >> Seperating to folds
  Xv_L0 = X_L0_data( (i-1)*100+1:i*100, :);
  Xt_L0 = X_L0_data;
  Xt_L0( (i-1)*100+1:i*100, :) = [];

  Xv_L1 = X_L1_data( (i-1)*100+1:i*100, :);
  Xt_L1 = X_L1_data;
  Xt_L1( (i-1)*100+1:i*100, :) = [];

  %display( ' ========= ') 
  %display( ' round ') 
  %i

% >> Training data is used for training and validation
  % display( ' -- Training data results: ') ;
  false_prediction = qda_main(Xt_L0, Xt_L1, Xt_L0, Xt_L1, 0 );
  falsePrediction_Sum = falsePrediction_Sum + false_prediction ; 


% >> Training data is used for training and validation data is used for validation
  % display( ' -- Validation data results: ') ;
  false_prediction2 = qda_main(Xt_L0, Xt_L1, Xv_L0, Xv_L1, 0 );
  falsePrediction_Sum2 = falsePrediction_Sum2 + false_prediction2 ; 

end % end of for
% --------------------------------


display('================== RESULTS =====================');
% Sum of the false predictions and calcuate all error rate for Traning Data
disp(['>> TRANING Data Classification Results:  ' ]);
disp(['False Predictions : ' num2str(falsePrediction_Sum)]);
Error_rate = (falsePrediction_Sum)*100 / ( 20*size(Xt_L0, 1) ) ; 
disp(['Classification Error : %' num2str(Error_rate)]);

display('------------------------------------------------');

% Sum of the false predictions and calcuate all error rate for Validation Data
disp(['>> VALIDATION Data Classification Results:  ' ]);
disp(['Total False Predictions: ' num2str(falsePrediction_Sum2)]);
Error_rate = (falsePrediction_Sum2)*100 / ( 20*size(Xv_L0, 1) ) ; 
disp(['Classification Error : %' num2str(Error_rate)]);
display('================================================');

% -----------------------------------------
% Sample Plotting of Decision Boundary
i = 1; 
 Xv_L0 = X_L0_data( (i-1)*100+1:i*100, :);
 Xt_L0 = X_L0_data;
 Xt_L0( (i-1)*100+1:i*100, :) = [];

 Xv_L1 = X_L1_data( (i-1)*100+1:i*100, :);
 Xt_L1 = X_L1_data;
 Xt_L1( (i-1)*100+1:i*100, :) = [];

display('=== Results for plotting of Validation data decision boundary:');
 qda_main(Xt_L0, Xt_L1, Xv_L0, Xv_L1, 1 );  % plot with validation data 
display('=== Results for plotting of Training data decision boundary:');
 qda_main(Xt_L0, Xt_L1, Xt_L0, Xt_L1, 1 ); % plot with training data 

% -----------------------------------------


end % end of hw2qda



%#######################################################################################
% Definitions: Xt_L0: Train data of Label 0,  Xt_L1: Train data of Label 1
%              Xv_L0: Validation data of Label 0,  Xv_L1: Validation data of Label 1
% If plotting of decision boundary is desired, last argument of lda_main function should be 1

function false_pred = qda_main(Xt_L0, Xt_L1, Xv_L0, Xv_L1, plotSelection )

  falsePrediction_L1_Sum = 0;
  falsePrediction_L0_Sum = 0;
  Xv_data = [ Xv_L0 , Xv_L1 ];

  mu_L0=mean(Xt_L0);
  mu_L1=mean(Xt_L1);
  p_Y = 0.5 ; % label counts are equal

% >> Computing Covariance Matrices
  sigma_L0=(Xt_L0-ones(size(Xt_L0,1),1)*mu_L0)'*(Xt_L0-ones(size(Xt_L0,1),1)*mu_L0)/size(Xt_L0,1);
  sigma_L1=(Xt_L1-ones(size(Xt_L1,1),1)*mu_L1)'*(Xt_L1-ones(size(Xt_L1,1),1)*mu_L1)/size(Xt_L1,1);

% >> Finding of the A, b, w parameters 
  A=( inv(sigma_L0)-inv(sigma_L1) ) / 2;
  b=(mu_L0*inv(sigma_L0)*mu_L0' - mu_L1*inv(sigma_L1)*mu_L1')/2 +log(sqrt(det(sigma_L0)/det(sigma_L1))) + log(p_Y/(1-p_Y)) ;
  w=( inv(sigma_L1)*mu_L1' - inv(sigma_L0 )*mu_L0');

% >> Classification Errors
  prediction_label = 1./(1+exp(-qda_func(A,Xv_data(:,1),Xv_data(:,2)) - w(1)*Xv_data(:,1)-w(2)*Xv_data(:,2)-b));

  prediction_L0 = 1./(1+exp(- qda_func(A,Xv_L0(:,1),Xv_L0(:,2)) - w(1)*Xv_L0(:,1)-w(2)*Xv_L0(:,2)-b));
  prediction_L1 = 1./(1+exp(- qda_func(A,Xv_L1(:,1),Xv_L1(:,2)) - w(1)*Xv_L1(:,1)-w(2)*Xv_L1(:,2)-b));

  falsePrediction_L1 =  sum(prediction_L1 <  p_Y );
  falsePrediction_L0 =  sum(prediction_L0 >= p_Y );
  false_pred = falsePrediction_L0 + falsePrediction_L1 ;

%disp(['False Prediction of Label-0 : ' num2str(falsePrediction_L0) ' (over ' num2str(size(Xv_L0,1)) ' Label-0 data)'  ]);
%disp(['False Prediction of Label-1 : ' num2str(falsePrediction_L1) ' (over ' num2str(size(Xv_L1,1)) ' Label-1 data)' ]);

  % --- Plotting datas and decision boundary, plotting depends on selection ---
  if(plotSelection == 1)
	% Firstly creating data for contour, then calculate values via quadratic discriminant

	x_sample = [-6:.01:6];
	y_sample= [-6:.01:10];
	[x_mg ,y_mg] = meshgrid(x_sample, y_sample);

	q_qda = qda_func(A, x_mg, y_mg);
	quad_disc_val = 1./(1+exp(-q_qda-w(1)*x_mg-w(2)*y_mg-b));

	figure;
	%plot(Xv_L0(:,1), Xv_L0(:,2), 'b.', Xv_L1(:,1), Xv_L1(:,2), 'g.'); hold on,
	scatter(Xv_L0(:,1), Xv_L0(:,2), 'b','filled'); hold on,
	scatter(Xv_L1(:,1), Xv_L1(:,2), 'g','filled'); hold on,

	scatter(Xv_L1(prediction_L1 < p_Y,1),Xv_L1(prediction_L1 < p_Y, 2),'r','filled'); hold on,
	scatter(Xv_L0(prediction_L0 >= p_Y,1),Xv_L0(prediction_L0 >= p_Y, 2),'r','filled'); hold on,
       
	title( 'Decision Boundary of Quadratic Discriminant Classifier ' );legend('label=0', 'label=1','False Predictions');
	xlabel('x');
	ylabel('y');
	[Cntr,con_h] = contour(x_mg, y_mg, quad_disc_val, [p_Y, p_Y], 'm');
	
   end % End of Plot Selection

end % End of qda_main


%#######################################################################################

function retval = qda_func(A,X1,X2)

retval = A(1,1)*X1.^2+A(2,2)*X2.^2 + 2*A(1,2)*X1.*X2;

end






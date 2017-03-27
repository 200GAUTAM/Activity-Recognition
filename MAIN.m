train_features = dlmread('dataset/X_train.txt');
train_labels   = dlmread('dataset/y_train.txt');

num_of_fatures = size(train_features,1);
num_dimension  = size(train_features,2);
num_of_labels  = size(unique(train_labels),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%      PCA       %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%                %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%normalization of traing_set
printf('Normalizing dataset for pca\n')
[X_norm, mu, sigma] = normalize(train_features);
 
%Running pca on normalized dataset
printf('Running pca Algorithm on normalized dataset\n');

[U, S] = pca(X_norm);


Reduced = projectData(X_norm, U, 30);

%%display(size(Reduced));

printf('The dataset has been reduced to 30 features from 561\n');
printf('press Enter to coninue further algorithms\n');
pause;
printf('The end\n');


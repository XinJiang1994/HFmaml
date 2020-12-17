
This is the version for MobiHoc.

For some coding reasons, the HFmaml actually means the  method we proposed which is called ADMM-FedMeta algorithm.

Step 1:
  prepare the data with the following command:
  cd data
  python3 DataDivision.py
  

Step 2:
    Go back to the root folder of the project, then run the following command to perform the constrast experiments
    
    ./run_contrast_cifar10.sh
    ./run_contrast_Fmnist.sh
    ./run_contrast_cifar100.sh
    
    The results are saved in the folders of contrast_results100, contrast_results,contrast_results_cifar100_r100 respectively.
    
    
    
Step3:
    Run the experiments to test the impact of rho:
    
    ./run_rho  # This step perform test on cifar10
    ./run_rho_cifar100.sh # this step perfor test on cifar100
    
    The results are saved in the folders of  rho_test and rho_test100 respectly.
    
step4:
    Run the experiments to test the impact of theta_c and lambda:
    
    ./run_thetaC.sh

step5:
    Run the forget test.

     ./run_forget_test.sh     

The results are saved in the folders of theta_c_test_results 

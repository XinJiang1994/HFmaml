
This is the version for MobiHoc.

CIFAR-10: 用前五类的data先训练一个theta_0，保存最后的accuracy。然后做下面的事情：
1.	用后五类的data，把这个theta_0当作训练的初始值，分别用友军和敌军训练得到\theta_yj和\theta_dj，并保存最后的accuracy。
2.	同样地，用后五类的data，把这个theta_0当作训练的初始值，并令theta_c=theta_0，设置适当的lambda（可能需要多次尝试）用我们的算法训练得到\theta_us，并保存accuracy。
做完以上两步之后，我们用得到的theta_jy、theta_dj、theta_us分别在前五类的data上做一次adaptation，然后保存accuracy。



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
     
     The results are saved in the folders of theta_c_test_results 

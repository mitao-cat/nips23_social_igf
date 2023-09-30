# nips23_social_igf

> This is the implementation of our paper accepted by NeurIPS' 23:
>
> **Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach**
>
> - Jinqiu Jin, Haoxuan Li, Fuli Feng, Sihao Ding, Peng Wu, Xiangnan He

**To reproduce the results of our proposed method, there are two steps:**

- **Step 1: Pretrain a BPRMF model**

   To initialize all the baselines and ablations with a well-trained BPRMF model, you can run the following commands:

  ```shell
  python bprmf.py --dataset KuaiRec --batch_size 4096 --lr 1e-3 --reg 1e-5 --max_epoch 200 --patience 10
  python bprmf.py --dataset Epinions --batch_size 8192 --lr 1e-2 --reg 1e-5 --max_epoch 200 --patience 10
  ```

  > This step is **optional**, since we have provided the well-trained bprmf model in `./data/dataset/best_model.pth.tar`. 

- **Step 2: Run our proposed method**

  run `python main.py --dataset KuaiRec --batch_size 4096 --reg 1e-5 --max_epoch 50 --patience 5 ` **(for KuaiRec)** or `python main.py --dataset Epinions --batch_size 8192 --reg 1e-5 --max_epoch 50 --patience 5 ` **(for Epinions)** <u>***plus the following arguments:***</u>

  | --argument     | description                         | (optimal) values                             |
  | -------------- | ----------------------------------- | -------------------------------------------- |
  | `--pref_idx`   | the preference (solution) region    | `0`, `1`, `2`, `3`, `4`                      |
  | `--lr`         | lr for solving the subproblem       | see the parameter config page                |
  | `--initsol_lr` | lr for finding the initial solution | swee the parameter config page               |
  | `--thre`       | accuracy threhold $\xi$ in Eq.(11)  | **`0.34`** - KuaiRec, **`0.405`** - Epinions |
  | `--mode`       | trade-off between SP/NSP or EO/NEO  | `sp`,  `eo`                                  |
  
- **Below are hyper-parameters for reproducing the main results:**

  - **For KuaiRec dataset:** run `python main.py --dataset KuaiRec --batch_size 4096 --reg 1e-5 --max_epoch 50 --patience 5 ` plus the following arguments:

    ```shell
    --pref_idx 0 --lr 1e-4 --initsol_lr 3e-4 --thre 0.34 --mode sp
    --pref_idx 1 --lr 1e-4 --initsol_lr 1e-3 --thre 0.34 --mode sp
    --pref_idx 2 --lr 3e-3 --initsol_lr 3e-3 --thre 0.34 --mode sp
    --pref_idx 3 --lr 3e-4 --initsol_lr 3e-4 --thre 0.34 --mode sp
    --pref_idx 4 --lr 1e-3 --initsol_lr 3e-4 --thre 0.34 --mode sp
    
    --pref_idx 0 --lr 3e-5 --initsol_lr 1e-3 --thre 0.4 --mode eo
    --pref_idx 1 --lr 1e-4 --initsol_lr 3e-4 --thre 0.5 --mode eo
    --pref_idx 2 --lr 1e-4 --initsol_lr 3e-3 --thre 0.34 --mode eo
    --pref_idx 3 --lr 1e-4 --initsol_lr 3e-3 --thre 0.4 --mode eo
    --pref_idx 4 --lr 3e-4 --initsol_lr 3e-3 --thre 0.34 --mode eo
    ```
    
  - **For Epinions dataset:** run `python main.py --dataset Epinions --batch_size 8192 --reg 1e-5 --max_epoch 50 --patience 5 ` plus the following options:
  
    ```shell
    --pref_idx 0 --lr 3e-4 --initsol_lr 3e-3 --thre 0.6 --mode sp
    --pref_idx 1 --lr 1e-3 --initsol_lr 1e-2 --thre 0.405 --mode sp
    --pref_idx 2 --lr 1e-3 --initsol_lr 1e-2 --thre 0.405 --mode sp
    --pref_idx 3 --lr 1e-2 --initsol_lr 1e-2 --thre 0.405 --mode sp
    --pref_idx 4 --lr 3e-3 --initsol_lr 3e-2 --thre 0.5 --mode sp
    
    --pref_idx 0 --lr 3e-4 --initsol_lr 3e-2 --thre 0.405 --mode eo
    --pref_idx 1 --lr 1e-3 --initsol_lr 3e-2 --thre 0.405 --mode eo
    --pref_idx 2 --lr 1e-2 --initsol_lr 1e-2 --thre 0.405 --mode eo
    --pref_idx 3 --lr 3e-3 --initsol_lr 1e-2 --thre 0.405 --mode eo
    --pref_idx 4 --lr 1e-3 --initsol_lr 3e-2 --thre 0.405 --mode eo
    ```

- Besides, we also provide the code of other baselines and ablations. The following table shows the usage of these codes:

  | File                  | Method                                          | Usage (Example)                                              |
  | --------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
  | `regularization.py`   | SP(EO) Reg, NSP(NEO) Reg, SP&NSP(EO&NEO) Reg    | `python regularization.py --lr 1e-4 --reg1 2 --mode eo --dataset KuaiRec` |
  | `post_reconstruct.py` | SP(EO) Post, NSP(NEO) Post, SP&NSP(EO&NEO) Post | `python post_reconstruct.py --lr 3e-5 --reg1 0.1 --reg2 0.1 --mode sp_nsp --dataset KuaiRec` |
  | `moomtl.py`           | MOOMTL                                          | `python moomtl.py --lr 1e-3 --initsol_lr 1e-3 --dataset Epinions` |
  | `wo_con.py`           | w/o con                                         | `python wo_con.py --lr 3e-4 --mode sp --initsol_lr 1e-2 --pref_idx 0 --dataset Epinions` |
  | `single_obj.py`       | w/o NSP(NEO), w/o SP(EO)                        | `python single_obj.py --lr 3e-3 --mode nsp --thre 0.4 --dataset KuaiRec` |
  | `wo_region.py`        | w/o region                                      | `python wo_region.py --lr 1e-3 --thre 0.405 --mode eo --dataset Epinions` |

- We also provide the hyper-parameters for reproducing all results of various baselines and abalations [here](./data/hyper.xlsx).

- **Reference:** If you want to use our codes in your research, please cite:

   ```
   @inproceedings{social_igf,   
         author    = {Jinqiu Jin and
                      Haoxuan Li and 
                      Fuli Feng and 
                      Sihao Ding and
                      Peng Wu and
                      Xiangnan He}, 
         title     = {Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach},  
         booktitle = {{NeurIPS}},  
         year      = {2023},   
   }
   ```


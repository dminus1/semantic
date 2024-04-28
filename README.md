**The code and data for "Fine-Tuning Language Models to Recognize Semantic Relations" by Dmitri Roussinov, Serge Sharoff, Nadezhda Puchnina. LREV 2023, which is the expanded version of our ECIR 2020 paper.**


Here, you can find the datasets and source files that we used in our experiments. 

The dataset files are in the folder “glue_data\MRPC”
Rename the dataset files used to train.tsv and dev.tsv accordingly.
To run: 
```bash
phython finetune_classifier-dyna-runs-bless.py
```
No command line parameters are needed.
If you are running another (not bless) dataset, you may need to change ‘num_classes’ accordingly and get_labels() if you are using more than 7 semantic classes.

While running and testing every 10 batches (as it is currently set up) will be too slow if the entire test set is used for that. So, I recommend using first 300 lines only or reduce the frequency of testing. (increasing 10 in the "if (batch_id + 1) % (10) == 0:" line)

Email us for the code and datasets for our ablaition studies if needed.



# GMM-HMM from scratch (letter writing sequence recognition)

This repository is a Python implementation for GMM-HMM model from scratch using Viterbi method. Also, we would use this model for recognizing letters from the sequence of the writing movements. The dataset is place in the `data` folder of the repository which includes writing sequence of 5 letters of `a`, `e`, `i`, `o`, `u`.



## Getting Started

### Installation

Clone the program.

`git clone https://github.com/raminnakhli/GMM-HMM-from-scratch.git`



### Prerequisites

The requirements are some common packages in machine learning. You can install them using below command.

`pip install -r requirement.txt`





## Execution

Now, you can run the experiments with default configuration using the below command.

`python main.py`

In addition, one can change configuration of the program using command-line flags while running the above command, which are explained in the following section.



## Controlling Flags

You can change configuration of the model using the below flags.

|   Short Format    |          Long Format          | Valid Values  |                         Explanation                          |
| :---------------: | :---------------------------: | :-----------: | :----------------------------------------------------------: |
|        -ht        |     --hyperparameter-test     |   No Value    | sets random values for state and mixture count, runs training for 5 times, and reports the best parameter |
|  -stpr Stop_Rate  |     --stop-rate Stop_Rate     |  Float Value  | specifies the stop difference criteria of the EM algorithm in GMM-HMM model |
| -stc State_Count  |   --state-count State_Count   | Integer Value |            specifies the number of Markov states             |
| -mc Mitxure_Count | --mixture-count Mitxure_Count | Integer Value |            specifies the number of mixture in GMM            |
|       -blk        |           --belkin            |   No Value    |       enables using Belkin method in GMM-HMM training        |
|       -vtb        |           --viterbi           |   No Value    |       enabled using Viterbi method in GMM-HMM training       |
|       -vft        |    --viterbi-forward-test     |   No Value    |   enabled comparison between  Viterbi and forward accuracy   |



## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Contact

Ramin Ebrahim Nakhli - raminnakhli@gmail.com

Project Link: https://github.com/raminnakhli/GMM-HMM-from-scratch


# Machine Learning Project

```bash
├── data
    └── clean_test_data.h5
    └── clean_validation_data.h5
    └── eyebrows_poisoned_data.h5
    └── sunglasses_poisoned_data.h5
├── fine_pruning
    ├── data
        └── clean_test_data.h5
        └── clean_validation_data.h5
        └── eyebrows_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
    ├── models
        └── anonymous_bd_net.h5
        └── anonymous_bd_weights.h5
        └── multi_trigger_multi_target_bd_net.h5
        └── multi_trigger_multi_target_bd_weights.h5
        └── pruned_anonymous_bd_net.h5
        └── pruned_multi_trigger_multi_target_bd_net.h5
        └── pruned_sunglasses_bd_net.h5
        └── sunglasses_bd_net.h5
        └── sunglasses_bd_weights.h5
    ├── prune_anonymous.py
    ├── prune_multitrgger.py
    └── prune_sunglasses.py
├── gangsweep
    ├── data
        └── clean_validation_data.h5
        └── clean_test_data.h5
        └── sunglasses_poisoned_data.h5
        └── eyebrows_poisoned_data.h5
    ├── models
        └── anonymous_bd_net.h5
        └── anonymous_bd_weights.h5
        └── sunglasses_bd_net.h5
        └── sunglasses_bd_weights.h5
        └── multi_trigger_multi_target_bd_net.h5
        └── multi_trigger_multi_target_bd_weights.h5
    └── train_gan.py
├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
├── neural_cleanse
    ├── data
        └── clean_test_data.h5
        └── clean_validation_data.h5
        └── eyebrows_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
    ├── models
        └── anonymous_bd_net.h5
        └── anonymous_bd_weights.h5
        └── multi_trigger_multi_target_bd_net.h5
        └── multi_trigger_multi_target_bd_weights.h5
        └── sunglasses_bd_net.h5
        └── sunglasses_bd_weights.h5
    ├── triggers
        └── anonymous.h5
        └── multitrigger.h5
        └── sunglasses.h5
    ├── prune_anonymous.py
    ├── prune_multitrgger.py
    ├── prune_sunglasses.py
    ├── train_trigger_anonymous.py
    ├── train_trigger_multitrigger.py
    ├── train_trigger_sunglasses.py
    ├── trigger_detection.py
    └── trigger_generation.py
├── retrain_with_pois
    ├── data
        └── clean_test_data.h5
        └── clean_validation_data.h5
        └── eyebrows_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
    ├── models
        └── anonymous_bd_net.h5
        └── anonymous_bd_weights.h5
        └── multi_trigger_multi_target_bd_net.h5
        └── multi_trigger_multi_target_bd_weights.h5
        └── retrain_with_pois_sunglasses_bd_net_temp.h5
        └── sunglasses_bd_net.h5
        └── sunglasses_bd_weights.h5
    └── retrain_with_pois.py
├── architecture.py
├── eval.py // this is the evaluation script
└── model_architecture.png // this is the architecture of the model
```

## Technical Details

### 1. Retraining with clean data + poisoned data

One of the simpler methods to mitigate a backdoor model if we have access to the known poisoned dataset is to retrain the model with the cleaned and poisoned data. Where the poisoned data is classified as N+1. This can only be applied to the sunglasses badnet in this case.

### 2. Neural Cleanse

In a normal Neural network without any backdoors, there is a minimum delta (Δ) required to misclassify a label as a different label. The aim of an attacker is to reduce this minimum delta (Δ) to classify the correct label to the targeted label. This is achieved by adding a trigger to the training data and training the model to classify images with the trigger to the targeted label. Our aim is to try to detect if there exists these minimum triggers to misclassify all other labels to the target label and if so to identify and reverse engineer these triggers.

### 3. Gangsweep

The reason for selecting Gangsweep approach is attackers and defenders are always trying to maximize and minimize the attack success rate of the backdoor in the network which fits very well with GANs philosophy. Here the generator is trying to generate masks which can maximize the attackers success rate. The discriminator is not parallely trained like in a traditional GAN, instead it is re-trained in the later stages of implementation.

### 4. Pruning and retraining with clean data

Another easy method to mitigate backdoors if fine pruning the model and retraining it with clean data in order to remove the triggers from the model. Every neuron has a certain activation when a prediction is run. In a backdoored neural network there will be a few neurons which will activate only for the backdoor trigger. We use fine pruning and a clean dataset to find these neurons and prune them out. This removes the backdoor from the model. This may not work on pruning aware attacks as these attacks are much more sophisticated and the neurons which activate for the backdoors are also the same neurons which activate for the clean dataset.

## REFERENCES

1. Liuwan Zhu, Rui Ning, Cong Wang, Chunsheng Xin and Hongyi Wu. 2020. GangSweep: Sweep out Neural Backdoors by GAN <https://www.lions.odu.edu/~h1wu/paper/gangsweep.pdf>
2. Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y Zhao. 2019. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In Advances in Neural Information Processing Systems. IEEE. Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks <https://sites.cs.ucsb.edu/~bolunwang/assets/docs/backdoor-sp19.pdf>
3. Fine Pruning and retraining with clean data <https://arxiv.org/pdf/1805.12185.pdf>
4. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. In Advances in neural information processing systems. 2672–2680.

# FedForgery: Generalized Face Forgery Detection with Residual Federated Learning

The zip file contains the source code we used in this paper to test the accuracy of face forgery detection in the Hybrid-domain forgery dataset.

## Dependencies

* Anaconda3 (Python3.6, with Numpy etc.)
* Pytorch 1.10.0
* tensorboardX

More details about dependencies are shown in requirements.txt.

## Datasets

[Hybrid-domain forgery dataset] Combine four diverse forgery subtypes of the FF++ dataset and the WildDeepfake dataset into the whole dataset with five different artifact types the training set contains 20,000 images where true images have the same number of fake images; the ratio of the training set and testing set is kept at 7: 3.]

## Usage

### Download pretrained model

| Model   | Download                                                     |
| ------- | ------------------------------------------------------------ |
| FedForgery | [MEGA](https://mega.nz/file/Ba0R1C4I#nRVi0u5Am9zuK_5TOvms8eCsYyxHyqLqwoj1aOgbH80) |

After downloading the pretrained model, we should put the model to `./pretrained`

### Download dataset

| Dataset Name | Download                                                   | Images  |
| ------------ | ---------------------------------------------------------- | ------- |
| Hybrid-domain forgery dataset | [Hybrid-domain forgery- dataset](https://mega.nz/file/9b9GGQqL#cfNu3PQ05Ssg68OHakK-h_Ghm97E2stD3vojmhNYxuU) | 4,2800 |

After downloading the whole dataset, you can unzip **test.zip** to the `./testset`.

### Test the model

```
./run_test.sh
```

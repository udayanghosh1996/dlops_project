from SimCLR import Classifier
import time


if __name__ == "__main__":
    st = time.time()

    clf = Classifier(100, -1)
    clf.pretext_train("CIFAR100",
                      epochs=100,
                      enc_lr=3e-5,
                      proj_lr=3e-4,
                      fine_tune_layers=0,
                      temperature=0.5,
                      batch_size=2048
                      )
    clf.load_pretexted_model()
    clf.fine_tuning(dataset_name="cifar100",
                    epochs=1,
                    clf_lr=3e-4,
                    batch_size=2048
                    )
    ed = time.time()
    print(f"Time taken in seconds = {ed - st}")


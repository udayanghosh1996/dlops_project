from models.SimCLR import Classifier
import time


if __name__ == "__main__":
    st = time.time()

    clf = Classifier(100)
    clf.pretext_train("CIFAR10",
                      epochs=2,
                      enc_lr=3e-4,
                      proj_lr=3e-4,
                      fine_tune_layers=1,
                      temperature=0.5,
                      batch_size=512)
    clf.load_pretexted_model()
    clf.fine_tuning(dataset_name="cifar100",
                    epochs=2,
                    enc_lr=3e-4,
                    proj_lr=3e-4,
                    clf_lr=3e-3,
                    batch_size=256,
                    base_enc_finetune_layers=0)
    ed = time.time()
    print(f"Time taken in seconds = {ed - st}")


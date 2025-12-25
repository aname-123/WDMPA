import csv
import os.path


def write_to_csv(save_dir,
                 epoch,
                 train_loss,
                 lr,
                 val_loss,
                 avg_error):
    csv_file = os.path.join(save_dir, "results.csv")

    fieldnames = ['epoch', 'train_loss', 'lr', 'val_loss', 'avg_error']
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': lr,
            'val_loss': val_loss,
            'avg_error': avg_error
        })


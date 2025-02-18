import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support

class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience  # How many epochs to wait before stopping
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta  # Minimum improvement to be considered as an improvement

    def __call__(self, current_score):
        # First epoch (initialize best score)
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement detected, reset counter and update best score
            self.best_score = current_score
            self.counter = 0


def train(opt, data_loader, model, visualizer):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0

    early_stopping = EarlyStopping(patience=10, delta=0.01)  # Set your patience and delta here

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_psnr = 0  # To track PSNR for this epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                results = model.get_current_visuals()
                psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
                print('PSNR on Train = %f' % psnrMetric)
                epoch_psnr += psnrMetric
                visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

        # Calculate average PSNR for this epoch
        avg_psnr = epoch_psnr / len(dataset)
        print(f'Epoch {epoch}: Average PSNR = {avg_psnr}')

        # Early stopping check
        early_stopping(avg_psnr)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

if __name__ == '__main__':
    freeze_support()
    opt = TrainOptions().parse()
    opt.learn_residual = True
    opt.resize_or_crop = "crop"
    opt.gan_type = "gan"

    default = 5000
    opt.save_latest_freq = 100
    default = 100
    opt.print_freq = 20

    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    train(opt, data_loader, model, visualizer)

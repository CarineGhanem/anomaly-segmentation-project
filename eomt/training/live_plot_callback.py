import lightning as L
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch

class LivePlotCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.val_miou = []
        self.epochs = []
        self.current_train_loss = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Collect training loss from each batch
        if 'loss' in outputs:
            self.current_train_loss.append(outputs['loss'].item())
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Average training loss for this epoch
        if self.current_train_loss:
            avg_train_loss = sum(self.current_train_loss) / len(self.current_train_loss)
            self.train_loss.append(avg_train_loss)
            self.current_train_loss = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation metrics
        metrics = trainer.callback_metrics
        
        val_miou = metrics.get("metrics/val_iou_all", None)
        if val_miou is not None:
            self.val_miou.append(float(val_miou) * 100)
            self.epochs.append(trainer.current_epoch)
        
        # Get validation loss if available
        val_loss = metrics.get("val_loss_total", None)
        if val_loss is not None:
            self.val_loss.append(float(val_loss))
        
        # Update plot
        self.plot_metrics()
    
    def plot_metrics(self):
        # Clear previous output and plot new one
        clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Training Loss
        if self.train_loss:
            axes[0].plot(range(len(self.train_loss)), self.train_loss, 
                        'b-o', linewidth=2, markersize=6)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Add value annotation on last point
            if len(self.train_loss) > 0:
                last_val = self.train_loss[-1]
                axes[0].annotate(f'{last_val:.4f}', 
                               xy=(len(self.train_loss)-1, last_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        # Plot 2: Validation Loss
        if self.val_loss:
            axes[1].plot(self.epochs, self.val_loss, 
                        'r-o', linewidth=2, markersize=6)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            if len(self.val_loss) > 0:
                last_val = self.val_loss[-1]
                axes[1].annotate(f'{last_val:.4f}', 
                               xy=(self.epochs[-1], last_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        # Plot 3: Validation mIoU
        if self.val_miou:
            axes[2].plot(self.epochs, self.val_miou, 
                        'g-o', linewidth=2, markersize=6, label='Val mIoU')
            axes[2].axhline(y=77.41, color='gray', linestyle='--', 
                          label='Baseline (77.41%)', alpha=0.7, linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('mIoU (%)', fontsize=12)
            axes[2].set_title('Validation mIoU', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([76, 82])
            
            if len(self.val_miou) > 0:
                last_val = self.val_miou[-1]
                axes[2].annotate(f'{last_val:.2f}%', 
                               xy=(self.epochs[-1], last_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        if self.epochs:
            print(f"\n{'='*60}")
            print(f"Epoch {self.epochs[-1]} Summary:")
            print(f"{'='*60}")
            if self.train_loss:
                print(f"  Train Loss: {self.train_loss[-1]:.4f}")
            if self.val_loss:
                print(f"  Val Loss:   {self.val_loss[-1]:.4f}")
            if self.val_miou:
                print(f"  Val mIoU:   {self.val_miou[-1]:.2f}%")
            print(f"{'='*60}\n")
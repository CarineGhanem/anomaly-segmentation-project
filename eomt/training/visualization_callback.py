# training/visualization_callback.py
"""
Real-time training visualization callback for Google Colab
Plots training/validation loss and mIoU per epoch
"""

import lightning as L
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


class RealtimeVisualizationCallback(L.Callback):
    """
    Callback to visualize training metrics in real-time during training.
    Updates plots after each epoch showing loss and mIoU progression.
    """
    
    def __init__(self, update_every_n_epochs=1):
        super().__init__()
        self.update_every_n_epochs = update_every_n_epochs
        
        # Storage for metrics
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.epochs = []
        
        # Loss components (for detailed view)
        self.train_loss_ce = []
        self.train_loss_mask = []
        self.train_loss_dice = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of training epoch"""
        # Get current epoch
        current_epoch = trainer.current_epoch
        
        # Get training loss from logged metrics
        if 'loss_total' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['loss_total'].item()
            self.train_losses.append(train_loss)
            
        # Get loss components if available
        if 'loss_ce' in trainer.callback_metrics:
            self.train_loss_ce.append(trainer.callback_metrics['loss_ce'].item())
        if 'loss_mask' in trainer.callback_metrics:
            self.train_loss_mask.append(trainer.callback_metrics['loss_mask'].item())
        if 'loss_dice' in trainer.callback_metrics:
            self.train_loss_dice.append(trainer.callback_metrics['loss_dice'].item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch"""
        current_epoch = trainer.current_epoch
        
        # Store epoch number
        self.epochs.append(current_epoch)
        
        # Get validation metrics
        if 'metrics/val_iou_all' in trainer.callback_metrics:
            val_miou = trainer.callback_metrics['metrics/val_iou_all'].item()
            self.val_mious.append(val_miou * 100)  # Convert to percentage
        
        # Get validation loss if available (might be in logged metrics)
        # Note: You may need to explicitly log val_loss in your validation_step
        if 'val_loss_total' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_loss_total'].item()
            self.val_losses.append(val_loss)
        
        # Update plot every N epochs
        if (current_epoch + 1) % self.update_every_n_epochs == 0:
            self.plot_metrics()
    
    def plot_metrics(self):
        """Create and display the plots"""
        clear_output(wait=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs_array = np.array(self.epochs)
        
        # Plot 1: Training Loss (Total)
        if self.train_losses:
            ax1 = axes[0, 0]
            ax1.plot(epochs_array[:len(self.train_losses)], self.train_losses, 
                    'b-o', linewidth=2, markersize=4, label='Train Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training Loss (Total)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add value annotation on last point
            if len(self.train_losses) > 0:
                last_loss = self.train_losses[-1]
                ax1.annotate(f'{last_loss:.3f}', 
                           xy=(epochs_array[len(self.train_losses)-1], last_loss),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='blue')
        
        # Plot 2: Loss Components (CE, Mask, Dice)
        if self.train_loss_ce:
            ax2 = axes[0, 1]
            epochs_comp = epochs_array[:len(self.train_loss_ce)]
            ax2.plot(epochs_comp, self.train_loss_ce, 'r-o', 
                    linewidth=2, markersize=4, label='Classification (CE)')
            if self.train_loss_mask:
                ax2.plot(epochs_comp, self.train_loss_mask, 'g-s', 
                        linewidth=2, markersize=4, label='Mask')
            if self.train_loss_dice:
                ax2.plot(epochs_comp, self.train_loss_dice, 'orange', 
                        marker='^', linewidth=2, markersize=4, label='Dice')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Highlight final CE loss (most important for LogitNorm)
            if len(self.train_loss_ce) > 0:
                last_ce = self.train_loss_ce[-1]
                ax2.annotate(f'CE: {last_ce:.4f}', 
                           xy=(epochs_comp[-1], last_ce),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='red', fontweight='bold')
        
        # Plot 3: Validation mIoU
        if self.val_mious:
            ax3 = axes[1, 0]
            ax3.plot(epochs_array, self.val_mious, 'g-o', 
                    linewidth=2, markersize=5, label='Val mIoU')
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('mIoU (%)', fontsize=12)
            ax3.set_title('Validation mIoU', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([min(self.val_mious) - 1, max(self.val_mious) + 1])
            ax3.legend()
            
            # Add best mIoU marker
            best_miou = max(self.val_mious)
            best_epoch = self.val_mious.index(best_miou)
            ax3.scatter([epochs_array[best_epoch]], [best_miou], 
                       color='red', s=100, zorder=5, marker='*', 
                       label=f'Best: {best_miou:.2f}%')
            ax3.legend()
            
            # Add value annotation
            if len(self.val_mious) > 0:
                last_miou = self.val_mious[-1]
                ax3.annotate(f'{last_miou:.2f}%', 
                           xy=(epochs_array[-1], last_miou),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='green', fontweight='bold')
        
        # Plot 4: Validation Loss (if available)
        ax4 = axes[1, 1]
        if self.val_losses:
            ax4.plot(epochs_array[:len(self.val_losses)], self.val_losses, 
                    'purple', marker='o', linewidth=2, markersize=5, 
                    label='Val Loss')
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Loss', fontsize=12)
            ax4.set_title('Validation Loss', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            # If no val loss, show training progress summary
            ax4.text(0.5, 0.5, 
                    f'Epoch: {epochs_array[-1] if len(epochs_array) > 0 else 0}\n'
                    f'Train Loss: {self.train_losses[-1]:.3f if self.train_losses else "N/A"}\n'
                    f'Val mIoU: {self.val_mious[-1]:.2f if self.val_mious else "N/A"}\n'
                    f'CE Loss: {self.train_loss_ce[-1]:.4f if self.train_loss_ce else "N/A"}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax4.transAxes,
                    fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('Training Summary', fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary to console
        if len(self.epochs) > 0:
            print(f"\n{'='*60}")
            print(f"Epoch {self.epochs[-1]} Summary:")
            print(f"{'='*60}")
            if self.train_loss_ce:
                print(f"  Train CE Loss: {self.train_loss_ce[-1]:.4f}")
            if self.train_losses:
                print(f"  Train Total Loss: {self.train_losses[-1]:.3f}")
            if self.val_mious:
                print(f"  Val mIoU: {self.val_mious[-1]:.2f}%")
                print(f"  Best mIoU: {max(self.val_mious):.2f}% (Epoch {self.val_mious.index(max(self.val_mious))})")
            print(f"{'='*60}\n")
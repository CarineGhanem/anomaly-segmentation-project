import lightning as L
from IPython.display import clear_output, display, HTML
import pandas as pd

class SimpleProgressCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.history = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        # Collect metrics
        epoch_data = {
            'Epoch': trainer.current_epoch,
            'Train Loss': metrics.get('loss_total', 0.0),
            'Val Loss': metrics.get('val_loss_total', 0.0),
            'Val mIoU (%)': metrics.get('metrics/val_iou_all', 0.0) * 100,
            'Loss CE': metrics.get('loss_ce', 0.0),
        }
        
        self.history.append(epoch_data)
        
        # Clear and display updated table
        clear_output(wait=True)
        
        df = pd.DataFrame(self.history)
        
        # Style the dataframe
        styled_df = df.style.format({
            'Train Loss': '{:.4f}',
            'Val Loss': '{:.4f}',
            'Val mIoU (%)': '{:.2f}',
            'Loss CE': '{:.4f}',
        }).set_properties(**{
            'text-align': 'center',
            'font-size': '12pt',
        }).set_table_styles([
            {'selector': 'th', 'props': [('font-size', '13pt'), ('font-weight', 'bold')]},
        ])
        
        # Highlight best mIoU
        if len(df) > 0:
            best_idx = df['Val mIoU (%)'].idxmax()
            styled_df = styled_df.apply(
                lambda x: ['background-color: lightgreen' if x.name == best_idx else '' for _ in x],
                axis=1
            )
        
        display(HTML("<h3>📊 Training Progress</h3>"))
        display(styled_df)
        
        # Print current status
        if self.history:
            latest = self.history[-1]
            print(f"\n✅ Latest: Epoch {latest['Epoch']} | "
                  f"mIoU: {latest['Val mIoU (%)']:.2f}% | "
                  f"Loss: {latest['Loss CE']:.4f}")
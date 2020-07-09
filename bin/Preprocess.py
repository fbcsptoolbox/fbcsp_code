import mne

class PreprocessKU:
    def __init__(self):
        self.selected_channels=['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']

    def select_channels(self,x_data,ch_names,selected_channels=[]):
        if not selected_channels:
            selected_channels = self.selected_channels
        selected_channels_idx = mne.pick_channels(ch_names, selected_channels,[])
        x_data_selected = x_data[:,selected_channels_idx,:].copy()
        return x_data_selected

